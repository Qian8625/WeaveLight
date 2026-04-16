"""
用途：
- 查找指定类别目标，并框选目标
- 分割目标，并计算目标面积
- 计算两组目标之间的真实距离

调用工具链：
- TextToBbox 或 ObjectDetection
- SegmentObjectPixels
- DrawBox
- AddText
- Calculator
"""
import argparse
import ast
import math
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"TargetLocateMeasureSkill_worker_{worker_id}.log")


class TargetLocateMeasureSkillWorker(BaseToolWorker):
    """
    First implementation of a composite skill for object locating / measuring tasks.

    Supported modes:
      - locate: detect target(s), optionally draw boxes
      - area: detect + segment + report pixel area, optionally convert to real area
      - segment: same as area, with segmentation-oriented wording
      - distance: detect target(s) and reference target(s), estimate center-to-center distance

    Notes:
      - Real distance / real area requires gsd_m_per_pixel.
      - This worker internally calls existing tools through the controller.
      - It prioritizes TextToBbox for target-specific queries.
    """

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str = "auto",
        worker_id: str = worker_id,
        no_register: bool = False,
        model_name: str = "TargetLocateMeasureSkill",
        limit_model_concurrency: int = 2,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        wait_timeout: float = 180.0,
        task_timeout: float = 180.0,
    ):
        self.tool_addr_cache: Dict[str, str] = {}
        super().__init__(
            controller_addr=controller_addr,
            worker_addr=worker_addr,
            worker_id=worker_id,
            no_register=no_register,
            model_name=model_name,
            limit_model_concurrency=limit_model_concurrency,
            host=host,
            port=port,
            wait_timeout=wait_timeout,
            task_timeout=task_timeout,
        )

    def init_model(self):
        logger.info(f"{self.model_name} initialized successfully.")

    # -----------------------------
    # Helpers
    # -----------------------------
    def _resolve_tool_addr(self, tool_name: str) -> str:
        cached = self.tool_addr_cache.get(tool_name)
        if cached:
            return cached

        ret = requests.post(
            self.controller_addr + "/get_worker_address",
            json={"model": tool_name},
            timeout=5,
        )
        ret.raise_for_status()
        addr = (ret.json().get("address") or "").strip()
        if not addr:
            raise RuntimeError(f"No worker address returned for tool '{tool_name}'")
        self.tool_addr_cache[tool_name] = addr
        return addr

    def _call_tool(self, tool_name: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
        addr = self._resolve_tool_addr(tool_name)
        ret = requests.post(
            addr + "/worker_generate",
            headers={"User-Agent": self.model_name},
            json=payload,
            timeout=timeout,
        )
        ret.raise_for_status()
        return ret.json()

    @staticmethod
    def _parse_detections(text: str) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        for line in (text or "").splitlines():
            line = line.strip()
            if not line:
                continue

            # Format A: (x1,y1,x2,y2), 0.87
            m_a = re.match(
                r"^\(([-+]?\d+),\s*([-+]?\d+),\s*([-+]?\d+),\s*([-+]?\d+)\),\s*([-+]?\d*\.?\d+)$",
                line,
            )
            if m_a:
                x1, y1, x2, y2, score = m_a.groups()
                detections.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "score": float(score),
                        "label": None,
                    }
                )
                continue

            # Format B: (x1, y1, x2, y2), label, score 0.87
            m_b = re.match(
                r"^\(([-+]?\d+),\s*([-+]?\d+),\s*([-+]?\d+),\s*([-+]?\d+)\),\s*(.*?),\s*score\s*([-+]?\d*\.?\d+)$",
                line,
            )
            if m_b:
                x1, y1, x2, y2, label, score = m_b.groups()
                detections.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "score": float(score),
                        "label": label.strip(),
                    }
                )
                continue
        return detections

    @staticmethod
    def _bbox_center(bbox: List[int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _bbox_area_pixels(bbox: List[int]) -> int:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _normalize_query_tokens(query: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9_]+", (query or "").lower())
        stopwords = {
            "the", "a", "an", "all", "of", "in", "on", "with", "and", "or",
            "large", "small", "big", "tiny", "largest", "smallest",
        }
        return [t for t in tokens if t not in stopwords]

    def _filter_object_detection_results(
        self,
        detections: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        query_tokens = self._normalize_query_tokens(query)
        if not query_tokens:
            return detections

        filtered = []
        for det in detections:
            label = (det.get("label") or "").lower()
            if not label:
                continue
            if any(tok in label for tok in query_tokens):
                filtered.append(det)
        return filtered

    def _detect_target(
        self,
        image: str,
        query: str,
        detector: str = "auto",
        top1: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        trace: List[Dict[str, Any]] = []
        errors: List[str] = []

        call_order: List[Tuple[str, Dict[str, Any], int]] = []
        if detector in ["auto", "text_to_bbox"]:
            call_order.append(
                (
                    "TextToBbox",
                    {"image": image, "text": query, "top1": top1},
                    120,
                )
            )
        if detector in ["auto", "object_detection"]:
            call_order.append(
                (
                    "ObjectDetection",
                    {"image": image},
                    180,
                )
            )

        for tool_name, payload, timeout in call_order:
            try:
                resp = self._call_tool(tool_name, payload, timeout=timeout)
                trace.append(
                    {
                        "tool": tool_name,
                        "status": "success" if resp.get("error_code") == 0 else "failed",
                        "text": resp.get("text", ""),
                    }
                )
                if resp.get("error_code") != 0:
                    errors.append(f"{tool_name}: {resp.get('text', '')}")
                    continue

                detections = self._parse_detections(resp.get("text", ""))
                if tool_name == "ObjectDetection":
                    detections = self._filter_object_detection_results(detections, query)

                if top1 and detections:
                    detections = detections[:1]

                if detections:
                    return detections, trace

                errors.append(f"{tool_name}: parse returned 0 detections")
            except Exception as e:
                trace.append({"tool": tool_name, "status": "failed", "text": str(e)})
                errors.append(f"{tool_name}: {e}")

        raise RuntimeError("; ".join(errors) if errors else f"No detections found for query '{query}'")

    def _draw_boxes(
        self,
        image: str,
        detections: List[Dict[str, Any]],
        annotation_prefix: str,
        top_k: int = 10,
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        current_image = image
        trace: List[Dict[str, Any]] = []
        kept = detections[: max(1, top_k)]

        for idx, det in enumerate(kept, start=1):
            bbox = det["bbox"]
            bbox_str = f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"
            score = det.get("score")
            annotation = f"{annotation_prefix}_{idx}"
            if isinstance(score, (float, int)):
                annotation = f"{annotation}:{float(score):.2f}"

            resp = self._call_tool(
                "DrawBox",
                {
                    "image": current_image,
                    "bbox": bbox_str,
                    "annotation": annotation,
                },
                timeout=120,
            )
            trace.append(
                {
                    "tool": "DrawBox",
                    "status": "success" if resp.get("error_code") == 0 else "failed",
                    "text": resp.get("text", ""),
                }
            )
            if resp.get("error_code") != 0:
                break
            current_image = resp.get("image", current_image)

        return (current_image if current_image != image else None), trace

    @staticmethod
    def _pairwise_min_distance(
        src_dets: List[Dict[str, Any]],
        tar_dets: List[Dict[str, Any]],
        same_set: bool,
    ) -> Dict[str, Any]:
        best: Optional[Dict[str, Any]] = None
        for i, s in enumerate(src_dets):
            for j, t in enumerate(tar_dets):
                if same_set and i >= j:
                    continue
                cx1, cy1 = TargetLocateMeasureSkillWorker._bbox_center(s["bbox"])
                cx2, cy2 = TargetLocateMeasureSkillWorker._bbox_center(t["bbox"])
                dist_px = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                candidate = {
                    "src_index": i,
                    "tar_index": j,
                    "src_bbox": s["bbox"],
                    "tar_bbox": t["bbox"],
                    "distance_px": dist_px,
                }
                if best is None or candidate["distance_px"] < best["distance_px"]:
                    best = candidate
        if best is None:
            raise RuntimeError("Unable to compute a valid pairwise distance.")
        return best

    @staticmethod
    def _parse_pixel_counts(text: str) -> List[int]:
        seg_text = (text or "").strip()
        if not seg_text:
            return []

        try:
            parsed = ast.literal_eval(seg_text)
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
            if isinstance(parsed, (int, float)):
                return [int(parsed)]
        except Exception:
            pass

        values = re.findall(r"\d+", seg_text)
        return [int(v) for v in values]

    # -----------------------------
    # Main
    # -----------------------------
    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image = params.get("image")
        target = params.get("target") or params.get("text")
        mode = str(params.get("mode", "locate")).lower().strip()
        reference_target = params.get("reference_target")
        gsd_m_per_pixel = params.get("gsd_m_per_pixel")
        visualize = bool(params.get("visualize", True))
        top1 = bool(params.get("top1", False))
        detector = params.get("detector", "auto")
        max_draw = int(params.get("max_draw", 10))

        if not image:
            return {"text": "Missing required parameter: image", "error_code": 2}
        if not target:
            return {"text": "Missing required parameter: target", "error_code": 2}
        if not os.path.isfile(image):
            return {"text": f"Image not found: {image}", "error_code": 3}
        if mode not in ["locate", "area", "segment", "distance"]:
            return {"text": f"Unsupported mode: {mode}", "error_code": 2}

        skill_trace: List[Dict[str, Any]] = []

        try:
            # 1) detect main target(s)
            target_dets, detect_trace = self._detect_target(
                image=image,
                query=target,
                detector=detector,
                top1=top1,
            )
            skill_trace.extend(detect_trace)

            result: Dict[str, Any] = {
                "error_code": 0,
                "detections": target_dets,
                "skill_trace": skill_trace,
            }

            # 2) mode-specific logic
            if mode in ["area", "segment"]:
                seg_resp = self._call_tool(
                    "SegmentObjectPixels",
                    {
                        "image": image,
                        "text": target,
                        "flag": True,
                    },
                    timeout=180,
                )
                skill_trace.append(
                    {
                        "tool": "SegmentObjectPixels",
                        "status": "success" if seg_resp.get("error_code") == 0 else "failed",
                        "text": seg_resp.get("text", ""),
                    }
                )
                if seg_resp.get("error_code") != 0:
                    return {
                        "text": f"SegmentObjectPixels failed: {seg_resp.get('text', '')}",
                        "error_code": 1,
                        "detections": target_dets,
                        "skill_trace": skill_trace,
                    }

                pixel_counts = self._parse_pixel_counts(seg_resp.get("text", ""))
                result["pixel_counts"] = pixel_counts
                result["bbox_pixel_areas"] = [
                    self._bbox_area_pixels(det["bbox"]) for det in target_dets
                ]

                if gsd_m_per_pixel is not None and pixel_counts:
                    gsd_val = float(gsd_m_per_pixel)
                    area_expression = f"({'+'.join(str(v) for v in pixel_counts)})*({gsd_val}**2)"
                    calc_resp = self._call_tool(
                        "Calculator",
                        {"expression": area_expression},
                        timeout=30,
                    )
                    skill_trace.append(
                        {
                            "tool": "Calculator",
                            "status": "success" if calc_resp.get("error_code") == 0 else "failed",
                            "text": calc_resp.get("text", ""),
                        }
                    )
                    if calc_resp.get("error_code") == 0:
                        try:
                            result["total_area_m2"] = float(calc_resp.get("text", "0"))
                        except Exception:
                            pass

                result["text"] = (
                    f"Detected {len(target_dets)} target(s) for '{target}'. "
                    f"Pixel counts: {pixel_counts}."
                )

            elif mode == "distance":
                ref_query = reference_target or target
                ref_dets, ref_trace = self._detect_target(
                    image=image,
                    query=ref_query,
                    detector=detector,
                    top1=False,
                )
                skill_trace.extend(ref_trace)
                same_set = ref_query == target
                best = self._pairwise_min_distance(target_dets, ref_dets, same_set=same_set)

                result["reference_detections"] = ref_dets
                result["distance_px"] = round(best["distance_px"], 3)
                result["closest_pair"] = best

                if gsd_m_per_pixel is not None:
                    gsd_val = float(gsd_m_per_pixel)
                    calc_resp = self._call_tool(
                        "Calculator",
                        {"expression": f"{best['distance_px']}*{gsd_val}"},
                        timeout=30,
                    )
                    skill_trace.append(
                        {
                            "tool": "Calculator",
                            "status": "success" if calc_resp.get("error_code") == 0 else "failed",
                            "text": calc_resp.get("text", ""),
                        }
                    )
                    if calc_resp.get("error_code") == 0:
                        try:
                            result["distance_m"] = float(calc_resp.get("text", "0"))
                        except Exception:
                            pass

                if "distance_m" in result:
                    result["text"] = (
                        f"Detected {len(target_dets)} source target(s) for '{target}' and "
                        f"{len(ref_dets)} reference target(s) for '{ref_query}'. "
                        f"Closest center-to-center distance is {result['distance_m']:.3f} meters."
                    )
                else:
                    result["text"] = (
                        f"Detected {len(target_dets)} source target(s) for '{target}' and "
                        f"{len(ref_dets)} reference target(s) for '{ref_query}'. "
                        f"Closest center-to-center distance is {result['distance_px']:.3f} pixels."
                    )

            else:  # locate
                result["text"] = f"Detected {len(target_dets)} target(s) for '{target}'."

            # 3) optional visualization
            if visualize and target_dets:
                visual_path, draw_trace = self._draw_boxes(
                    image=image,
                    detections=target_dets,
                    annotation_prefix=(target or "target").replace(" ", "_")[:16],
                    top_k=max_draw,
                )
                skill_trace.extend(draw_trace)
                if visual_path:
                    result["image"] = visual_path

            result["skill_trace"] = skill_trace
            return result

        except Exception as e:
            logger.exception(f"{self.model_name} failed: {e}")
            return {
                "text": f"{self.model_name} failed: {e}",
                "error_code": 1,
                "skill_trace": skill_trace,
            }

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": (
                    "Composite skill for target locating, pixel-area estimation, segmentation-oriented measurement, "
                    "and center-to-center distance estimation on a single image. It internally calls existing tools such as "
                    "TextToBbox, ObjectDetection, SegmentObjectPixels, DrawBox, and Calculator."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "Path to the input image.",
                        },
                        "target": {
                            "type": "string",
                            "description": "Natural-language description of the primary target, e.g. 'large ship' or 'red aircraft'.",
                        },
                        "mode": {
                            "type": "string",
                            "description": "One of: locate, area, segment, distance.",
                        },
                        "reference_target": {
                            "type": "string",
                            "description": "Optional reference target description for distance mode. If omitted, the skill computes same-class nearest distance.",
                        },
                        "gsd_m_per_pixel": {
                            "type": "number",
                            "description": "Optional ground sampling distance in meters per pixel. Required for real-world distance / area conversion.",
                        },
                        "top1": {
                            "type": "boolean",
                            "description": "If true, keep only the highest-confidence detection for the primary target query.",
                        },
                        "detector": {
                            "type": "string",
                            "description": "Detector preference: auto, text_to_bbox, or object_detection.",
                        },
                        "visualize": {
                            "type": "boolean",
                            "description": "Whether to draw detected boxes onto the image and return the annotated image path.",
                        },
                        "max_draw": {
                            "type": "integer",
                            "description": "Maximum number of boxes to draw when visualize=true.",
                        },
                    },
                    "required": ["image", "target", "mode"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20121)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="TargetLocateMeasureSkill")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = TargetLocateMeasureSkillWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()
