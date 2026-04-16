"""
用途：
- 基于自然语言属性描述定位目标，例如“red car”“large ship”
- 对指定目标区域或整幅图像描述属性信息
- 统计整幅图像或指定区域内某类目标的数量
- 生成场景级图像描述
- 在已有数值结果时做简单数值计算

调用工具链：
- TextToBbox
- RegionAttributeDescription
- CountGivenObject
- ImageDescription
- Calculator
"""
import argparse
import ast
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import requests

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"TargetAttributeSkill_worker_{worker_id}.log")


class TargetAttributeSkillWorker(BaseToolWorker):
    """
    Composite skill for fine-grained attribute tasks.

    Supported task types:
      - filter: locate targets that match an attribute constraint, e.g. "red ship"
      - compare: compare an attribute between two selected targets,
                 e.g. smallest vs largest target
      - describe_and_count: describe the scene and count a specified target class
    """

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str = "auto",
        worker_id: str = worker_id,
        no_register: bool = False,
        model_name: str = "FineGrainedAttributeSkill",
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
    def _bbox_area(bbox: List[int]) -> int:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _bbox_to_str(bbox: List[int]) -> str:
        return f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"

    @staticmethod
    def _safe_extract_int(text: str) -> Optional[int]:
        vals = re.findall(r"-?\d+", text or "")
        if not vals:
            return None
        try:
            return int(vals[0])
        except Exception:
            return None

    @staticmethod
    def _normalize_attribute_text(text: str) -> str:
        t = (text or "").strip().lower()
        t = re.sub(r"\s+", " ", t)
        return t

    def _detect_by_query(self, image: str, query: str, top1: bool = False) -> Dict[str, Any]:
        resp = self._call_tool(
            "TextToBbox",
            {"image": image, "text": query, "top1": top1},
            timeout=180,
        )
        if resp.get("error_code") != 0:
            raise RuntimeError(f"TextToBbox failed: {resp.get('text', '')}")

        detections = self._parse_detections(resp.get("text", ""))
        return {
            "raw_response": resp,
            "detections": detections,
        }

    def _draw_boxes(
        self,
        image: str,
        detections: List[Dict[str, Any]],
        annotation_prefix: str,
        top_k: int = 10,
    ) -> Optional[str]:
        current_image = image
        kept = detections[: max(1, top_k)]

        for idx, det in enumerate(kept, start=1):
            bbox = self._bbox_to_str(det["bbox"])
            score = det.get("score")
            annotation = f"{annotation_prefix}_{idx}"
            if isinstance(score, (float, int)):
                annotation = f"{annotation}:{float(score):.2f}"

            resp = self._call_tool(
                "DrawBox",
                {"image": current_image, "bbox": bbox, "annotation": annotation},
                timeout=120,
            )
            if resp.get("error_code") != 0:
                return None
            current_image = resp.get("image", current_image)

        return current_image if current_image != image else None

    def _describe_attribute(self, image: str, bbox: List[int], attribute: str) -> str:
        resp = self._call_tool(
            "RegionAttributeDescription",
            {
                "image": image,
                "bbox": self._bbox_to_str(bbox),
                "attribute": attribute,
            },
            timeout=120,
        )
        if resp.get("error_code") != 0:
            raise RuntimeError(f"RegionAttributeDescription failed: {resp.get('text', '')}")
        return resp.get("text", "")

    # -----------------------------
    # Main
    # -----------------------------
    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image = params.get("image")
        task_type = str(params.get("task_type", "filter")).lower().strip()

        if not image:
            return {"text": "Missing required parameter: image", "error_code": 2}
        if not os.path.isfile(image):
            return {"text": f"Image not found: {image}", "error_code": 3}
        if task_type not in ["filter", "compare", "describe_and_count"]:
            return {"text": f"Unsupported task_type: {task_type}", "error_code": 2}

        skill_trace: List[Dict[str, Any]] = []

        try:
            # -------------------------------------------------
            # A) filter
            # -------------------------------------------------
            if task_type == "filter":
                target = params.get("target")
                attribute = params.get("attribute")
                attribute_value = params.get("attribute_value")
                visualize = bool(params.get("visualize", True))
                max_draw = int(params.get("max_draw", 10))
                top1 = bool(params.get("top1", False))

                if not target:
                    return {"text": "Missing required parameter: target", "error_code": 2}
                if not attribute_value and not attribute:
                    return {
                        "text": "For task_type='filter', provide at least one of: attribute_value, attribute",
                        "error_code": 2,
                    }

                # Prefer explicit attribute value in the query, fallback to attribute phrase
                if attribute_value:
                    query = f"{attribute_value} {target}"
                else:
                    query = f"{attribute} {target}"

                det = self._detect_by_query(image=image, query=query, top1=top1)
                skill_trace.append(
                    {
                        "tool": "TextToBbox",
                        "status": "success",
                        "text": det["raw_response"].get("text", ""),
                    }
                )

                detections = det["detections"]
                if not detections:
                    return {
                        "text": f"No target matching '{query}' was detected.",
                        "error_code": 0,
                        "detections": [],
                        "skill_trace": skill_trace,
                    }

                result = {
                    "text": f"Detected {len(detections)} target(s) matching '{query}'.",
                    "error_code": 0,
                    "detections": detections,
                    "skill_trace": skill_trace,
                }

                if visualize:
                    image_out = self._draw_boxes(
                        image=image,
                        detections=detections,
                        annotation_prefix=query.replace(" ", "_")[:16],
                        top_k=max_draw,
                    )
                    if image_out:
                        result["image"] = image_out

                return result

            # -------------------------------------------------
            # B) compare
            # -------------------------------------------------
            if task_type == "compare":
                target = params.get("target")
                attribute = params.get("attribute")
                compare_mode = str(params.get("compare_mode", "smallest_vs_largest")).lower().strip()
                selector_a = str(params.get("selector_a", "smallest")).lower().strip()
                selector_b = str(params.get("selector_b", "largest")).lower().strip()
                use_segmentation = bool(params.get("use_segmentation", False))

                if not target:
                    return {"text": "Missing required parameter: target", "error_code": 2}
                if not attribute:
                    return {"text": "Missing required parameter: attribute", "error_code": 2}

                # First detect all targets
                det = self._detect_by_query(image=image, query=target, top1=False)
                skill_trace.append(
                    {
                        "tool": "TextToBbox",
                        "status": "success",
                        "text": det["raw_response"].get("text", ""),
                    }
                )

                detections = det["detections"]
                if len(detections) < 2:
                    return {
                        "text": f"Need at least 2 '{target}' targets for comparison, but found {len(detections)}.",
                        "error_code": 1,
                        "detections": detections,
                        "skill_trace": skill_trace,
                    }

                # Default compare pattern
                if compare_mode == "smallest_vs_largest":
                    selector_a = "smallest"
                    selector_b = "largest"

                # Choose bbox-area ranking by default
                ranked = sorted(
                    detections,
                    key=lambda d: self._bbox_area(d["bbox"]),
                )

                def pick(selector: str) -> Dict[str, Any]:
                    if selector == "smallest":
                        return ranked[0]
                    if selector == "largest":
                        return ranked[-1]
                    if selector == "first":
                        return detections[0]
                    if selector == "last":
                        return detections[-1]
                    raise ValueError(f"Unsupported selector: {selector}")

                det_a = pick(selector_a)
                det_b = pick(selector_b)

                attr_a = self._describe_attribute(image=image, bbox=det_a["bbox"], attribute=attribute)
                skill_trace.append(
                    {
                        "tool": "RegionAttributeDescription",
                        "status": "success",
                        "text": attr_a,
                        "selector": selector_a,
                    }
                )

                attr_b = self._describe_attribute(image=image, bbox=det_b["bbox"], attribute=attribute)
                skill_trace.append(
                    {
                        "tool": "RegionAttributeDescription",
                        "status": "success",
                        "text": attr_b,
                        "selector": selector_b,
                    }
                )

                norm_a = self._normalize_attribute_text(attr_a)
                norm_b = self._normalize_attribute_text(attr_b)
                same = norm_a == norm_b

                result = {
                    "text": (
                        f"Compared attribute '{attribute}' between '{selector_a}' and '{selector_b}' {target}. "
                        f"Same attribute description: {same}."
                    ),
                    "error_code": 0,
                    "detections": detections,
                    "selected_pair": {
                        "selector_a": selector_a,
                        "selector_b": selector_b,
                        "bbox_a": det_a["bbox"],
                        "bbox_b": det_b["bbox"],
                    },
                    "attribute_a": attr_a,
                    "attribute_b": attr_b,
                    "same_attribute": same,
                    "skill_trace": skill_trace,
                }

                if params.get("visualize", True):
                    image_out = self._draw_boxes(
                        image=image,
                        detections=[det_a, det_b],
                        annotation_prefix=attribute.replace(" ", "_")[:16],
                        top_k=2,
                    )
                    if image_out:
                        result["image"] = image_out

                return result

            # -------------------------------------------------
            # C) describe_and_count
            # -------------------------------------------------
            target = params.get("target")
            bbox = params.get("bbox")

            scene_desc_resp = self._call_tool("ImageDescription", {"image": image}, timeout=120)
            skill_trace.append(
                {
                    "tool": "ImageDescription",
                    "status": "success" if scene_desc_resp.get("error_code") == 0 else "failed",
                    "text": scene_desc_resp.get("text", ""),
                }
            )
            if scene_desc_resp.get("error_code") != 0:
                return {
                    "text": f"ImageDescription failed: {scene_desc_resp.get('text', '')}",
                    "error_code": 1,
                    "skill_trace": skill_trace,
                }

            result: Dict[str, Any] = {
                "error_code": 0,
                "scene_description": scene_desc_resp.get("text", ""),
                "skill_trace": skill_trace,
            }

            count_text = None
            count_value = None
            if target:
                count_payload: Dict[str, Any] = {
                    "image": image,
                    "text": target,
                }
                if bbox:
                    count_payload["bbox"] = bbox

                count_resp = self._call_tool("CountGivenObject", count_payload, timeout=120)
                skill_trace.append(
                    {
                        "tool": "CountGivenObject",
                        "status": "success" if count_resp.get("error_code") == 0 else "failed",
                        "text": count_resp.get("text", ""),
                    }
                )
                if count_resp.get("error_code") == 0:
                    count_text = count_resp.get("text", "")
                    count_value = self._safe_extract_int(count_text)
                    result["count_target"] = target
                    result["count_value"] = count_value
                    result["count_raw_text"] = count_text

            if target and count_value is not None:
                result["text"] = (
                    f"Scene description completed. Estimated number of '{target}': {count_value}.\n\n"
                    f"{result['scene_description']}"
                )
            elif target and count_text is not None:
                result["text"] = (
                    f"Scene description completed. Count result for '{target}': {count_text}.\n\n"
                    f"{result['scene_description']}"
                )
            else:
                result["text"] = f"Scene description completed.\n\n{result['scene_description']}"

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
                    "Composite skill for fine-grained attribute tasks. "
                    "It supports attribute-based target filtering, attribute comparison "
                    "between selected targets, and scene description plus object counting."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "Path to the input image.",
                        },
                        "task_type": {
                            "type": "string",
                            "description": "One of: filter, compare, describe_and_count.",
                        },
                        "target": {
                            "type": "string",
                            "description": "Target object class, e.g. 'ship', 'aircraft', 'vehicle'.",
                        },
                        "attribute": {
                            "type": "string",
                            "description": "Attribute name or attribute phrase, e.g. 'color', 'orientation', 'damage state'.",
                        },
                        "attribute_value": {
                            "type": "string",
                            "description": "Desired attribute value for filtering, e.g. 'red', 'blue', 'docked'.",
                        },
                        "compare_mode": {
                            "type": "string",
                            "description": "Comparison preset. Currently supports 'smallest_vs_largest'.",
                        },
                        "selector_a": {
                            "type": "string",
                            "description": "Target selector for compare task: smallest, largest, first, last.",
                        },
                        "selector_b": {
                            "type": "string",
                            "description": "Target selector for compare task: smallest, largest, first, last.",
                        },
                        "bbox": {
                            "type": "string",
                            "description": "Optional region restriction in format '(x1,y1,x2,y2)' for describe_and_count.",
                        },
                        "top1": {
                            "type": "boolean",
                            "description": "If true, keep only the top detection for filter task.",
                        },
                        "visualize": {
                            "type": "boolean",
                            "description": "Whether to draw detected boxes and return annotated image.",
                        },
                        "max_draw": {
                            "type": "integer",
                            "description": "Maximum number of boxes to draw for filter task.",
                        },
                        "use_segmentation": {
                            "type": "boolean",
                            "description": "Reserved flag for future segmentation-based ranking.",
                        },
                    },
                    "required": ["image", "task_type"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20125)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="TargetAttributeSkill")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = TargetAttributeSkillWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()