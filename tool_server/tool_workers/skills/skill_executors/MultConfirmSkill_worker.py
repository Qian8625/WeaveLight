"""
用途：
- 对 RGB 与 SAR 图像中的同类目标做跨模态确认
- 比较两种模态下的目标检测结果
- 融合 RGB 与 SAR 的目标证据，输出统一结论

调用工具链：
- TargetLocateMeasureSkill
- SARPreprocessing 或 SARToRGB
"""

import argparse
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"CrossModalConfirmationSkill_worker_{worker_id}.log")


class MultConfirmSkillWorker(BaseToolWorker):
    """
    Cross-modal confirmation / comparison / fusion skill.

    Workflow:
      1. Run TargetLocateMeasureSkill on RGB image
      2. Preprocess SAR image using SARPreprocessing or SARToRGB
      3. Run TargetLocateMeasureSkill on processed SAR image
      4. Match RGB detections and SAR detections by IoU
      5. Produce fused conclusion

    Assumption:
      - rgb_image and sar_image are roughly spatially aligned.
    """

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str = "auto",
        worker_id: str = worker_id,
        no_register: bool = False,
        model_name: str = "CrossModalConfirmationSkill",
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

    def _preprocess_sar(
        self,
        image: str,
        preprocess_mode: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        mode = (preprocess_mode or "sar_preprocess").strip().lower()

        if mode == "sar_preprocess":
            payload = {"image": image}
            if output_path:
                payload["output_path"] = output_path
            return self._call_tool("SARPreprocessing", payload, timeout=180)

        if mode == "sar_to_rgb":
            payload = {"image": image}
            if output_path:
                payload["output_path"] = output_path
            return self._call_tool("SARToRGB", payload, timeout=240)

        raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")

    @staticmethod
    def _bbox_iou(box_a: List[int], box_b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _match_detections(
        self,
        rgb_dets: List[Dict[str, Any]],
        sar_dets: List[Dict[str, Any]],
        iou_threshold: float,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        matched: List[Dict[str, Any]] = []
        used_sar = set()

        for rgb_idx, rgb_det in enumerate(rgb_dets):
            best_j = None
            best_iou = 0.0
            for sar_idx, sar_det in enumerate(sar_dets):
                if sar_idx in used_sar:
                    continue
                iou = self._bbox_iou(rgb_det["bbox"], sar_det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = sar_idx

            if best_j is not None and best_iou >= iou_threshold:
                used_sar.add(best_j)
                matched.append(
                    {
                        "rgb_index": rgb_idx,
                        "sar_index": best_j,
                        "iou": round(best_iou, 4),
                        "rgb_bbox": rgb_det["bbox"],
                        "sar_bbox": sar_dets[best_j]["bbox"],
                        "rgb_score": rgb_det.get("score"),
                        "sar_score": sar_dets[best_j].get("score"),
                    }
                )

        rgb_only = [
            {
                "rgb_index": i,
                "rgb_bbox": det["bbox"],
                "rgb_score": det.get("score"),
            }
            for i, det in enumerate(rgb_dets)
            if i not in {m["rgb_index"] for m in matched}
        ]
        sar_only = [
            {
                "sar_index": j,
                "sar_bbox": det["bbox"],
                "sar_score": det.get("score"),
            }
            for j, det in enumerate(sar_dets)
            if j not in {m["sar_index"] for m in matched}
        ]
        return matched, rgb_only, sar_only

    def _run_target_skill(
        self,
        image: str,
        target: str,
        detector: str,
        top1: bool,
        visualize: bool,
        max_draw: int,
    ) -> Dict[str, Any]:
        payload = {
            "image": image,
            "target": target,
            "mode": "locate",
            "detector": detector,
            "top1": top1,
            "visualize": visualize,
            "max_draw": max_draw,
        }
        return self._call_tool("TargetLocateMeasureSkill", payload, timeout=240)

    # -----------------------------
    # Main
    # -----------------------------
    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        rgb_image = params.get("rgb_image")
        sar_image = params.get("sar_image")
        target = params.get("target")
        task_type = str(params.get("task_type", "confirm")).lower().strip()

        if not rgb_image:
            return {"text": "Missing required parameter: rgb_image", "error_code": 2}
        if not sar_image:
            return {"text": "Missing required parameter: sar_image", "error_code": 2}
        if not target:
            return {"text": "Missing required parameter: target", "error_code": 2}
        if not os.path.isfile(rgb_image):
            return {"text": f"RGB image not found: {rgb_image}", "error_code": 3}
        if not os.path.isfile(sar_image):
            return {"text": f"SAR image not found: {sar_image}", "error_code": 3}
        if task_type not in ["confirm", "compare", "fuse"]:
            return {"text": f"Unsupported task_type: {task_type}", "error_code": 2}

        preprocess_mode = params.get("preprocess_mode", "sar_preprocess")
        detector = params.get("detector", "auto")
        top1 = bool(params.get("top1", False))
        visualize = bool(params.get("visualize", True))
        max_draw = int(params.get("max_draw", 10))
        iou_threshold = float(params.get("iou_threshold", 0.2))
        sar_preprocessed_output_path = params.get("sar_preprocessed_output_path")

        skill_trace: List[Dict[str, Any]] = []

        try:
            # 1) RGB detection
            rgb_resp = self._run_target_skill(
                image=rgb_image,
                target=target,
                detector=detector,
                top1=top1,
                visualize=visualize,
                max_draw=max_draw,
            )
            skill_trace.append(
                {
                    "tool": "TargetLocateMeasureSkill(rgb)",
                    "status": "success" if rgb_resp.get("error_code") == 0 else "failed",
                    "text": rgb_resp.get("text", ""),
                }
            )
            if rgb_resp.get("error_code") != 0:
                return {
                    "text": f"RGB branch failed: {rgb_resp.get('text', '')}",
                    "error_code": 1,
                    "skill_trace": skill_trace + rgb_resp.get("skill_trace", []),
                }

            rgb_dets = rgb_resp.get("detections", []) or []

            # 2) SAR preprocessing
            sar_pre_resp = self._preprocess_sar(
                image=sar_image,
                preprocess_mode=preprocess_mode,
                output_path=sar_preprocessed_output_path,
            )
            skill_trace.append(
                {
                    "tool": "SARPreprocessing" if preprocess_mode == "sar_preprocess" else "SARToRGB",
                    "status": "success" if sar_pre_resp.get("error_code") == 0 else "failed",
                    "text": sar_pre_resp.get("text", ""),
                }
            )
            if sar_pre_resp.get("error_code") != 0:
                return {
                    "text": f"SAR preprocessing failed: {sar_pre_resp.get('text', '')}",
                    "error_code": 1,
                    "rgb_detections": rgb_dets,
                    "skill_trace": skill_trace + rgb_resp.get("skill_trace", []),
                }

            processed_sar_image = (
                sar_pre_resp.get("image")
                or sar_pre_resp.get("output_path")
                or sar_pre_resp.get("path")
            )
            if not processed_sar_image or not os.path.isfile(processed_sar_image):
                return {
                    "text": "SAR preprocessing succeeded but no valid processed SAR image path was returned.",
                    "error_code": 1,
                    "rgb_detections": rgb_dets,
                    "skill_trace": skill_trace + rgb_resp.get("skill_trace", []),
                }

            # 3) SAR detection on processed image
            sar_resp = self._run_target_skill(
                image=processed_sar_image,
                target=target,
                detector=detector,
                top1=top1,
                visualize=visualize,
                max_draw=max_draw,
            )
            skill_trace.append(
                {
                    "tool": "TargetLocateMeasureSkill(sar_processed)",
                    "status": "success" if sar_resp.get("error_code") == 0 else "failed",
                    "text": sar_resp.get("text", ""),
                }
            )
            if sar_resp.get("error_code") != 0:
                return {
                    "text": f"SAR branch failed after preprocessing: {sar_resp.get('text', '')}",
                    "error_code": 1,
                    "rgb_detections": rgb_dets,
                    "preprocessed_sar_image": processed_sar_image,
                    "skill_trace": skill_trace + rgb_resp.get("skill_trace", []) + sar_resp.get("skill_trace", []),
                }

            sar_dets = sar_resp.get("detections", []) or []

            # 4) Cross-modal matching
            matched, rgb_only, sar_only = self._match_detections(
                rgb_dets=rgb_dets,
                sar_dets=sar_dets,
                iou_threshold=iou_threshold,
            )

            # 5) Final conclusion
            if task_type == "confirm":
                if matched:
                    conclusion = (
                        f"Cross-modal confirmation succeeded for '{target}'. "
                        f"{len(matched)} target(s) are supported by both RGB and SAR evidence."
                    )
                elif rgb_dets and not sar_dets:
                    conclusion = (
                        f"RGB suggests candidate '{target}' target(s), but SAR did not confirm them under the current threshold."
                    )
                elif sar_dets and not rgb_dets:
                    conclusion = (
                        f"SAR suggests candidate '{target}' target(s), while RGB did not provide matching evidence."
                    )
                else:
                    conclusion = f"No reliable '{target}' target was confirmed across both modalities."

            elif task_type == "compare":
                conclusion = (
                    f"RGB detected {len(rgb_dets)} candidate(s), SAR detected {len(sar_dets)} candidate(s), "
                    f"and {len(matched)} candidate(s) overlap across modalities."
                )

            else:  # fuse
                if matched:
                    conclusion = (
                        f"Fused conclusion: {len(matched)} '{target}' target(s) have consistent cross-modal evidence, "
                        f"with {len(rgb_only)} RGB-only candidate(s) and {len(sar_only)} SAR-only candidate(s)."
                    )
                else:
                    conclusion = (
                        f"Fused conclusion: no strong cross-modal consensus was found for '{target}'. "
                        f"RGB-only: {len(rgb_only)}, SAR-only: {len(sar_only)}."
                    )

            return {
                "text": conclusion,
                "error_code": 0,
                "rgb_detections": rgb_dets,
                "sar_detections": sar_dets,
                "matched_pairs": matched,
                "rgb_only": rgb_only,
                "sar_only": sar_only,
                "preprocessed_sar_image": processed_sar_image,
                "rgb_image_annotated": rgb_resp.get("image"),
                "sar_image_annotated": sar_resp.get("image"),
                "image": rgb_resp.get("image"),
                "skill_trace": skill_trace + rgb_resp.get("skill_trace", []) + sar_resp.get("skill_trace", []),
            }

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
                    "Cross-modal confirmation skill for aligned RGB and SAR imagery. "
                    "It detects the same target on RGB and SAR branches, compares their detections, "
                    "and returns matched targets, RGB-only candidates, SAR-only candidates, and a fused conclusion."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rgb_image": {
                            "type": "string",
                            "description": "Path to the RGB image.",
                        },
                        "sar_image": {
                            "type": "string",
                            "description": "Path to the SAR image.",
                        },
                        "target": {
                            "type": "string",
                            "description": "Natural-language target description, e.g. 'ship', 'airplane', or 'metal object'.",
                        },
                        "task_type": {
                            "type": "string",
                            "description": "One of: confirm, compare, fuse.",
                        },
                        "preprocess_mode": {
                            "type": "string",
                            "description": "SAR preprocessing route: sar_preprocess or sar_to_rgb.",
                        },
                        "sar_preprocessed_output_path": {
                            "type": "string",
                            "description": "Optional output path for the preprocessed SAR image.",
                        },
                        "iou_threshold": {
                            "type": "number",
                            "description": "IoU threshold for RGB/SAR cross-modal matching.",
                        },
                        "top1": {
                            "type": "boolean",
                            "description": "If true, keep only the highest-confidence detection in each branch.",
                        },
                        "detector": {
                            "type": "string",
                            "description": "Detector preference: auto, text_to_bbox, or object_detection.",
                        },
                        "visualize": {
                            "type": "boolean",
                            "description": "Whether to draw boxes on branch outputs.",
                        },
                        "max_draw": {
                            "type": "integer",
                            "description": "Maximum number of boxes to draw in each branch.",
                        },
                    },
                    "required": ["rgb_image", "sar_image", "target"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20123)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="MultConfirmSkill")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = MultConfirmSkillWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()
