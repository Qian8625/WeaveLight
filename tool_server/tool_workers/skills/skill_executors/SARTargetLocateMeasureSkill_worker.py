"""
用途：
- 对 SAR 图像先做预处理或伪彩色转换
- 在转换结果上执行目标发现、面积估计、距离计算

调用工具链：
- SARPreprocessing 或 SARToRGB
- TextToBbox 或 ObjectDetection
- SegmentObjectPixels
- Calculator
- DrawBox / AddText

"""
import argparse
import os
import uuid
from typing import Any, Dict, List, Optional

import requests

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"SARTargetLocateMeasureSkill_worker_{worker_id}.log")


class SARTargetLocateMeasureSkillWorker(BaseToolWorker):
    """
    Composite skill for SAR target locating / measuring tasks.

    Workflow:
      1. Preprocess SAR image using SARPreprocessing or SARToRGB
      2. Reuse TargetLocateMeasureSkill on the processed image

    Supported modes:
      - locate
      - area
      - segment
      - distance
    """

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str = "auto",
        worker_id: str = worker_id,
        no_register: bool = False,
        model_name: str = "SARTargetLocateMeasureSkill",
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

    # -----------------------------
    # Main
    # -----------------------------
    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image = params.get("image")
        target = params.get("target") or params.get("text")
        mode = str(params.get("mode", "locate")).lower().strip()

        if not image:
            return {"text": "Missing required parameter: image", "error_code": 2}
        if not target:
            return {"text": "Missing required parameter: target", "error_code": 2}
        if not os.path.isfile(image):
            return {"text": f"Image not found: {image}", "error_code": 3}
        if mode not in ["locate", "area", "segment", "distance"]:
            return {"text": f"Unsupported mode: {mode}", "error_code": 2}

        preprocess_mode = params.get("preprocess_mode", "sar_preprocess")
        preprocessed_output_path = params.get("preprocessed_output_path")

        skill_trace: List[Dict[str, Any]] = []

        try:
            # 1) SAR preprocessing
            preprocess_resp = self._preprocess_sar(
                image=image,
                preprocess_mode=preprocess_mode,
                output_path=preprocessed_output_path,
            )

            skill_trace.append(
                {
                    "tool": "SARPreprocessing" if preprocess_mode == "sar_preprocess" else "SARToRGB",
                    "status": "success" if preprocess_resp.get("error_code") == 0 else "failed",
                    "text": preprocess_resp.get("text", ""),
                }
            )

            if preprocess_resp.get("error_code") != 0:
                return {
                    "text": f"SAR preprocessing failed: {preprocess_resp.get('text', '')}",
                    "error_code": 1,
                    "skill_trace": skill_trace,
                }

            processed_image = (
                preprocess_resp.get("image")
                or preprocess_resp.get("output_path")
                or preprocess_resp.get("path")
            )
            if not processed_image or not os.path.isfile(processed_image):
                return {
                    "text": "SAR preprocessing succeeded but no valid processed image path was returned.",
                    "error_code": 1,
                    "skill_trace": skill_trace,
                }

            # 2) Delegate to TargetLocateMeasureSkill
            downstream_payload: Dict[str, Any] = {
                "image": processed_image,
                "target": target,
                "mode": mode,
                "visualize": params.get("visualize", True),
                "top1": params.get("top1", False),
                "detector": params.get("detector", "auto"),
                "max_draw": params.get("max_draw", 10),
            }

            if params.get("reference_target") is not None:
                downstream_payload["reference_target"] = params.get("reference_target")
            if params.get("gsd_m_per_pixel") is not None:
                downstream_payload["gsd_m_per_pixel"] = params.get("gsd_m_per_pixel")

            downstream_resp = self._call_tool(
                "TargetLocateMeasureSkill",
                downstream_payload,
                timeout=240,
            )

            skill_trace.append(
                {
                    "tool": "TargetLocateMeasureSkill",
                    "status": "success" if downstream_resp.get("error_code") == 0 else "failed",
                    "text": downstream_resp.get("text", ""),
                }
            )

            if downstream_resp.get("error_code") != 0:
                return {
                    "text": f"TargetLocateMeasureSkill failed after SAR preprocessing: {downstream_resp.get('text', '')}",
                    "error_code": 1,
                    "preprocessed_image": processed_image,
                    "skill_trace": skill_trace + downstream_resp.get("skill_trace", []),
                }

            result: Dict[str, Any] = {
                "text": (
                    f"SAR workflow completed using '{preprocess_mode}'. "
                    f"{downstream_resp.get('text', '')}"
                ).strip(),
                "error_code": 0,
                "preprocessed_image": processed_image,
                "skill_trace": skill_trace + downstream_resp.get("skill_trace", []),
            }

            # passthrough fields
            for key in [
                "image",
                "detections",
                "reference_detections",
                "pixel_counts",
                "bbox_pixel_areas",
                "distance_px",
                "distance_m",
                "closest_pair",
                "total_area_m2",
            ]:
                if key in downstream_resp:
                    result[key] = downstream_resp[key]

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
                    "Composite skill for SAR images. It first preprocesses the SAR image "
                    "using SARPreprocessing or SARToRGB, then reuses TargetLocateMeasureSkill "
                    "to perform locating, area estimation, segmentation-oriented measurement, "
                    "or center-to-center distance estimation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "Path to the input SAR image.",
                        },
                        "target": {
                            "type": "string",
                            "description": "Natural-language description of the primary target, e.g. 'ship' or 'aircraft'.",
                        },
                        "mode": {
                            "type": "string",
                            "description": "One of: locate, area, segment, distance.",
                        },
                        "reference_target": {
                            "type": "string",
                            "description": "Optional reference target description for distance mode.",
                        },
                        "gsd_m_per_pixel": {
                            "type": "number",
                            "description": "Optional ground sampling distance in meters per pixel.",
                        },
                        "preprocess_mode": {
                            "type": "string",
                            "description": "SAR preprocessing route: sar_preprocess or sar_to_rgb.",
                        },
                        "preprocessed_output_path": {
                            "type": "string",
                            "description": "Optional output path for the preprocessed SAR result.",
                        },
                        "top1": {
                            "type": "boolean",
                            "description": "If true, keep only the highest-confidence detection for the primary target query.",
                        },
                        "detector": {
                            "type": "string",
                            "description": "Detector preference passed downstream: auto, text_to_bbox, or object_detection.",
                        },
                        "visualize": {
                            "type": "boolean",
                            "description": "Whether to draw detected boxes and return the annotated image path.",
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
    parser.add_argument("--port", type=int, default=20122)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="SARTargetLocateMeasureSkill")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = SARTargetLocateMeasureSkillWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()