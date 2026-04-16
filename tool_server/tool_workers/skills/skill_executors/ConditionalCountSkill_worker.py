"""
用途：
- 统计满足指定视觉条件的目标数量
- 支持先直接计数，再回退到检测框计数
- 可对部分检测结果做属性核验与可视化

调用工具链：
- CountGivenObject
- TextToBbox
- RegionAttributeDescription
- DrawBox
"""

import argparse
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import requests

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"ConditionalCountSkill_worker_{worker_id}.log")


class ConditionalCountSkillWorker(BaseToolWorker):
    """
    Composite skill for conditional target counting.

    Supported use cases:
      - Count docked ships
      - Count parked aircraft
      - Count targets facing a given direction
      - Count targets satisfying visually describable conditions

    Workflow:
      1. Try CountGivenObject with a composed text query
      2. Fallback to TextToBbox and count detections
      3. Optionally verify some detections using RegionAttributeDescription
      4. Optionally draw boxes for visualization
    """

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str = "auto",
        worker_id: str = worker_id,
        no_register: bool = False,
        model_name: str = "ConditionalCountSkill",
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
    def _safe_extract_int(text: str) -> Optional[int]:
        vals = re.findall(r"-?\d+", text or "")
        if not vals:
            return None
        try:
            return int(vals[0])
        except Exception:
            return None

    @staticmethod
    def _bbox_to_str(bbox: List[int]) -> str:
        return f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"

    @staticmethod
    def _build_query(target: str, condition: str) -> str:
        target = (target or "").strip()
        condition = (condition or "").strip()
        if condition and target:
            return f"{condition} {target}"
        if target:
            return target
        return condition

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

    def _verify_examples(
        self,
        image: str,
        detections: List[Dict[str, Any]],
        condition: str,
        max_verify: int,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for det in detections[: max(0, max_verify)]:
            resp = self._call_tool(
                "RegionAttributeDescription",
                {
                    "image": image,
                    "bbox": self._bbox_to_str(det["bbox"]),
                    "attribute": condition,
                },
                timeout=120,
            )
            results.append(
                {
                    "bbox": det["bbox"],
                    "error_code": resp.get("error_code", 1),
                    "text": resp.get("text", ""),
                }
            )
        return results

    # -----------------------------
    # Main
    # -----------------------------
    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image = params.get("image")
        target = params.get("target")
        condition = params.get("condition")
        bbox = params.get("bbox")

        if not image:
            return {"text": "Missing required parameter: image", "error_code": 2}
        if not target:
            return {"text": "Missing required parameter: target", "error_code": 2}
        if not condition:
            return {"text": "Missing required parameter: condition", "error_code": 2}
        if not os.path.isfile(image):
            return {"text": f"Image not found: {image}", "error_code": 3}

        query = self._build_query(target=target, condition=condition)
        visualize = bool(params.get("visualize", True))
        max_draw = int(params.get("max_draw", 10))
        verify_examples = bool(params.get("verify_examples", False))
        max_verify = int(params.get("max_verify", 3))

        skill_trace: List[Dict[str, Any]] = []

        try:
            # 1) Primary route: CountGivenObject
            count_payload: Dict[str, Any] = {
                "image": image,
                "text": query,
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

            count_value = None
            count_method = None
            detections: List[Dict[str, Any]] = []

            if count_resp.get("error_code") == 0:
                count_value = self._safe_extract_int(count_resp.get("text", ""))
                if count_value is not None and count_value >= 0:
                    count_method = "CountGivenObject"

            # 2) Fallback route: TextToBbox
            text_to_bbox_needed = count_value is None
            if text_to_bbox_needed or visualize or verify_examples:
                detect_payload: Dict[str, Any] = {
                    "image": image,
                    "text": query,
                    "top1": False,
                }
                detect_resp = self._call_tool("TextToBbox", detect_payload, timeout=180)
                skill_trace.append(
                    {
                        "tool": "TextToBbox",
                        "status": "success" if detect_resp.get("error_code") == 0 else "failed",
                        "text": detect_resp.get("text", ""),
                    }
                )

                if detect_resp.get("error_code") == 0:
                    detections = self._parse_detections(detect_resp.get("text", ""))
                    if count_value is None:
                        count_value = len(detections)
                        count_method = "TextToBbox"
                elif count_value is None:
                    return {
                        "text": (
                            f"Both counting routes failed for query '{query}'. "
                            f"CountGivenObject: {count_resp.get('text', '')}; "
                            f"TextToBbox: {detect_resp.get('text', '')}"
                        ),
                        "error_code": 1,
                        "skill_trace": skill_trace,
                    }

            # 3) Optional verification snippets
            verification_results = None
            if verify_examples and detections:
                verification_results = self._verify_examples(
                    image=image,
                    detections=detections,
                    condition=condition,
                    max_verify=max_verify,
                )
                skill_trace.append(
                    {
                        "tool": "RegionAttributeDescription",
                        "status": "success",
                        "text": f"Verified {len(verification_results)} example region(s).",
                    }
                )

            # 4) Optional visualization
            annotated_image = None
            if visualize and detections:
                annotated_image = self._draw_boxes(
                    image=image,
                    detections=detections,
                    annotation_prefix=query.replace(" ", "_")[:16],
                    top_k=max_draw,
                )

            result: Dict[str, Any] = {
                "text": (
                    f"Estimated number of '{query}' targets: {count_value}. "
                    f"Method used: {count_method}."
                ),
                "error_code": 0,
                "count": count_value,
                "query": query,
                "method_used": count_method,
                "skill_trace": skill_trace,
            }

            if detections:
                result["detections"] = detections
            if annotated_image:
                result["image"] = annotated_image
            if verification_results is not None:
                result["verification_examples"] = verification_results
                result["text"] += f" Verified {len(verification_results)} example region(s)."

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
                    "Composite skill for counting targets under a visual condition, "
                    "such as docked ships, parked aircraft, or targets facing a certain direction. "
                    "It first tries CountGivenObject, then falls back to TextToBbox-based counting, "
                    "and can optionally verify some detections using RegionAttributeDescription."
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
                            "description": "Target object class, e.g. 'ship', 'aircraft', 'vehicle'.",
                        },
                        "condition": {
                            "type": "string",
                            "description": "Visual condition, e.g. 'docked', 'parked', 'facing east'.",
                        },
                        "bbox": {
                            "type": "string",
                            "description": "Optional region restriction in format '(x1,y1,x2,y2)'.",
                        },
                        "visualize": {
                            "type": "boolean",
                            "description": "Whether to draw detected boxes and return annotated image.",
                        },
                        "max_draw": {
                            "type": "integer",
                            "description": "Maximum number of boxes to draw when visualize=true.",
                        },
                        "verify_examples": {
                            "type": "boolean",
                            "description": "Whether to verify a few detected examples using RegionAttributeDescription.",
                        },
                        "max_verify": {
                            "type": "integer",
                            "description": "Maximum number of detected regions to verify.",
                        },
                    },
                    "required": ["image", "target", "condition"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20128)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="ConditionalCountSkill")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = ConditionalCountSkillWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()
