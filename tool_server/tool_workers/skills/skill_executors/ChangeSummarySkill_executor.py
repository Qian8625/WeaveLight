"""
用途：
- 对两时相图像进行变化总结
- 针对新增船只、飞机转移或消失、设施损毁或扩建等任务生成标准化描述

调用工具链：
- ChangeDetection
"""

import argparse
import os
import uuid
from typing import Any, Dict, List, Optional

import requests

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"ChangeSummarySkill_worker_{worker_id}.log")


class ChangeSummarySkillWorker(BaseToolWorker):
    """
    Composite skill for text-level change summarization between two chronological images.

    Supported task types:
      - generic
      - new_ships
      - disappear_or_transfer_aircraft
      - facility_damage_or_expansion

    This skill wraps the existing ChangeDetection tool and standardizes prompts
    for common remote-sensing change questions.
    """

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str = "auto",
        worker_id: str = worker_id,
        no_register: bool = False,
        model_name: str = "ChangeSummarySkill",
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

    def _call_tool(self, tool_name: str, payload: Dict[str, Any], timeout: int = 180) -> Dict[str, Any]:
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
    def _build_change_query(task_type: str, user_query: Optional[str], target: Optional[str]) -> str:
        if user_query and str(user_query).strip():
            return str(user_query).strip()

        task_type = (task_type or "generic").strip().lower()
        target = (target or "").strip()

        templates = {
            "generic": "Describe the major changes between the two images in detail.",
            "new_ships": (
                "Identify newly added ships or vessels between the two images, "
                "especially in the harbor or port region, and summarize the evidence."
            ),
            "disappear_or_transfer_aircraft": (
                "Identify aircraft that disappeared, moved, or were transferred between the two images, "
                "and summarize the observed evidence."
            ),
            "facility_damage_or_expansion": (
                "Determine whether facilities or built structures were damaged, destroyed, expanded, "
                "or newly constructed between the two images, and explain the evidence."
            ),
        }

        if task_type in templates:
            return templates[task_type]

        if target:
            return f"Describe the major changes related to {target} between the two images in detail."

        return templates["generic"]

    @staticmethod
    def _build_summary_prefix(task_type: str) -> str:
        task_type = (task_type or "generic").strip().lower()
        if task_type == "new_ships":
            return "Ship change summary completed."
        if task_type == "disappear_or_transfer_aircraft":
            return "Aircraft change summary completed."
        if task_type == "facility_damage_or_expansion":
            return "Facility change summary completed."
        return "Change summary completed."

    # -----------------------------
    # Main
    # -----------------------------
    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pre_image = params.get("pre_image")
        post_image = params.get("post_image")
        task_type = str(params.get("task_type", "generic")).lower().strip()
        query = params.get("query")
        target = params.get("target")

        if not pre_image:
            return {"text": "Missing required parameter: pre_image", "error_code": 2}
        if not post_image:
            return {"text": "Missing required parameter: post_image", "error_code": 2}
        if not os.path.isfile(pre_image):
            return {"text": f"Pre-image not found: {pre_image}", "error_code": 3}
        if not os.path.isfile(post_image):
            return {"text": f"Post-image not found: {post_image}", "error_code": 3}

        if task_type not in [
            "generic",
            "new_ships",
            "disappear_or_transfer_aircraft",
            "facility_damage_or_expansion",
        ]:
            return {"text": f"Unsupported task_type: {task_type}", "error_code": 2}

        skill_trace: List[Dict[str, Any]] = []

        try:
            final_query = self._build_change_query(
                task_type=task_type,
                user_query=query,
                target=target,
            )

            change_resp = self._call_tool(
                "ChangeDetection",
                {
                    "pre_image": pre_image,
                    "post_image": post_image,
                    "text": final_query,
                },
                timeout=300,
            )

            skill_trace.append(
                {
                    "tool": "ChangeDetection",
                    "status": "success" if change_resp.get("error_code") == 0 else "failed",
                    "text": change_resp.get("text", ""),
                    "query": final_query,
                }
            )

            if change_resp.get("error_code") != 0:
                return {
                    "text": f"ChangeDetection failed: {change_resp.get('text', '')}",
                    "error_code": 1,
                    "query_used": final_query,
                    "skill_trace": skill_trace,
                }

            prefix = self._build_summary_prefix(task_type)

            return {
                "text": f"{prefix}\n\n{change_resp.get('text', '')}",
                "error_code": 0,
                "query_used": final_query,
                "task_type": task_type,
                "skill_trace": skill_trace,
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
                    "Composite skill for summarizing changes between two chronological images. "
                    "It standardizes common change-detection tasks such as newly added ships, "
                    "disappearing or transferred aircraft, and facility damage or expansion, "
                    "then calls the ChangeDetection tool."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pre_image": {
                            "type": "string",
                            "description": "Path to the earlier image.",
                        },
                        "post_image": {
                            "type": "string",
                            "description": "Path to the later image.",
                        },
                        "task_type": {
                            "type": "string",
                            "description": (
                                "One of: generic, new_ships, disappear_or_transfer_aircraft, "
                                "facility_damage_or_expansion."
                            ),
                        },
                        "query": {
                            "type": "string",
                            "description": (
                                "Optional custom change question. If provided, it overrides the default template."
                            ),
                        },
                        "target": {
                            "type": "string",
                            "description": "Optional target phrase used when task_type='generic'.",
                        },
                    },
                    "required": ["pre_image", "post_image"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20126)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="ChangeSummarySkill")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = ChangeSummarySkillWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()
