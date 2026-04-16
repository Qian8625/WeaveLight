import argparse
import os
import uuid
from typing import Any, Dict, List, Optional

import requests

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger
from tool_server.tool_workers.skills.registry import SKILL_REGISTRY


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"SkillExecutor_worker_{worker_id}.log")


class SkillExecutorWorker(BaseToolWorker):
    """
    Generic runtime for hybrid skills.

    Responsibilities:
      1. Accept a skill_name
      2. Load the corresponding markdown skill spec
      3. Validate / enrich input arguments from registry
      4. Dispatch execution to the mapped executor worker
      5. Normalize and return the final result

    Important:
      - This worker should be the only skill-facing action exposed to the agent.
      - Concrete business logic remains inside underlying executor workers.
    """

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str = "auto",
        worker_id: str = worker_id,
        no_register: bool = False,
        model_name: str = "SkillExecutor",
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
    def _resolve_tool_addr(self, model_name: str) -> str:
        cached = self.tool_addr_cache.get(model_name)
        if cached:
            return cached

        ret = requests.post(
            self.controller_addr + "/get_worker_address",
            json={"model": model_name},
            timeout=5,
        )
        ret.raise_for_status()
        addr = (ret.json().get("address") or "").strip()
        if not addr:
            raise RuntimeError(f"No worker address returned for model '{model_name}'")

        self.tool_addr_cache[model_name] = addr
        return addr

    def _call_executor(self, model_name: str, payload: Dict[str, Any], timeout: int = 240) -> Dict[str, Any]:
        addr = self._resolve_tool_addr(model_name)
        ret = requests.post(
            addr + "/worker_generate",
            headers={"User-Agent": self.model_name},
            json=payload,
            timeout=timeout,
        )
        ret.raise_for_status()
        return ret.json()

    def _load_skill_doc(self, md_path: str) -> str:
        candidate_paths = [
            md_path,
            os.path.abspath(md_path),
            os.path.join(os.getcwd(), md_path),
        ]
        for p in candidate_paths:
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
        raise FileNotFoundError(f"Skill markdown not found: {md_path}")

    def _get_skill_spec(self, skill_name: str) -> Dict[str, Any]:
        if skill_name not in SKILL_REGISTRY:
            raise KeyError(f"Unknown skill_name: {skill_name}")
        return SKILL_REGISTRY[skill_name]

    def _validate_and_prepare_payload(self, skill_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        spec = self._get_skill_spec(skill_name)

        required_inputs = spec.get("required_inputs", [])
        defaults = spec.get("defaults", {})
        optional_inputs = spec.get("optional_inputs", [])
        allow_extra_args = spec.get("allow_extra_args", True)

        merged = dict(defaults)
        merged.update(params)

        missing = [k for k in required_inputs if merged.get(k) in [None, ""]]
        if missing:
            raise ValueError(
                f"Missing required parameter(s) for {skill_name}: {', '.join(missing)}"
            )

        if allow_extra_args:
            return merged

        allowed = set(required_inputs) | set(optional_inputs) | set(defaults.keys())
        return {k: v for k, v in merged.items() if k in allowed}

    @staticmethod
    def _summarize_skill_doc(skill_doc: str, max_chars: int = 600) -> str:
        doc = (skill_doc or "").strip()
        doc = doc.replace("\n\n", "\n")
        return doc[:max_chars]

    @staticmethod
    def _normalize_output(
        skill_name: str,
        executor_model: str,
        md_path: str,
        executor_resp: Dict[str, Any],
        skill_trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        result = dict(executor_resp)
        prior_trace = result.get("skill_trace", [])
        if not isinstance(prior_trace, list):
            prior_trace = []

        result["skill_name"] = skill_name
        result["executor_model"] = executor_model
        result["skill_md_path"] = md_path
        result["skill_trace"] = skill_trace + prior_trace

        if "text" not in result or not result.get("text"):
            result["text"] = f"Skill '{skill_name}' executed by '{executor_model}'."

        if "error_code" not in result:
            result["error_code"] = 0

        return result

    # -----------------------------
    # Main
    # -----------------------------
    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        skill_name = params.get("skill_name")
        if not skill_name:
            return {"text": "Missing required parameter: skill_name", "error_code": 2}

        skill_trace: List[Dict[str, Any]] = []

        try:
            spec = self._get_skill_spec(skill_name)
            md_path = spec["md_path"]
            executor_model = spec["executor_model"]

            # 1) load markdown skill
            skill_doc = self._load_skill_doc(md_path)
            skill_trace.append(
                {
                    "stage": "load_skill_doc",
                    "status": "success",
                    "skill_name": skill_name,
                    "skill_md_path": md_path,
                    "doc_excerpt": self._summarize_skill_doc(skill_doc),
                }
            )

            # 2) prepare payload
            executor_payload = {
                k: v for k, v in params.items() if k != "skill_name"
            }
            executor_payload = self._validate_and_prepare_payload(
                skill_name=skill_name,
                params=executor_payload,
            )
            skill_trace.append(
                {
                    "stage": "validate_payload",
                    "status": "success",
                    "executor_model": executor_model,
                    "payload_keys": sorted(list(executor_payload.keys())),
                }
            )

            # 3) dispatch executor
            executor_resp = self._call_executor(
                model_name=executor_model,
                payload=executor_payload,
                timeout=int(spec.get("timeout_sec", 240)),
            )
            skill_trace.append(
                {
                    "stage": "dispatch_executor",
                    "status": "success" if executor_resp.get("error_code") == 0 else "failed",
                    "executor_model": executor_model,
                    "text": executor_resp.get("text", ""),
                }
            )

            # 4) normalize output
            return self._normalize_output(
                skill_name=skill_name,
                executor_model=executor_model,
                md_path=md_path,
                executor_resp=executor_resp,
                skill_trace=skill_trace,
            )

        except Exception as e:
            logger.exception(f"{self.model_name} failed: {e}")
            return {
                "text": f"{self.model_name} failed: {e}",
                "error_code": 1,
                "skill_name": skill_name,
                "skill_trace": skill_trace,
            }

    def get_tool_instruction(self):
        skill_names = sorted(SKILL_REGISTRY.keys())
        return {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": (
                    "Execute a named skill defined in the skill library. "
                    "The runtime loads the markdown skill specification, validates arguments, "
                    "then dispatches the request to the mapped executor worker."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": (
                                "Name of the skill to execute. "
                                f"Available skills: {', '.join(skill_names)}"
                            ),
                        }
                    },
                    "required": ["skill_name"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20130)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="SkillExecutor")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = SkillExecutorWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()