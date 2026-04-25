import argparse
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger
from tool_server.tool_workers.skills.registry import SKILL_REGISTRY
from tool_server.tool_workers.skills.normalizers import normalize_skill_params


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"SkillExecutor_worker_{worker_id}.log")


class SkillExecutorWorker(BaseToolWorker):
    """
    Generic runtime for hybrid skills.

    Responsibilities:
      1. Accept a skill_name
      2. Load the corresponding markdown skill spec
      3. Normalize / validate / enrich input arguments from registry
      4. Dispatch execution to the mapped executor worker
      5. Normalize and return the final result

    Important:
      - This worker should be the only skill-facing action exposed to the agent.
      - Concrete business logic remains inside underlying executor workers.
      - Skill-specific malformed argument compatibility should be handled by
        tool_server.tool_workers.skills.normalizers, not by app_v3.py.
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
    # Worker / dispatch helpers
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

    def _call_executor(
        self,
        model_name: str,
        payload: Dict[str, Any],
        timeout: int = 240,
    ) -> Dict[str, Any]:
        addr = self._resolve_tool_addr(model_name)

        ret = requests.post(
            addr + "/worker_generate",
            headers={"User-Agent": self.model_name},
            json=payload,
            timeout=timeout,
        )
        ret.raise_for_status()

        response = ret.json()
        if not isinstance(response, dict):
            return {
                "text": f"{model_name} returned non-dict response: {response}",
                "error_code": 1,
            }

        return response

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

    def _validate_and_prepare_payload(
        self,
        skill_name: str,
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Merge defaults, normalize skill-specific parameters, validate required inputs,
        and optionally filter unsupported fields.

        Normalization must happen before required-input validation, otherwise malformed
        but recoverable arguments such as [{"museum": {}}] cannot be converted into
        the canonical poi_specs format.
        """
        spec = self._get_skill_spec(skill_name)

        required_inputs = spec.get("required_inputs", [])
        optional_inputs = spec.get("optional_inputs", [])
        defaults = spec.get("defaults", {})
        allow_extra_args = spec.get("allow_extra_args", True)

        merged = dict(defaults)
        merged.update(params or {})

        merged, normalize_trace = normalize_skill_params(
            skill_name=skill_name,
            spec=spec,
            params=merged,
        )

        missing = [k for k in required_inputs if merged.get(k) in [None, ""]]
        if missing:
            raise ValueError(
                f"Missing required parameter(s) for {skill_name}: {', '.join(missing)}"
            )

        if allow_extra_args:
            return merged, normalize_trace

        allowed = set(required_inputs) | set(optional_inputs) | set(defaults.keys())
        filtered = {k: v for k, v in merged.items() if k in allowed}
        return filtered, normalize_trace

    # -----------------------------
    # Output normalization helpers
    # -----------------------------
    @staticmethod
    def _summarize_skill_doc(skill_doc: str, max_chars: int = 600) -> str:
        doc = (skill_doc or "").strip()
        doc = doc.replace("\n\n", "\n")
        return doc[:max_chars]

    @staticmethod
    def _resolve_existing_file(path: Any) -> Optional[str]:
        if not isinstance(path, str) or not path.strip():
            return None

        candidate_paths = [
            path,
            os.path.abspath(path),
            os.path.join(os.getcwd(), path),
        ]

        for candidate in candidate_paths:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

        return None

    @staticmethod
    def _infer_artifact_type(path: str, key: str = "") -> str:
        suffix = Path(path).suffix.lower()
        key = (key or "").lower()

        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            return "image"

        if suffix in {".tif", ".tiff"}:
            return "raster"

        if suffix in {".gpkg", ".geojson", ".shp", ".kml"}:
            return "vector"

        if suffix in {".csv", ".xlsx", ".xls", ".json"}:
            return "table"

        if suffix in {".txt", ".md", ".log"}:
            return "text"

        if key in {"image", "preview", "png", "primary_image"}:
            return "image"

        if key in {"gpkg", "primary_vector"}:
            return "vector"

        return "other"

    @classmethod
    def _make_artifact(
        cls,
        artifact_id: str,
        path: Any,
        source: str,
        artifact_type: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        resolved = cls._resolve_existing_file(path)
        if not resolved:
            return None

        final_type = artifact_type or cls._infer_artifact_type(
            path=resolved,
            key=artifact_id,
        )

        return {
            "id": artifact_id,
            "type": final_type,
            "path": resolved,
            "name": name or os.path.basename(resolved),
            "source": source,
        }

    @classmethod
    def _normalize_artifacts(
        cls,
        executor_resp: Dict[str, Any],
        executor_model: str,
    ) -> List[Dict[str, Any]]:
        """
        Convert both new-style explicit artifacts and legacy output fields into
        a unified artifacts list.

        Legacy supported fields:
          - image
          - output_path
          - out_file
          - path
          - preview
          - png
          - gpkg
        """
        artifacts: List[Dict[str, Any]] = []
        seen_paths = set()

        def add_artifact(item: Optional[Dict[str, Any]]):
            if not item:
                return

            path = item.get("path")
            if not path or path in seen_paths:
                return

            seen_paths.add(path)
            artifacts.append(item)

        raw_artifacts = executor_resp.get("artifacts")
        if isinstance(raw_artifacts, list):
            for idx, raw in enumerate(raw_artifacts, start=1):
                if not isinstance(raw, dict):
                    continue

                artifact_id = raw.get("id") or f"artifact_{idx}"
                path = raw.get("path") or raw.get("file") or raw.get("url")
                artifact_type = raw.get("type")
                name = raw.get("name")
                source = raw.get("source") or executor_model

                add_artifact(
                    cls._make_artifact(
                        artifact_id=artifact_id,
                        path=path,
                        source=source,
                        artifact_type=artifact_type,
                        name=name,
                    )
                )

        legacy_fields = [
            ("image", "primary_image"),
            ("output_path", "primary_output"),
            ("out_file", "primary_output"),
            ("path", "primary_output"),
            ("preview", "preview"),
            ("png", "primary_image"),
            ("gpkg", "primary_vector"),
        ]

        for key, artifact_id in legacy_fields:
            value = executor_resp.get(key)
            if isinstance(value, str):
                add_artifact(
                    cls._make_artifact(
                        artifact_id=artifact_id,
                        path=value,
                        source=executor_model,
                    )
                )

        return artifacts

    @staticmethod
    def _build_structured_output(executor_resp: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect common business fields into a stable structured object.
        The original fields are preserved at the top level for backward compatibility.
        """
        structured_keys = [
            "poi_summary",
            "total_count",
            "exists",
            "available_layers",
            "distance_computed",
            "distance_layer",
            "distances",
            "travel_times",
            "bbox",
            "area",
            "count",
            "measurement",
            "attributes",
            "changes",
            "environment_description",
            "render_warning",
        ]

        structured: Dict[str, Any] = {}

        for key in structured_keys:
            if key in executor_resp:
                structured[key] = executor_resp[key]

        old_structured = executor_resp.get("structured")
        if isinstance(old_structured, dict):
            structured.update(old_structured)

        return structured

    @classmethod
    def _normalize_output(
        cls,
        skill_name: str,
        executor_model: str,
        md_path: str,
        executor_resp: Dict[str, Any],
        skill_trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Return a stable SkillExecutor-level output schema.

        Canonical fields:
          - text
          - error_code
          - skill_name
          - executor_model
          - skill_md_path
          - skill_trace
          - artifacts
          - primary_image
          - primary_vector
          - download_path
          - structured

        Backward-compatible aliases:
          - image
          - gpkg
        """
        if not isinstance(executor_resp, dict):
            executor_resp = {
                "text": f"{executor_model} returned non-dict response: {executor_resp}",
                "error_code": 1,
            }

        prior_trace = executor_resp.get("skill_trace", [])
        if not isinstance(prior_trace, list):
            prior_trace = []

        text = executor_resp.get("text")
        if not text:
            text = f"Skill '{skill_name}' executed by '{executor_model}'."

        error_code = executor_resp.get("error_code", 0)
        try:
            error_code = int(error_code)
        except Exception:
            error_code = 1

        artifacts = cls._normalize_artifacts(
            executor_resp=executor_resp,
            executor_model=executor_model,
        )

        primary_image = None
        primary_vector = None

        for item in artifacts:
            item_type = item.get("type")
            if primary_image is None and item_type in {"image", "raster"}:
                primary_image = item.get("path")
            if primary_vector is None and item_type == "vector":
                primary_vector = item.get("path")

        download_path = (
            primary_image
            or primary_vector
            or (artifacts[0]["path"] if artifacts else None)
        )

        structured = cls._build_structured_output(executor_resp)

        result = dict(executor_resp)
        result.update(
            {
                "text": text,
                "error_code": error_code,
                "skill_name": skill_name,
                "executor_model": executor_model,
                "skill_md_path": md_path,
                "skill_trace": skill_trace + prior_trace,
                "artifacts": artifacts,
                "primary_image": primary_image,
                "primary_vector": primary_vector,
                "download_path": download_path,
                "structured": structured,
            }
        )

        # Backward-compatible aliases for current app logic.
        if primary_image and not result.get("image"):
            result["image"] = primary_image

        if primary_vector and not result.get("gpkg"):
            result["gpkg"] = primary_vector

        return result

    # -----------------------------
    # Main
    # -----------------------------
    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {
                "text": f"Invalid SkillExecutor params: expected dict, got {type(params).__name__}",
                "error_code": 2,
                "skill_trace": [],
            }

        skill_name = params.get("skill_name")
        if not skill_name:
            return {
                "text": "Missing required parameter: skill_name",
                "error_code": 2,
                "skill_trace": [],
            }

        skill_trace: List[Dict[str, Any]] = []

        try:
            spec = self._get_skill_spec(skill_name)
            md_path = spec["md_path"]
            executor_model = spec["executor_model"]

            # 1) Load markdown skill doc.
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

            # 2) Normalize and validate payload.
            executor_payload = {
                k: v for k, v in params.items() if k != "skill_name"
            }

            executor_payload, normalize_trace = self._validate_and_prepare_payload(
                skill_name=skill_name,
                params=executor_payload,
            )

            skill_trace.extend(normalize_trace)

            skill_trace.append(
                {
                    "stage": "validate_payload",
                    "status": "success",
                    "executor_model": executor_model,
                    "payload_keys": sorted(list(executor_payload.keys())),
                }
            )

            # 3) Dispatch executor.
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

            # 4) Normalize output.
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
                "artifacts": [],
                "primary_image": None,
                "primary_vector": None,
                "download_path": None,
                "structured": {},
            }

    def get_tool_instruction(self):
        skill_names = sorted(SKILL_REGISTRY.keys())

        return {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": (
                    "Execute a named skill defined in the skill library. "
                    "Pass skill_name plus the skill-specific arguments described in the Selected Skills section. "
                    "The runtime normalizes common malformed skill arguments, validates them, "
                    "then dispatches the request to the mapped executor worker. "
                    "Do not call concrete skill worker names directly when using selected skills."
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