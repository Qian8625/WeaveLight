"""
用途：
- 从 GeoTIFF 提取空间范围并构建 AOI
- 查询并加载一类或多类 POI
- 支持 POI 可视化、存在性判断、数量统计和周边描述

调用工具链：
- GetBboxFromGeotiff
- GetAreaBoundary
- AddPoisLayer
- DisplayOnGeotiff
- ImageDescription
"""

import argparse
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import requests

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.tool_workers.offline_workers.GetBboxFromGeotiff_worker import (
    generate as get_bbox_generate,
)
from tool_server.tool_workers.offline_workers.DisplayOnGeotiff_worker import (
    generate as display_on_geotiff_generate,
)
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"GeoTIFFPoiExploreSkill_worker_{worker_id}.log")


class GeoTIFFPoiExploreSkillWorker(BaseToolWorker):
    """
    Composite skill for GeoTIFF POI exploration tasks.

    Supported task types:
      - visualize: visualize one or more POI layers on a GeoTIFF
      - existence: determine whether specified POIs exist in the GeoTIFF AOI
      - count: count POIs for one or more classes
      - surrounding_description: visualize POIs, then describe the rendered result

    Workflow:
      1. Get bbox from GeoTIFF
      2. Create AOI gpkg from bbox via GetAreaBoundary
      3. Add one or more POI layers via AddPoisLayer
      4. Optionally render layers on GeoTIFF via DisplayOnGeotiff
      5. Optionally describe rendered image via ImageDescription
    """

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str = "auto",
        worker_id: str = worker_id,
        no_register: bool = False,
        model_name: str = "GeoTIFFPoiExploreSkill",
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
    def _parse_bbox_from_text(text: str) -> List[float]:
        m = re.search(
            r"Bbox\s*\(west,\s*south,\s*east,\s*north\):\s*\(\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*\)",
            text,
        )
        if not m:
            raise ValueError(f"Unable to parse bbox from text: {text}")
        return [float(m.group(i)) for i in range(1, 5)]

    @staticmethod
    def _sanitize_layer_name(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip().lower())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned[:30] or "poi_layer"

    def _normalize_poi_specs(self, poi_specs: Any) -> List[Dict[str, Any]]:
        if not isinstance(poi_specs, list) or len(poi_specs) == 0:
            raise ValueError("poi_specs must be a non-empty list.")

        normalized: List[Dict[str, Any]] = []

        for idx, item in enumerate(poi_specs):
            if not isinstance(item, dict):
                raise ValueError(f"poi_specs[{idx}] must be an object.")

            query = item.get("query")
            if query is None:
                raise ValueError(f"poi_specs[{idx}] missing required field 'query'.")

            layer_name = item.get("layer_name")
            if not layer_name:
                if isinstance(query, str):
                    layer_name = self._sanitize_layer_name(query)
                elif isinstance(query, dict) and len(query) == 1:
                    k, v = next(iter(query.items()))
                    layer_name = self._sanitize_layer_name(f"{k}_{v}")
                else:
                    layer_name = f"poi_layer_{idx + 1}"

            normalized.append(
                {
                    "query": query,
                    "layer_name": layer_name,
                }
            )

        return normalized

    @staticmethod
    def _extract_integer_candidates(text: str) -> List[int]:
        return [int(x) for x in re.findall(r"\d+", text or "")]

    # -----------------------------
    # Main
    # -----------------------------
    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        geotiff = params.get("geotiff")
        poi_specs = params.get("poi_specs")
        task_type = str(params.get("task_type", "visualize")).lower().strip()

        if not geotiff:
            return {"text": "Missing required parameter: geotiff", "error_code": 2}
        if not poi_specs:
            return {"text": "Missing required parameter: poi_specs", "error_code": 2}
        if not os.path.isfile(geotiff):
            return {"text": f"GeoTIFF not found: {geotiff}", "error_code": 3}
        if task_type not in ["visualize", "existence", "count", "surrounding_description"]:
            return {"text": f"Unsupported task_type: {task_type}", "error_code": 2}

        buffer_m = params.get("buffer_m")
        show_names = bool(params.get("show_names", True))
        describe_rendered = bool(params.get("describe_rendered", task_type == "surrounding_description"))

        skill_trace: List[Dict[str, Any]] = []
        current_gpkg: Optional[str] = None
        rendered_image: Optional[str] = None

        try:
            # 1) bbox from geotiff
            bbox_resp = get_bbox_generate({"geotiff": geotiff})
            skill_trace.append(
                {
                    "tool": "GetBboxFromGeotiff",
                    "status": "success" if bbox_resp.get("error_code") == 0 else "failed",
                    "text": bbox_resp.get("text", ""),
                }
            )
            if bbox_resp.get("error_code") != 0:
                return {
                    "text": f"GetBboxFromGeotiff failed: {bbox_resp.get('text', '')}",
                    "error_code": 1,
                    "skill_trace": skill_trace,
                }

            bbox = self._parse_bbox_from_text(bbox_resp.get("text", ""))
            area_str = f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"

            # 2) AOI boundary
            boundary_payload: Dict[str, Any] = {"area": area_str}
            if buffer_m is not None:
                boundary_payload["buffer_m"] = buffer_m

            boundary_resp = self._call_tool("GetAreaBoundary", boundary_payload, timeout=180)
            skill_trace.append(
                {
                    "tool": "GetAreaBoundary",
                    "status": "success" if boundary_resp.get("error_code") == 0 else "failed",
                    "text": boundary_resp.get("text", ""),
                }
            )
            if boundary_resp.get("error_code") != 0 or not boundary_resp.get("gpkg"):
                return {
                    "text": f"GetAreaBoundary failed: {boundary_resp.get('text', '')}",
                    "error_code": 1,
                    "skill_trace": skill_trace,
                }

            current_gpkg = boundary_resp["gpkg"]

            # 3) add poi layers
            normalized_specs = self._normalize_poi_specs(poi_specs)
            poi_summary: List[Dict[str, Any]] = []
            added_layers: List[str] = []

            for spec in normalized_specs:
                add_resp = self._call_tool(
                    "AddPoisLayer",
                    {
                        "gpkg": current_gpkg,
                        "query": spec["query"],
                        "layer_name": spec["layer_name"],
                    },
                    timeout=180,
                )
                skill_trace.append(
                    {
                        "tool": "AddPoisLayer",
                        "status": "success" if add_resp.get("error_code") == 0 else "failed",
                        "layer_name": spec["layer_name"],
                        "text": add_resp.get("text", ""),
                    }
                )
                if add_resp.get("error_code") != 0:
                    return {
                        "text": f"AddPoisLayer failed for layer '{spec['layer_name']}': {add_resp.get('text', '')}",
                        "error_code": 1,
                        "gpkg": current_gpkg,
                        "skill_trace": skill_trace,
                    }

                nums = self._extract_integer_candidates(add_resp.get("text", ""))
                count_guess = nums[-1] if nums else 0

                poi_summary.append(
                    {
                        "layer_name": spec["layer_name"],
                        "query": spec["query"],
                        "count": count_guess,
                    }
                )
                added_layers.append(spec["layer_name"])

            # 4) Optionally render
            if task_type in ["visualize", "surrounding_description"]:
                display_resp = display_on_geotiff_generate(
                    {
                        "gpkg": current_gpkg,
                        "layers": added_layers,
                        "geotiff": geotiff,
                        "show_names": show_names,
                    }
                )
                skill_trace.append(
                    {
                        "tool": "DisplayOnGeotiff",
                        "status": "success" if display_resp.get("error_code") == 0 else "failed",
                        "text": display_resp.get("text", ""),
                    }
                )
                if display_resp.get("error_code") != 0:
                    return {
                        "text": f"DisplayOnGeotiff failed: {display_resp.get('text', '')}",
                        "error_code": 1,
                        "gpkg": current_gpkg,
                        "poi_summary": poi_summary,
                        "skill_trace": skill_trace,
                    }
                rendered_image = display_resp.get("image")

            # 5) Optional surrounding description
            environment_description = None
            if describe_rendered and rendered_image and os.path.isfile(rendered_image):
                desc_resp = self._call_tool(
                    "ImageDescription",
                    {"image": rendered_image},
                    timeout=120,
                )
                skill_trace.append(
                    {
                        "tool": "ImageDescription",
                        "status": "success" if desc_resp.get("error_code") == 0 else "failed",
                        "text": desc_resp.get("text", ""),
                    }
                )
                if desc_resp.get("error_code") == 0:
                    environment_description = desc_resp.get("text", "")

            # 6) Build final response
            total_count = sum(item["count"] for item in poi_summary)
            existence_answer = all(item["count"] > 0 for item in poi_summary)

            if task_type == "visualize":
                text = (
                    f"GeoTIFF POI visualization completed successfully. "
                    f"Added {len(added_layers)} layer(s) with an estimated total of {total_count} POIs."
                )
            elif task_type == "existence":
                text = (
                    f"Existence check completed. "
                    f"All requested POI classes exist in the AOI: {existence_answer}."
                )
            elif task_type == "count":
                text = (
                    f"POI counting completed successfully. "
                    f"Estimated total count across requested layers: {total_count}."
                )
            else:  # surrounding_description
                text = (
                    f"GeoTIFF POI exploration and rendered-scene description completed successfully. "
                    f"Estimated total count across requested layers: {total_count}."
                )

            result: Dict[str, Any] = {
                "text": text,
                "error_code": 0,
                "gpkg": current_gpkg,
                "poi_summary": poi_summary,
                "total_count": total_count,
                "exists": existence_answer,
                "skill_trace": skill_trace,
            }

            if rendered_image:
                result["image"] = rendered_image
            if environment_description:
                result["environment_description"] = environment_description
                result["text"] = text + "\n\n" + environment_description

            return result

        except Exception as e:
            logger.exception(f"{self.model_name} failed: {e}")
            return {
                "text": f"{self.model_name} failed: {e}",
                "error_code": 1,
                "gpkg": current_gpkg,
                "skill_trace": skill_trace,
            }

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": (
                    "Composite skill for GeoTIFF-based POI exploration. It extracts the AOI from a GeoTIFF, "
                    "retrieves one or more POI classes, optionally renders them on the GeoTIFF, "
                    "checks existence, estimates counts, and can describe the rendered surrounding environment."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "geotiff": {
                            "type": "string",
                            "description": "Path to the input GeoTIFF file.",
                        },
                        "poi_specs": {
                            "type": "array",
                            "description": (
                                "A non-empty list of POI specs. "
                                "Each item should include query and optionally layer_name."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "description": (
                                            "OSM query dict or single POI name string. "
                                            "Examples: {'amenity':'hospital'} or 'Edinburgh Castle'"
                                        )
                                    },
                                    "layer_name": {
                                        "type": "string",
                                        "description": "Layer name to save in the GeoPackage.",
                                    },
                                },
                                "required": ["query"],
                            },
                        },
                        "task_type": {
                            "type": "string",
                            "description": "One of: visualize, existence, count, surrounding_description.",
                        },
                        "buffer_m": {
                            "type": "number",
                            "description": "Optional buffer distance in meters around the bbox-derived AOI.",
                        },
                        "show_names": {
                            "type": "boolean",
                            "description": "Whether to render feature names on the output GeoTIFF.",
                        },
                        "describe_rendered": {
                            "type": "boolean",
                            "description": "Whether to run ImageDescription on the rendered output image.",
                        },
                    },
                    "required": ["geotiff", "poi_specs"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20124)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="GeoTIFFPoiExploreSkill")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = GeoTIFFPoiExploreSkillWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()
