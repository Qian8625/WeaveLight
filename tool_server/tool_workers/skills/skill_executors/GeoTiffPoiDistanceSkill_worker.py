"""
用途：
- 从 GeoTIFF 提取空间范围
- 基于范围生成 AOI 边界数据
- 向 AOI 中添加多类 POI 图层
- 计算两类 POI 之间的距离
- 将结果重新渲染回 GeoTIFF

调用工具链：
- GetBboxFromGeotiff
- GetAreaBoundary
- AddPoisLayer
- ComputeDistance
- DisplayOnGeotiff
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
logger = build_logger(__file__, f"GeoTiffPoiDistanceSkill_worker_{worker_id}.log")


class GeoTiffPoiDistanceSkillWorker(BaseToolWorker):
    """
    First-version composite skill.

    High-level flow:
      1) Get bbox from GeoTIFF
      2) Create AOI gpkg from bbox
      3) Add POI layers into gpkg
      4) Compute distances between two chosen layers
      5) Render layers back on GeoTIFF

    Notes:
      - This first version assumes:
          * GetAreaBoundary / AddPoisLayer / ComputeDistance are already registered online tools
          * GetBboxFromGeotiff / DisplayOnGeotiff are available as offline workers
      - For stability, poi_specs should be explicit OSM queries, for example:
          [
            {"query": {"tourism": "museum"}, "layer_name": "museums"},
            {"query": {"shop": "mall"}, "layer_name": "malls"}
          ]
    """

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str = "auto",
        worker_id: str = worker_id,
        no_register: bool = False,
        model_name: str = "GeoTiffPoiDistanceSkill",
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

        try:
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
        except Exception as e:
            raise RuntimeError(f"Failed to resolve tool '{tool_name}': {e}") from e

    def _call_online_tool(self, tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        addr = self._resolve_tool_addr(tool_name)
        resp = requests.post(
            addr + "/worker_generate",
            headers={"User-Agent": self.model_name},
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _sanitize_layer_name(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip().lower())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned[:30] or "poi_layer"

    @staticmethod
    def _parse_bbox_from_text(text: str) -> List[float]:
        """
        Expected text format from GetBboxFromGeotiff:
          "Bbox (west, south, east, north): (w, s, e, n)"
        """
        m = re.search(
            r"Bbox\s*\(west,\s*south,\s*east,\s*north\):\s*\(\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*\)",
            text,
        )
        if not m:
            raise ValueError(f"Unable to parse bbox from text: {text}")
        return [float(m.group(i)) for i in range(1, 5)]

    def _normalize_poi_specs(self, poi_specs: Any) -> List[Dict[str, Any]]:
        if not isinstance(poi_specs, list) or len(poi_specs) < 2:
            raise ValueError("poi_specs must be a list with at least 2 items.")

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

    # -----------------------------
    # Main
    # -----------------------------
    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        geotiff = params.get("geotiff")
        poi_specs = params.get("poi_specs")

        if not geotiff:
            return {"text": "Missing required parameter: geotiff", "error_code": 2}
        if not poi_specs:
            return {"text": "Missing required parameter: poi_specs", "error_code": 2}
        if not os.path.isfile(geotiff):
            return {"text": f"GeoTIFF not found: {geotiff}", "error_code": 3}

        buffer_m = params.get("buffer_m")
        top = int(params.get("top", 1))
        show_names = bool(params.get("show_names", True))
        render_distance_layer = bool(params.get("render_distance_layer", True))

        skill_trace: List[Dict[str, Any]] = []
        current_gpkg: Optional[str] = None

        try:
            # 1) bbox from geotiff (offline)
            bbox_resp = get_bbox_generate({"geotiff": geotiff})
            skill_trace.append(
                {
                    "step": 1,
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

            # 2) area boundary gpkg (online)
            aoi_payload: Dict[str, Any] = {"area": area_str}
            if buffer_m is not None:
                aoi_payload["buffer_m"] = buffer_m

            aoi_resp = self._call_online_tool("GetAreaBoundary", aoi_payload)
            skill_trace.append(
                {
                    "step": 2,
                    "tool": "GetAreaBoundary",
                    "status": "success" if aoi_resp.get("error_code") == 0 else "failed",
                    "text": aoi_resp.get("text", ""),
                }
            )
            if aoi_resp.get("error_code") != 0 or not aoi_resp.get("gpkg"):
                return {
                    "text": f"GetAreaBoundary failed: {aoi_resp.get('text', '')}",
                    "error_code": 1,
                    "skill_trace": skill_trace,
                }

            current_gpkg = aoi_resp["gpkg"]

            # 3) add poi layers (online)
            normalized_specs = self._normalize_poi_specs(poi_specs)
            added_layers: List[str] = []

            for i, spec in enumerate(normalized_specs, start=1):
                add_resp = self._call_online_tool(
                    "AddPoisLayer",
                    {
                        "gpkg": current_gpkg,
                        "query": spec["query"],
                        "layer_name": spec["layer_name"],
                    },
                )
                skill_trace.append(
                    {
                        "step": 2 + i,
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
                added_layers.append(spec["layer_name"])

            # 4) compute distance (online)
            src_layer = params.get("src_layer") or added_layers[0]
            tar_layer = params.get("tar_layer") or added_layers[1]

            dist_resp = self._call_online_tool(
                "ComputeDistance",
                {
                    "gpkg": current_gpkg,
                    "src_layer": src_layer,
                    "tar_layer": tar_layer,
                    "top": top,
                },
            )
            skill_trace.append(
                {
                    "step": 3 + len(added_layers),
                    "tool": "ComputeDistance",
                    "status": "success" if dist_resp.get("error_code") == 0 else "failed",
                    "text": dist_resp.get("text", ""),
                }
            )
            if dist_resp.get("error_code") != 0:
                return {
                    "text": f"ComputeDistance failed: {dist_resp.get('text', '')}",
                    "error_code": 1,
                    "gpkg": current_gpkg,
                    "skill_trace": skill_trace,
                }

            distance_layer = f"{src_layer}_to_{tar_layer}_distances"

            # 5) render on geotiff (offline)
            layers_to_render = params.get("layers_to_render")
            if not layers_to_render:
                layers_to_render = [src_layer, tar_layer]
                if render_distance_layer:
                    layers_to_render.append(distance_layer)

            display_resp = display_on_geotiff_generate(
                {
                    "gpkg": current_gpkg,
                    "layers": layers_to_render,
                    "geotiff": geotiff,
                    "show_names": show_names,
                }
            )
            skill_trace.append(
                {
                    "step": 4 + len(added_layers),
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
                    "skill_trace": skill_trace,
                }

            summary = (
                f"GeoTIFF POI distance workflow completed successfully. "
                f"Added layers: {added_layers}. "
                f"Computed distances between '{src_layer}' and '{tar_layer}'. "
                f"Rendered layers on the GeoTIFF."
            )

            return {
                "text": summary + "\n\n" + dist_resp.get("text", ""),
                "error_code": 0,
                "gpkg": current_gpkg,
                "image": display_resp.get("image"),
                "skill_trace": skill_trace,
            }

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
                    "Given a GeoTIFF and explicit POI specs, automatically extracts the AOI "
                    "from the GeoTIFF, creates an AOI GeoPackage, adds POI layers, computes "
                    "distances between two layers, and renders the result back onto the GeoTIFF."
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
                                "A list of at least 2 POI specs. "
                                "Each item should include query and optionally layer_name."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "description": (
                                            "OSM query dict or single POI name string. "
                                            "Examples: {'tourism':'museum'} or 'Edinburgh Castle'"
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
                        "src_layer": {
                            "type": "string",
                            "description": "Optional explicit source layer name for distance computation.",
                        },
                        "tar_layer": {
                            "type": "string",
                            "description": "Optional explicit target layer name for distance computation.",
                        },
                        "buffer_m": {
                            "type": "number",
                            "description": "Optional buffer distance in meters around the bbox-derived AOI.",
                        },
                        "top": {
                            "type": "integer",
                            "description": "How many nearest targets per source to keep. Default is 1.",
                        },
                        "show_names": {
                            "type": "boolean",
                            "description": "Whether to render feature names on the output GeoTIFF.",
                        },
                        "render_distance_layer": {
                            "type": "boolean",
                            "description": "Whether to overlay the computed distance layer on the GeoTIFF.",
                        },
                        "layers_to_render": {
                            "type": "array",
                            "description": "Optional custom layer list to render on the GeoTIFF.",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["geotiff", "poi_specs"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20120)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="GeoTiffPoiDistanceSkill")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = GeoTiffPoiDistanceSkillWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()
