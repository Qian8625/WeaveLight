import argparse
import os
import uuid
from pathlib import Path

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.tool_workers.online_workers.geotiff_slicer_core import process_single_geotiff
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"slice_geotiff_worker_{worker_id}.log")


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


class SliceGeoTIFFWorker(BaseToolWorker):
    def __init__(
        self,
        controller_addr,
        worker_addr="auto",
        worker_id=worker_id,
        no_register=False,
        model_name="SliceGeoTIFF",
        host="0.0.0.0",
        port=None,
        limit_model_concurrency=1,
        model_semaphore=None,
        save_path=None,
        wait_timeout=120.0,
        task_timeout=30.0,
    ):
        self.save_path = save_path
        super().__init__(
            controller_addr=controller_addr,
            worker_addr=worker_addr,
            worker_id=worker_id,
            no_register=no_register,
            model_name=model_name,
            device="cpu",
            limit_model_concurrency=limit_model_concurrency,
            host=host,
            port=port,
            model_semaphore=model_semaphore,
            wait_timeout=wait_timeout,
            task_timeout=task_timeout,
        )

    def init_model(self):
        logger.info("SliceGeoTIFFWorker does not load a model. Ready to run.")
        if self.save_path and os.path.isdir(self.save_path):
            logger.info(f"Outputs will be saved to: {self.save_path}")

    def _resolve_output_dir(self, tif_path: str, output_dir: str = None) -> str:
        if output_dir:
            return os.path.abspath(output_dir)

        tif_stem = Path(tif_path).stem
        if self.save_path and os.path.isdir(self.save_path):
            base_dir = self.save_path
        else:
            if self.save_path and not os.path.isdir(self.save_path):
                logger.warning(
                    f"Save path '{self.save_path}' is not a valid directory. "
                    "Falling back to default ./tools_output/"
                )
            base_dir = os.path.join(os.getcwd(), "tools_output")
            os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, tif_stem)

    def generate(self, params):
        if "tif_path" not in params:
            txt_e = "Missing required parameter: tif_path"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        tif_path = params.get("tif_path")
        meta_xml_path = params.get("meta_xml_path")
        rpb_path = params.get("rpb_path")
        output_dir = self._resolve_output_dir(tif_path, params.get("output_dir"))
        tile_size = int(params.get("tile_size", 512))
        overlap = int(params.get("overlap", 0))
        nodata_threshold = float(params.get("nodata_threshold", 0.2))
        use_global_normalization = _parse_bool(params.get("use_global_normalization", True))

        if not os.path.isfile(tif_path):
            txt_e = f"GeoTIFF not found: {tif_path}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 3}

        try:
            result = process_single_geotiff(
                tif_path=tif_path,
                output_dir=output_dir,
                meta_xml_path=meta_xml_path,
                rpb_path=rpb_path,
                tile_size=tile_size,
                overlap=overlap,
                nodata_threshold=nodata_threshold,
                use_global_normalization=use_global_normalization,
            )
        except Exception as exc:
            txt_e = f"Error in {self.model_name}: {exc}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}

        first_tile = result["tiles"][0]["file_path"] if result["tiles"] else None
        txt = (
            f"Sliced {os.path.basename(tif_path)} into {result['tile_count']} tiles. "
            f"Output directory: {result['output_dir']}. "
            f"Tiles index saved at {result['tiles_index_file']}."
        )
        payload = {
            "text": txt,
            "error_code": 0,
            "output_dir": result["output_dir"],
            "metadata_file": result["metadata_file"],
            "tiles_index": result["tiles_index_file"],
            "tile_count": result["tile_count"],
        }
        if first_tile is not None:
            payload["sample_tile"] = first_tile
        return payload

    def get_tool_instruction(self):
        return {
            "name": self.model_name,
            "description": "Slice a single GeoTIFF into fixed-size PNG tiles and save metadata and a tile index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tif_path": {"type": "string", "description": "Path to a single GeoTIFF file."},
                    "meta_xml_path": {"type": "string", "description": "Optional sidecar metadata XML path."},
                    "rpb_path": {"type": "string", "description": "Optional sidecar RPC file path."},
                    "output_dir": {"type": "string", "description": "Optional output directory for this GeoTIFF."},
                    "tile_size": {"type": "integer", "description": "Tile size in pixels. Default is 512."},
                    "overlap": {"type": "integer", "description": "Overlap between adjacent tiles in pixels. Default is 0."},
                    "nodata_threshold": {
                        "type": "number",
                        "description": "Skip tiles whose NoData or dark-pixel ratio exceeds this threshold. Default is 0.2.",
                    },
                    "use_global_normalization": {
                        "type": "boolean",
                        "description": "Whether to normalize tiles using sampled global percentiles. Default is true.",
                    },
                },
                "required": ["tif_path"],
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20026)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://0.0.0.0:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = SliceGeoTIFFWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        save_path=args.save_path,
        no_register=args.no_register,
    )
    worker.run()
