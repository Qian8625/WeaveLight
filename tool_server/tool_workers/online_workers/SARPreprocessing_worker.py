import argparse
import os
import uuid
import warnings
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from scipy.ndimage import uniform_filter

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"SARPreprocessing_worker_{worker_id}.log")

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def db2linear(img: np.ndarray) -> np.ndarray:
    """Convert dB to linear safely."""
    out = np.full(img.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(img)
    out[valid] = np.power(10.0, img[valid] / 10.0)
    return out


def linear2db(img: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Convert linear to dB safely."""
    out = np.full(img.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(img) & (img > floor)
    out[valid] = 10.0 * np.log10(img[valid])
    return out


def normalize_to_uint8(
    img: np.ndarray,
    lower_percent: float = 1.0,
    upper_percent: float = 99.0,
) -> np.ndarray:
    """Robust percentile stretch to uint8."""
    valid = img[np.isfinite(img)]
    if valid.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)

    low = np.percentile(valid, lower_percent)
    high = np.percentile(valid, upper_percent)

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(img.shape, dtype=np.uint8)

    img_clip = np.clip(img, low, high)
    img_norm = (img_clip - low) / (high - low + 1e-8)
    img_norm = np.clip(img_norm, 0.0, 1.0)
    img_norm[~np.isfinite(img_norm)] = 0.0
    return (img_norm * 255.0).astype(np.uint8)


def parse_tile_grid_size(tile_grid_size) -> Tuple[int, int]:
    if isinstance(tile_grid_size, (list, tuple)) and len(tile_grid_size) == 2:
        x, y = int(tile_grid_size[0]), int(tile_grid_size[1])
        if x > 0 and y > 0:
            return x, y
    raise ValueError("clahe_tile_grid_size must be [x, y] with positive integers.")


def lee_filter(img: np.ndarray, win_size: int = 5, noise_scale: float = 1.0) -> np.ndarray:
    """
    Simplified Lee filter with NaN support.
    Expect image in linear-like domain.
    """
    img = img.astype(np.float32)

    valid_mask = np.isfinite(img).astype(np.float32)
    img_filled = np.where(np.isfinite(img), img, 0.0)

    local_count = uniform_filter(valid_mask, size=win_size) * (win_size ** 2)
    local_sum = uniform_filter(img_filled, size=win_size) * (win_size ** 2)
    local_sum_sq = uniform_filter(img_filled ** 2, size=win_size) * (win_size ** 2)

    local_count = np.maximum(local_count, 1.0)
    local_mean = local_sum / local_count
    local_var = np.maximum(local_sum_sq / local_count - local_mean ** 2, 0.0)

    valid_var = local_var[np.isfinite(local_var)]
    noise_var = float(np.mean(valid_var) * noise_scale) if valid_var.size > 0 else 0.0

    weights = local_var / (local_var + noise_var + 1e-8)
    weights = np.clip(weights, 0.0, 1.0)

    filtered = local_mean + weights * (img_filled - local_mean)
    filtered[valid_mask == 0] = np.nan
    return filtered.astype(np.float32)


def apply_clahe(
    img_uint8: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tile_grid_size,
    )
    return clahe.apply(img_uint8)


def ensure_gray(img: np.ndarray) -> np.ndarray:
    """Convert PNG arrays to grayscale."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def infer_input_unit(
    img: np.ndarray,
    source_ext: str,
    requested_unit: str,
) -> str:
    """
    auto inference:
    - PNG -> display
    - TIFF -> db if many negatives / typical dB range, else linear
    """
    requested_unit = requested_unit.lower()
    if requested_unit != "auto":
        return requested_unit

    if source_ext == ".png":
        return "display"

    valid = img[np.isfinite(img)]
    if valid.size == 0:
        return "display"

    p1, p50, p99 = np.percentile(valid, [1, 50, 99])

    # heuristic for common SAR dB data
    if p1 < 0 and p99 < 60 and p50 < 20:
        return "db"

    return "linear"


def read_input_image(file_path: str, band_index: int = 1) -> Tuple[np.ndarray, Optional[dict], str]:
    """
    Read PNG or TIFF and return:
        img: float32, shape(H, W)
        profile: rasterio profile or None
        ext: '.png' / '.tif' / '.tiff'
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".png":
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Unable to read PNG: {file_path}")
        img = ensure_gray(img)
        return img.astype(np.float32), None, ext

    if ext in [".tif", ".tiff"]:
        with rasterio.open(path) as src:
            profile = src.profile.copy()

            if not (1 <= band_index <= src.count):
                raise ValueError(f"band_index must be in [1, {src.count}], got {band_index}")

            img = src.read(band_index).astype(np.float32)

            nodata = src.nodata
            if nodata is not None:
                img = np.where(img == nodata, np.nan, img)

            mask = src.read_masks(band_index) == 0
            img[mask] = np.nan

            return img, profile, ext

    raise ValueError("Only PNG / TIFF inputs are supported.")


def build_output_path(base_dir: Path, output_path: str, input_path: str) -> Path:
    """
    Always output PNG.
    """
    if output_path:
        out = base_dir / output_path
        if out.suffix.lower() != ".png":
            out = out.with_suffix(".png")
        return out

    input_name = Path(input_path).stem
    return base_dir / f"{input_name}_sar_preprocessed.png"


def save_png(output_path: Path, img_array: np.ndarray):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_save = np.clip(img_array, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(str(output_path), img_save)
    if not ok:
        raise IOError(f"Failed to save PNG: {output_path}")


class SARPreprocessingWorker(BaseToolWorker):
    def __init__(
        self,
        controller_addr,
        worker_addr="auto",
        worker_id=worker_id,
        no_register=False,
        model_name="SARPreprocessing",
        limit_model_concurrency=1,
        host="0.0.0.0",
        port=None,
        save_path=None,
        model_semaphore=None,
        wait_timeout=120.0,
        task_timeout=120.0,
    ):
        self.save_path = save_path

        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            None,
            None,
            model_name,
            False,
            False,
            "cpu",
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

    def init_model(self):
        logger.info(f"{self.model_name} initialized successfully.")
        if self.save_path:
            logger.info(f"Output directory: {self.save_path}")

    def _get_save_dir(self) -> Path:
        if self.save_path and os.path.isdir(self.save_path):
            return Path(self.save_path)
        if self.save_path and not os.path.isdir(self.save_path):
            logger.warning(f"Invalid save_path '{self.save_path}', fallback to ./tools_output")
        save_dir = Path(os.getcwd()) / "tools_output"
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def generate(self, params):
        if "input_path" not in params:
            msg = "Missing required parameter: 'input_path'."
            logger.error(msg)
            return {"text": msg, "error_code": 2}

        input_path = str(params["input_path"]).strip()
        output_path = str(params.get("output_path", "")).strip()

        filter_window = int(params.get("filter_window", 5))
        noise_scale = float(params.get("noise_scale", 1.0))
        band_index = int(params.get("band_index", 1))

        input_unit = str(params.get("input_unit", "auto")).lower()
        visualization_unit = str(params.get("visualization_unit", "auto")).lower()

        clahe_clip_limit = float(params.get("clahe_clip_limit", 2.0))
        clahe_tile_grid_size = parse_tile_grid_size(params.get("clahe_tile_grid_size", [8, 8]))

        stretch_lower = float(params.get("stretch_lower_percent", 1.0))
        stretch_upper = float(params.get("stretch_upper_percent", 99.0))

        try:
            if filter_window < 3 or filter_window % 2 == 0:
                raise ValueError("filter_window must be an odd integer >= 3.")

            if input_unit not in {"auto", "linear", "db", "display"}:
                raise ValueError("input_unit must be one of: auto, linear, db, display.")

            if visualization_unit not in {"auto", "linear", "db", "display"}:
                raise ValueError("visualization_unit must be one of: auto, linear, db, display.")

            raw_img, profile, ext = read_input_image(input_path, band_index=band_index)
            actual_input_unit = infer_input_unit(raw_img, ext, input_unit)

            # preprocessing domain
            if actual_input_unit == "db":
                proc_linear = db2linear(raw_img)
            elif actual_input_unit == "linear":
                proc_linear = raw_img.astype(np.float32)
            else:
                # display-domain PNG-like image
                proc_linear = raw_img.astype(np.float32) / 255.0

            denoised_linear = lee_filter(
                proc_linear,
                win_size=filter_window,
                noise_scale=noise_scale,
            )

            # visualization domain for output PNG
            actual_vis_unit = visualization_unit
            if actual_vis_unit == "auto":
                if actual_input_unit == "db":
                    actual_vis_unit = "db"
                elif actual_input_unit == "linear":
                    actual_vis_unit = "db"
                else:
                    actual_vis_unit = "display"

            if actual_vis_unit == "db":
                vis_img = linear2db(denoised_linear)
            elif actual_vis_unit == "linear":
                vis_img = denoised_linear
            else:
                # display
                if actual_input_unit == "display":
                    vis_img = denoised_linear * 255.0
                else:
                    vis_img = normalize_to_uint8(linear2db(denoised_linear)).astype(np.float32)

            stretched = normalize_to_uint8(
                vis_img,
                lower_percent=stretch_lower,
                upper_percent=stretch_upper,
            )
            enhanced = apply_clahe(
                stretched,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
            )

            save_dir = self._get_save_dir()
            final_output_path = build_output_path(save_dir, output_path, input_path)
            save_png(final_output_path, enhanced)

            valid = raw_img[np.isfinite(raw_img)]
            raw_min = float(np.min(valid)) if valid.size else None
            raw_max = float(np.max(valid)) if valid.size else None

            return {
                "text": f"SAR preprocessing completed. Output PNG saved to {final_output_path}",
                "error_code": 0,
                "image": str(final_output_path),
                "output_path": str(final_output_path),
                "input_unit": actual_input_unit,
                "visualization_unit": actual_vis_unit,
                "band_index": band_index,
                "input_stats": {
                    "min": raw_min,
                    "max": raw_max,
                    "valid_pixels": int(valid.size),
                },
            }

        except Exception as e:
            error_msg = f"Error in {self.model_name}: {e}"
            logger.error(error_msg)
            return {"text": error_msg, "error_code": 1}

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": (
                    "Preprocess a SAR image from PNG or TIFF and save the output as a PNG image. "
                    "Supports grayscale conversion, optional band selection for TIFF, "
                    "Lee speckle filtering, percentile stretch, and CLAHE enhancement."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_path": {
                            "type": "string",
                            "description": "Path to input image. Supports PNG, TIFF, or TIF."
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Optional relative output PNG path. If omitted, a default PNG name is generated."
                        },
                        "band_index": {
                            "type": "integer",
                            "description": "Band index for TIFF input. Default is 1."
                        },
                        "input_unit": {
                            "type": "string",
                            "description": "Input unit: auto, linear, db, or display. Default is auto."
                        },
                        "visualization_unit": {
                            "type": "string",
                            "description": "Visualization unit for output PNG: auto, linear, db, or display. Default is auto."
                        },
                        "filter_window": {
                            "type": "integer",
                            "description": "Lee filter window size, odd integer >= 3. Default is 5."
                        },
                        "noise_scale": {
                            "type": "number",
                            "description": "Noise variance scaling factor for Lee filter. Default is 1.0."
                        },
                        "clahe_clip_limit": {
                            "type": "number",
                            "description": "CLAHE clip limit. Default is 2.0."
                        },
                        "clahe_tile_grid_size": {
                            "type": "array",
                            "description": "CLAHE tile grid size as [x, y]. Default is [8, 8].",
                            "items": {"type": "integer"}
                        },
                        "stretch_lower_percent": {
                            "type": "number",
                            "description": "Lower percentile for contrast stretch. Default is 1.0."
                        },
                        "stretch_upper_percent": {
                            "type": "number",
                            "description": "Upper percentile for contrast stretch. Default is 99.0."
                        }
                    },
                    "required": ["input_path"]
                }
            }
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20101)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="SARPreprocessing")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = SARPreprocessingWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        save_path=args.save_path,
    )
    worker.run()