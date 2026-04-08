# SAR图像的预处理算法
import os
import uuid
import argparse
import warnings
import rasterio
import numpy as np
import cv2

from pathlib import Path
from scipy.ndimage import uniform_filter
from rasterio.errors import NotGeoreferencedWarning

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.utils import build_logger

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"SARPreprocessingTool_Worker_{worker_id}.log")

TEMP_DIR = Path("./temp_results")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def db2linear(img: np.ndarray) -> np.ndarray:
    """dB -> linear"""
    return np.power(10.0, img / 10.0).astype(np.float32)


def linear2db(img: np.ndarray) -> np.ndarray:
    """linear -> dB"""
    out = np.full(img.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(img) & (img > 0)
    out[valid] = 10.0 * np.log10(img[valid])
    return out


def normalize_to_uint8(
    img: np.ndarray,
    lower_percent: float = 1,
    upper_percent: float = 99
) -> np.ndarray:
    """按百分位拉伸到 0~255"""
    valid = img[np.isfinite(img)]
    if valid.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)

    low = np.percentile(valid, lower_percent)
    high = np.percentile(valid, upper_percent)

    if high <= low:
        return np.zeros(img.shape, dtype=np.uint8)

    img_clip = np.clip(img, low, high)
    img_norm = (img_clip - low) / (high - low + 1e-8) * 255.0
    img_norm[~np.isfinite(img_norm)] = 0
    return img_norm.astype(np.uint8)


def lee_filter(img: np.ndarray, win_size: int = 5, noise_scale: float = 1.2) -> np.ndarray:
    """
    简化版 Lee 滤波
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
    noise_var = np.mean(valid_var) * noise_scale if valid_var.size > 0 else 0.0

    weights = local_var / (local_var + noise_var + 1e-8)
    weights = np.clip(weights, 0.0, 1.0)

    filtered = local_mean + weights * (img_filled - local_mean)
    filtered[valid_mask == 0] = np.nan
    return filtered.astype(np.float32)


def image_enhancement(
    img: np.ndarray,
    clip_limit: float = 1.2,
    tile_grid_size=(4, 4)
) -> np.ndarray:
    """
    预览增强：百分位拉伸 + CLAHE
    """
    img_8u = normalize_to_uint8(img, lower_percent=1, upper_percent=99)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    enhanced = clahe.apply(img_8u)
    return enhanced.astype(np.uint8)


def parse_tile_grid_size(tile_grid_size):
    if isinstance(tile_grid_size, (list, tuple)) and len(tile_grid_size) == 2:
        x, y = int(tile_grid_size[0]), int(tile_grid_size[1])
        if x > 0 and y > 0:
            return (x, y)
    raise ValueError(
        "clahe_tile_grid_size must be a list/tuple with two positive integers, e.g. [4, 4]."
    )


def read_input_image(file_path: str):
    """
    支持：
    1. PNG（灰度或 RGB）
    2. 单波段 TIFF（典型 SAR）
    3. >=3 波段 TIFF（按 RGB 转灰度）
    返回：
        img: float32, shape(H, W)
        profile: rasterio profile or None
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".png":
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Unable to read PNG: {file_path}")

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img.astype(np.float32), None

    if ext in [".tif", ".tiff"]:
        with rasterio.open(file_path) as src:
            profile = src.profile.copy()

            if src.count == 1:
                img = src.read(1).astype(np.float32)
                mask = src.read_masks(1) == 0
                nodata = src.nodata
                if nodata is not None:
                    img = np.where(img == nodata, np.nan, img)
                img[mask] = np.nan
                return img, profile

            if src.count >= 3:
                rgb = src.read([1, 2, 3]).astype(np.float32)
                gray_img = 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]

                nodata = src.nodata
                if nodata is not None:
                    gray_img = np.where(gray_img == nodata, np.nan, gray_img)

                mask_stack = [(src.read_masks(i) == 0) for i in [1, 2, 3]]
                invalid_mask = np.any(np.stack(mask_stack, axis=0), axis=0)
                gray_img[invalid_mask] = np.nan

                return gray_img.astype(np.float32), profile

            raise ValueError(
                f"Unsupported TIFF band count: {src.count}. "
                "Expect 1-band SAR or >=3-band image."
            )

    raise ValueError("Only PNG / TIFF inputs are supported.")


def save_geotiff(output_path: Path, img: np.ndarray, src_profile=None):
    """
    保存单波段 float32 GeoTIFF
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nodata_value = -9999.0
    img_save = np.where(np.isfinite(img), img, nodata_value).astype(np.float32)

    if src_profile is not None:
        out_profile = src_profile.copy()
        out_profile.update(
            driver="GTiff",
            dtype=rasterio.float32,
            count=1,
            compress="lzw",
            nodata=nodata_value
        )
    else:
        out_profile = {
            "driver": "GTiff",
            "height": img.shape[0],
            "width": img.shape[1],
            "count": 1,
            "dtype": rasterio.float32,
            "compress": "lzw",
            "nodata": nodata_value
        }

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(img_save, 1)


def save_png(output_path: Path, img_array: np.ndarray):
    """
    保存 uint8 PNG
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_save = np.clip(img_array, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(str(output_path), img_save)
    if not ok:
        raise IOError(f"Failed to save PNG: {output_path}")


class SARPreprocessingWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="sar_preprocessing",
                 limit_model_concurrency=1,
                 host="0.0.0.0",
                 port=None):

        super().__init__(
            controller_addr=controller_addr,
            worker_addr=worker_addr,
            worker_id=worker_id,
            no_register=no_register,
            model_name=model_name,
            limit_model_concurrency=limit_model_concurrency,
            host=host,
            port=port
        )

    def init_model(self):
        """
        初始化方法。
        """
        logger.info(
            f"{self.model_name} 初始化成功！准备进行 SAR 图像预处理。"
        )

    def generate(self, params):
        """
        核心执行逻辑：
        1. 读取输入图像
        2. 若输入为 dB，则转 linear 后做 Lee 滤波
        3. 按要求输出 linear / dB 结果
        4. 生成 CLAHE 预览图
        """
        required_params = ["input_path", "output_filtered_path"]
        for req in required_params:
            if req not in params:
                error_msg = f"Missing required parameter: '{req}'."
                logger.error(error_msg)
                return {"text": error_msg, "error_code": 2}

        input_path = params.get("input_path")
        output_filtered_path = params.get("output_filtered_path")

        output_preview_path = params.get("output_preview_path")
        save_preview = bool(params.get("save_preview", True))

        filter_window = int(params.get("filter_window", 5))
        noise_scale = float(params.get("noise_scale", 1.2))
        clahe_clip_limit = float(params.get("clahe_clip_limit", 1.2))
        clahe_tile_grid_size = parse_tile_grid_size(
            params.get("clahe_tile_grid_size", [4, 4])
        )

        input_unit = str(params.get("input_unit", "linear")).lower()
        output_unit = str(params.get("output_unit", "same")).lower()

        try:
            if filter_window < 3 or filter_window % 2 == 0:
                raise ValueError("filter_window must be an odd integer >= 3.")

            if input_unit not in ["linear", "db"]:
                raise ValueError("input_unit must be 'linear' or 'dB'.")

            if output_unit not in ["same", "linear", "db"]:
                raise ValueError("output_unit must be 'same', 'linear', or 'dB'.")

            # 1. 读取图像
            raw_img, profile = read_input_image(input_path)

            # 2. 统一在 linear 域做 Lee 滤波
            if input_unit == "db":
                proc_img = db2linear(raw_img)
            else:
                proc_img = raw_img.astype(np.float32)

            denoised_linear = lee_filter(
                proc_img,
                win_size=filter_window,
                noise_scale=noise_scale
            )

            # 3. 按需求决定输出单位
            if output_unit == "same":
                output_unit = input_unit

            if output_unit == "db":
                denoised_to_save = linear2db(denoised_linear)
            else:
                denoised_to_save = denoised_linear

            # 4. 生成预览图
            preview_img = image_enhancement(
                denoised_to_save,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size
            )

            # 5. 保存滤波结果
            final_filtered_path = TEMP_DIR / output_filtered_path
            final_filtered_path.parent.mkdir(parents=True, exist_ok=True)

            filtered_ext = final_filtered_path.suffix.lower()
            if filtered_ext in [".tif", ".tiff"]:
                save_geotiff(final_filtered_path, denoised_to_save, profile)
            elif filtered_ext == ".png":
                save_png(final_filtered_path, normalize_to_uint8(denoised_to_save))
            else:
                raise ValueError(
                    "output_filtered_path must end with .tif, .tiff, or .png"
                )

            # # 6. 保存预览图
            # final_preview_path = None
            # if save_preview:
            #     if output_preview_path is None:
            #         final_preview_path = final_filtered_path.with_name(
            #             final_filtered_path.stem + "_preview.png"
            #         )
            #     else:
            #         final_preview_path = TEMP_DIR / output_preview_path

            #     save_png(final_preview_path, preview_img)

            result = {
                "text": f"Preprocessing completed. Filtered result saved at {final_filtered_path}",
                "error_code": 0,
                "filtered_path": str(final_filtered_path),
                "input_unit": input_unit,
                "output_unit": output_unit
            }
            return result

        except Exception as e:
            error_msg = f"Error in {self.model_name}: {e}"
            logger.error(error_msg)
            return {"text": error_msg, "error_code": 1}

    def get_tool_instruction(self):
        """返回工具说明文档"""
        return {
            "name": self.model_name,
            "description": (
                "SAR image preprocessing tool. Supports single-band SAR GeoTIFF or PNG input, "
                "optional RGB-to-gray conversion, Lee speckle filtering, and CLAHE preview enhancement."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to input SAR image. Supports single-band GeoTIFF, RGB GeoTIFF, or PNG."
                    },
                    "output_filtered_path": {
                        "type": "string",
                        "description": "Relative output path for the filtered image, e.g. 'task01/sar_filtered.tif'."
                    },
                    "output_preview_path": {
                        "type": "string",
                        "description": "Optional relative output path for the enhanced preview PNG, e.g. 'task01/sar_preview.png'."
                    },
                    "save_preview": {
                        "type": "boolean",
                        "description": "Whether to save a CLAHE-enhanced preview PNG. Default is true."
                    },
                    "filter_window": {
                        "type": "integer",
                        "description": "Lee filter window size. Must be an odd integer >= 3. Default is 5."
                    },
                    "noise_scale": {
                        "type": "number",
                        "description": "Noise variance scaling factor used in Lee filtering. Default is 1.2."
                    },
                    "clahe_clip_limit": {
                        "type": "number",
                        "description": "CLAHE clip limit for preview enhancement. Default is 1.2."
                    },
                    "clahe_tile_grid_size": {
                        "type": "array",
                        "description": "CLAHE tile grid size as [x, y]. Default is [4, 4].",
                        "items": {
                            "type": "integer"
                        }
                    },
                    "input_unit": {
                        "type": "string",
                        "description": "Input SAR backscatter unit: 'linear' or 'dB'. Default is 'linear'."
                    },
                    "output_unit": {
                        "type": "string",
                        "description": "Output unit for filtered result: 'same', 'linear', or 'dB'. Default is 'same'."
                    }
                },
                "required": ["input_path", "output_filtered_path"]
            }
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20101)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://0.0.0.0:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = SARPreprocessingWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        no_register=args.no_register
    )
    worker.run()