# 基于地表温度（LST）与归一化植被指数（NDVI）计算 TVDI（温度植被干旱指数）
import os
import uuid
import argparse
import rasterio
import numpy as np
from scipy.stats import linregress
from pathlib import Path

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"TVDIAnalysis_Worker_{worker_id}.log")
TEMP_DIR = Path("./temp_results")

class TVDIAnalysisWorker(BaseToolWorker):
    def __init__(self, controller_addr, worker_addr="auto", worker_id=worker_id,
                 no_register=False, model_name="TVDIAnalysis", limit_model_concurrency=1,
                 host="0.0.0.0", port=None):
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
        logger.info(f"{self.model_name} 初始化成功！准备进行 TVDI 计算与阈值分析。")

    def generate(self, params):
        # 1. 解析参数
        required_params = ["ndvi_path", "lst_path", "output_path"]
        for req in required_params:
            if req not in params:
                return {"text": f"Missing parameter: {req}", "error_code": 2}

        ndvi_path = params["ndvi_path"]
        lst_path = params["lst_path"]
        output_path = params["output_path"]
        
        # 阈值分析的默认参数
        threshold = params.get("threshold", 0.75)
        mode = params.get("mode", "above")

        try:
            # ==========================================
            #  计算 TVDI
            # ==========================================
            with rasterio.open(ndvi_path) as src_ndvi:
                ndvi = src_ndvi.read(1).astype(np.float32) * 0.0001
                profile = src_ndvi.profile
            with rasterio.open(lst_path) as src_lst:
                lst = src_lst.read(1).astype(np.float32) * 0.02

            valid_mask = (ndvi >= 0) & (ndvi <= 1) & (lst > 0)
            tvdi = np.full_like(ndvi, np.nan, dtype=np.float32)

            if np.any(valid_mask):
                ndvi_valid = ndvi[valid_mask]
                lst_valid = lst[valid_mask]

                if len(ndvi_valid) >= 100:
                    n_bins = 100
                    bins = np.linspace(ndvi_valid.min(), ndvi_valid.max(), n_bins + 1)
                    ndvi_bin_centers, lst_max_vals, lst_min_vals = [], [], []

                    for i in range(n_bins):
                        bin_mask = (ndvi_valid >= bins[i]) & (ndvi_valid < bins[i + 1])
                        if np.any(bin_mask):
                            ndvi_bin_centers.append((bins[i] + bins[i + 1]) / 2)
                            lst_max_vals.append(np.max(lst_valid[bin_mask]))
                            lst_min_vals.append(np.min(lst_valid[bin_mask]))

                    if len(ndvi_bin_centers) >= 2:
                        ndvi_bin_centers = np.array(ndvi_bin_centers)
                        lst_max_vals = np.array(lst_max_vals)
                        lst_min_vals = np.array(lst_min_vals)

                        # 线性回归拟合干湿边界
                        slope_max, intercept_max, _, _, _ = linregress(ndvi_bin_centers, lst_max_vals)
                        slope_min, intercept_min, _, _, _ = linregress(ndvi_bin_centers, lst_min_vals)

                        lst_max_arr = ndvi * slope_max + intercept_max
                        lst_min_arr = ndvi * slope_min + intercept_min

                        # 计算 TVDI
                        denominator = lst_max_arr - lst_min_arr
                        denominator[denominator == 0] = 1e-6
                        tvdi_calc = (lst - lst_min_arr) / denominator
                        tvdi_calc = np.clip(tvdi_calc, 0, 1).astype(np.float32)
                        tvdi[valid_mask] = tvdi_calc[valid_mask]
                    else:
                        logger.warning(f"Not enough data bins for regression in {output_path}")
                else:
                    logger.warning(f"Too few valid data points ({len(ndvi_valid)}) in {output_path}")
            else:
                logger.warning(f"No valid data points in {output_path}")

            # 保存 TVDI 栅格结果
            profile.update(dtype=rasterio.float32, count=1, compress='lzw')
            final_output_path = TEMP_DIR / output_path
            os.makedirs(final_output_path.parent, exist_ok=True)
            with rasterio.open(final_output_path, 'w', **profile) as dst:
                dst.write(tvdi, 1)

            # ==========================================
            #  计算阈值占比
            # ==========================================
            valid_pixels = ~np.isnan(tvdi)
            total_valid_pixels = np.sum(valid_pixels)
            percentage = 0.0

            if total_valid_pixels > 0:
                if mode == 'above':
                    matching_pixels = np.sum((tvdi > threshold) & valid_pixels)
                elif mode == 'below':
                    matching_pixels = np.sum((tvdi < threshold) & valid_pixels)
                elif mode == 'equal':
                    matching_pixels = np.sum((tvdi == threshold) & valid_pixels)
                elif mode == 'above_equal':
                    matching_pixels = np.sum((tvdi >= threshold) & valid_pixels)
                elif mode == 'below_equal':
                    matching_pixels = np.sum((tvdi <= threshold) & valid_pixels)
                else:
                    return {"text": f"Invalid mode '{mode}'.", "error_code": 1}

                percentage = (matching_pixels / total_valid_pixels) * 100

            # 组合返回结果
            res_text = (f"Result saved at {final_output_path}. "
                        f"The percentage of pixels with TVDI {mode} {threshold} is {percentage:.2f}%.")
            
            return {
                "text": res_text, 
                "error_code": 0,
                "percentage": float(percentage)  # 提供结构化数据以便可能被后续工具使用
            }

        except Exception as e:
            return {"text": f"Error during TVDI calculation: {e}", "error_code": 1}

    def get_tool_instruction(self):
        return {
            "name": self.model_name,
            "description": "Compute the Temperature Vegetation Dryness Index (TVDI) from NDVI and LST rasters, and then calculate the percentage of pixels relative to a specified threshold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ndvi_path": {"type": "string", "description": "Path to the NDVI GeoTIFF file"},
                    "lst_path": {"type": "string", "description": "Path to the LST GeoTIFF file"},
                    "output_path": {"type": "string", "description": "Relative path to save the TVDI raster"},
                    "threshold": {"type": "number", "description": "Threshold value for ratio calculation. Default = 0.75"},
                    "mode": {"type": "string", "description": "Comparison mode: 'above', 'below', 'equal', 'above_equal', 'below_equal'. Default = 'above'"}
                },
                "required": ["ndvi_path", "lst_path", "output_path"]
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20025) 
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://0.0.0.0:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = TVDIAnalysisWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        no_register=args.no_register
    )
    worker.run()