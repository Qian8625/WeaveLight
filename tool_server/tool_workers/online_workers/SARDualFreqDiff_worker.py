import os
import uuid
import argparse
import rasterio
import numpy as np
from pathlib import Path

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"SARDualFreqDiff_Worker_{worker_id}.log")

TEMP_DIR = Path("./temp_results")

class SARDualFreqDiffWorker(BaseToolWorker):
    def __init__(self, controller_addr, worker_addr="auto", worker_id=worker_id,
                 no_register=False, model_name="SARDualFreqDiff", limit_model_concurrency=1,
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
        logger.info(f"{self.model_name} 初始化成功！准备进行双频差分法参数反演。")

    def generate(self, params):
        required_params = ["band1_path", "band2_path", "parameter", "output_path"]
        for req in required_params:
            if req not in params:
                return {"text": f"Missing parameter: {req}", "error_code": 2}

        band1_path, band2_path = params["band1_path"], params["band2_path"]
        parameter, output_path = params["parameter"], params["output_path"]
        
        param_models = {
            "SM": {"alpha": 0.7, "beta": 0.1},
            "VI": {"alpha": 0.5, "beta": 0.0},
            "LAI": {"alpha": 0.6, "beta": 0.05},
        }
        
        param_upper = parameter.upper()
        if param_upper not in param_models:
            return {"text": "Unsupported parameter. Choose 'SM', 'VI', or 'LAI'.", "error_code": 1}

        alpha = params.get("alpha", param_models[param_upper]["alpha"])
        beta = params.get("beta", param_models[param_upper]["beta"])

        try:
            with rasterio.open(band1_path) as src1, rasterio.open(band2_path) as src2:
                band1 = src1.read(1).astype(np.float32)
                band2 = src2.read(1).astype(np.float32)
                profile = src1.profile

            mask = (band1 != src1.nodata) & (band2 != src2.nodata) if src1.nodata else (band1 > 0)
            
            diff = np.full_like(band1, np.nan, dtype=np.float32)
            diff[mask] = band1[mask] - band2[mask]

            param_img = np.full_like(band1, np.nan, dtype=np.float32)
            param_img[mask] = alpha * diff[mask] + beta
            param_img = np.clip(param_img, 0, 1)

            profile.update(dtype=rasterio.float32, count=2, compress='lzw')
            final_output_path = TEMP_DIR / output_path
            os.makedirs(final_output_path.parent, exist_ok=True)
            
            with rasterio.open(final_output_path, 'w', **profile) as dst:
                dst.write(diff, 1)      # Band 1: Difference
                dst.write(param_img, 2) # Band 2: Inverted Parameter

            return {"text": f'Result saved at {final_output_path}', "error_code": 0}
        except Exception as e:
            return {"text": f"Error: {e}", "error_code": 1}
        
    def get_tool_instruction(self):
        return {
            "name": self.model_name,
            "description": "Dual-frequency Differential Method (DDM) for parameter inversion (SM, VI, LAI).",
            "parameters": {
                "type": "object",
                "properties": {
                    "band1_path": {"type": "string"},
                    "band2_path": {"type": "string"},
                    "parameter": {"type": "string", "description": "'SM', 'VI', or 'LAI'"},
                    "output_path": {"type": "string"},
                    "alpha": {"type": "number"},
                    "beta": {"type": "number"}
                },
                "required": ["band1_path", "band2_path", "parameter", "output_path"]
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20024) 
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://0.0.0.0:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = SARDualFreqDiffWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        no_register=args.no_register
    )
    worker.run()