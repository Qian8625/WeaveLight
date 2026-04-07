import os
import uuid
import argparse
import rasterio
import numpy as np
from pathlib import Path
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.utils import build_logger

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"NASATeam_Worker_{worker_id}.log")
TEMP_DIR = Path("./temp_results")

class NASATeamSeaIceWorker(BaseToolWorker):
    def __init__(self, controller_addr, worker_addr="auto", worker_id=worker_id,
                 no_register=False, model_name="nasa_team_sea_ice_concentration", 
                 limit_model_concurrency=1, host="0.0.0.0", port=None):
        super().__init__(controller_addr, worker_addr, worker_id, no_register, model_name,
                         limit_model_concurrency, host, port)

    def init_model(self):
        logger.info(f"{self.model_name} 初始化成功！准备进行海冰浓度计算。")

    def generate(self, params):
        if "bt_paths" not in params or "output_path" not in params:
            return {"text": "Missing 'bt_paths' dict or 'output_path'", "error_code": 2}

        bt_paths = params["bt_paths"] # expected dict: {"19V": path, "19H": path, "37V": path, "37H": path}
        output_path = params["output_path"]
        
        # 默认经验参数
        nd_ice = params.get("nd_ice", 50.0)
        nd_water = params.get("nd_water", 0.0)
        s1_ice = params.get("s1_ice", 20.0)
        s1_water = params.get("s1_water", 0.0)

        try:
            arrays = {}
            profile = None
            for key, path in bt_paths.items():
                with rasterio.open(path) as src:
                    if profile is None: profile = src.profile
                    arrays[key] = src.read(1).astype(np.float32)

            valid_mask = (arrays["19V"] > 0) & (arrays["19H"] > 0) & (arrays["37V"] > 0) & (arrays["37H"] > 0)
            Ci = np.full(arrays["19V"].shape, np.nan, dtype=np.float32)

            # NASA Team 算法核心差值
            ND = arrays["19V"] - arrays["19H"]
            S1 = arrays["37V"] - arrays["37H"]

            term1 = (ND - nd_water) / (nd_ice - nd_water)
            term2 = (S1 - s1_water) / (s1_ice - s1_water)
            
            Ci[valid_mask] = (term1[valid_mask] + term2[valid_mask]) / 2
            Ci = np.clip(Ci, 0, 1) # 海冰浓度限制在 0-1 之间

            profile.update(dtype=rasterio.float32, count=1, compress='lzw')
            final_path = TEMP_DIR / output_path
            os.makedirs(final_path.parent, exist_ok=True)
            with rasterio.open(final_path, 'w', **profile) as dst:
                dst.write(Ci, 1)

            return {"text": f'Result saved at {final_path}', "error_code": 0}
        except Exception as e:
            return {"text": f"Error: {e}", "error_code": 1}
        
    def get_tool_instruction(self):
        return {
            "name": self.model_name,
            "description": "NASA Team algorithm for estimating sea ice concentration from passive microwave data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bt_paths": {
                        "type": "object", 
                        "description": "Dict of paths with keys '19V', '19H', '37V', '37H'"
                    },
                    "output_path": {"type": "string"},
                    "nd_ice": {"type": "number"},
                    "nd_water": {"type": "number"},
                    "s1_ice": {"type": "number"},
                    "s1_water": {"type": "number"}
                },
                "required": ["bt_paths", "output_path"]
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20102) # 分配专属端口
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://0.0.0.0:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = NASATeamSeaIceWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        no_register=args.no_register
    )
    # 启动 FastAPI 服务
    worker.run()