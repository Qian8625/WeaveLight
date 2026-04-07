import os
import uuid
import argparse
import rasterio
import numpy as np
from pathlib import Path

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.utils import build_logger

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"SARDualPolDiffTool_Worker_{worker_id}.log")

# 设定一个临时目录用于保存输出文件，与原库逻辑保持一致
TEMP_DIR = Path("./temp_results")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

class SARDualPolDiffWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="dual_polarization_differential",  # 修改为工具名称
                 limit_model_concurrency=1,
                 host="0.0.0.0",
                 port=None):
        
        # 调用父类初始化，完成 FastAPI 路由和 Controller 注册
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
        此工具主要基于 numpy 和 rasterio 进行计算，无需加载大型深度学习模型。
        """
        logger.info(f"{self.model_name} 初始化成功！准备进行 SAR 图像双极化差分反演。")

    def generate(self, params):
        """
        核心执行逻辑。当收到工具调用请求时，会被触发。
        包含双极化差分反演算法的具体实现。
        """
        # 1. 提取并校验必要参数
        required_params = ["pol1_path", "pol2_path", "parameter", "output_path"]
        for req in required_params:
            if req not in params:
                error_msg = f"Missing required parameter: '{req}'."
                logger.error(error_msg)
                return {"text": error_msg, "error_code": 2}

        pol1_path = params.get("pol1_path")
        pol2_path = params.get("pol2_path")
        parameter = params.get("parameter")
        output_path = params.get("output_path")
        
        # 提取可选参数
        a = params.get("a", 0.3)
        b = params.get("b", 0.1)
        input_unit = params.get("input_unit", "dB")

        try:
            # 2. 执行 SAR 图像处理逻辑
            def db2linear(db):
                """Convert decibel (dB) values to linear scale."""
                return 10 ** (db / 10)

            # 读取极化波段数据
            with rasterio.open(pol1_path) as src1, rasterio.open(pol2_path) as src2:
                band1 = src1.read(1).astype(np.float32)
                band2 = src2.read(1).astype(np.float32)
                profile = src1.profile

            # 如果输入数据是 dB 单位，转换为线性刻度
            if input_unit.lower() == "db":
                band1 = db2linear(band1)
                band2 = db2linear(band2)

            # 屏蔽无效数据 (非正值)
            valid_mask = (band1 > 0) & (band2 > 0)

            # 初始化输出数组为 NaN
            output = np.full(band1.shape, np.nan, dtype=np.float32)

            # 计算差值与和值
            diff = band1 - band2
            sum_ = band1 + band2
            sum_[sum_ == 0] = np.nan  # 防止除以 0

            param_lower = parameter.lower()
            if param_lower == "soil_moisture":
                # 对有效像素应用线性土壤水分模型
                output[valid_mask] = a * diff[valid_mask] + b
            elif param_lower == "vegetation_index":
                # 对有效像素计算植被指数比率
                output[valid_mask] = diff[valid_mask] / sum_[valid_mask]
            else:
                raise ValueError("Unsupported parameter. Choose 'soil_moisture' or 'vegetation_index'.")

            # 更新输出的 raster profile
            profile.update(dtype=rasterio.float32, count=1, compress='lzw')

            # 准备输出目录并保存结果
            final_output_path = TEMP_DIR / output_path
            os.makedirs(final_output_path.parent, exist_ok=True)

            with rasterio.open(final_output_path, 'w', **profile) as dst:
                dst.write(output, 1)

            result_msg = f'Result saved at {final_output_path}'
            
            # 3. 按照标准格式返回结果
            return {"text": result_msg, "error_code": 0}
            
        except Exception as e:
            error_msg = f"Error in {self.model_name}: {e}"
            logger.error(error_msg)
            return {"text": error_msg, "error_code": 1}
        
    def get_tool_instruction(self):
        """ 返回工具的说明文档，这部分内容最终会被组装到 LLM 的 Prompt 中 """
        return {
            "name": self.model_name,
            "description": "Dual-Polarization Differential Method (DPDM) for microwave remote sensing (SAR) parameter inversion. Supports soil moisture and vegetation index estimation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pol1_path": {
                        "type": "string",
                        "description": "File path for the first polarization band GeoTIFF (e.g., VV)."
                    },
                    "pol2_path": {
                        "type": "string",
                        "description": "File path for the second polarization band GeoTIFF (e.g., VH)."
                    },
                    "parameter": {
                        "type": "string",
                        "description": "Parameter to invert, options: 'soil_moisture' or 'vegetation_index'."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Relative path for the output raster file, e.g., 'question17/sm_2022-01-16.tif'."
                    },
                    "a": {
                        "type": "number",
                        "description": "Linear coefficient for soil moisture model. Default is 0.3."
                    },
                    "b": {
                        "type": "number",
                        "description": "Intercept for soil moisture model. Default is 0.1."
                    },
                    "input_unit": {
                        "type": "string",
                        "description": "Unit of input data, either 'dB' or 'linear'. Default is 'dB'."
                    }
                },
                "required": ["pol1_path", "pol2_path", "parameter", "output_path"]
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20100) # 分配专属端口
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://0.0.0.0:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = SARDualPolDiffWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        no_register=args.no_register
    )
    # 启动 FastAPI 服务
    worker.run()