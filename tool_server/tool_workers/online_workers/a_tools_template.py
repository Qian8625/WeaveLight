import uuid
import argparse
# 引入基础 Worker 类
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.utils import build_logger

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"MyNewTool_Worker_{worker_id}.log")

class MyNewToolWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="MyNewTool",  # 修改为您的工具名称
                 limit_model_concurrency=1,
                 host="0.0.0.0",
                 port=None,
                 # 可以添加您自定义的参数
                 my_custom_param="default_value"):
        
        self.my_custom_param = my_custom_param
        
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
        如果您需要加载深度学习模型、初始化数据库连接或第三方 API 客户端，请写在这里。
        """
        logger.info(f"{self.model_name} 初始化成功！自定义参数: {self.my_custom_param}")
        # self.my_model = load_my_heavy_model()

    def generate(self, params):
        """
        核心执行逻辑。当收到工具调用请求时，会被触发。
        """
        # 1. 从 params 字典中解析传入的参数
        if "input_data" not in params:
            error_msg = "Missing required parameter: 'input_data'."
            logger.error(error_msg)
            return {"text": error_msg, "error_code": 2}

        input_data = params.get("input_data")

        try:
            # 2. 执行您的工具逻辑 (例如：模型推理、API 调用、复杂计算)
            # result = self.my_model.predict(input_data)
            result = f"Processed: {input_data}" 
            
            # 3. 按照标准格式返回结果
            return {"text": str(result), "error_code": 0}
            
        except Exception as e:
            error_msg = f"Error in {self.model_name}: {e}"
            logger.error(error_msg)
            return {"text": error_msg, "error_code": 1}
        
    def get_tool_instruction(self):
        """ 返回工具的说明文档，这部分内容最终会被组装到LLM 的Prompt中  """
        return {
            "name": self.model_name,
            "description": "这是一个自定义的工具，用于处理特定的任务。当你需要对输入数据进行XXX计算或操作时，请调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_data": {
                        "type": "string",
                        "description": "需要处理的输入文本、图片路径或核心数据"
                    }
                },
                #TODO: 需要哪些参数
                "required": ["input_data"]
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20099) # 为新工具分配一个独立的端口
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://0.0.0.0:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = MyNewToolWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        no_register=args.no_register
    )
    # 启动 FastAPI 服务
    worker.run()