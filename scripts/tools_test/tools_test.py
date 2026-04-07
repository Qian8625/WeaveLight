import os
import json
from tool_server.tool_workers.tool_manager.base_manager import ToolManager

"""
Test script for a single tool worker.
"""

def abs_path(rel_path: str) -> str:
    """Convert relative test file paths to absolute paths."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), rel_path))

def test_single_tool(tool_name: str, params: dict):
    print("\n=== Initializing Tool Manager ===")
    tm = ToolManager(controller_url_location=None)

    # 检查工具是否已注册
    if tool_name not in tm.available_tools:
        print(f"⚠️  {tool_name} is not registered or unavailable — exiting.\n")
        return

    print(f"\n🔹 Testing {tool_name} with parameters:")
    print(json.dumps(params, indent=2))

    # 执行单个工具
    try:
        result = tm.call_tool(tool_name, params)

        if result.get("error_code") == 0:
            print(f"\n✅ {tool_name} executed successfully.")
            print(f"Result:\n{json.dumps(result, indent=2)}\n")
        else:
            print(f"\n❌ {tool_name} returned an error.")
            print(f"Result:\n{json.dumps(result, indent=2)}\n")

    except Exception as e:
        print(f"\n❌ Exception while testing {tool_name}: {str(e)}\n")

if __name__ == "__main__":
    # ==========================================
    # 在这里配置你想要单独测试的工具名称和参数
    # ==========================================
    
    # 示例 1: 测试 Calculator
    target_tool = "Calculator"
    target_params = {
        "expression": "round(302 / (pi), 4)"
    }

    # 示例 2: 测试 TextToBbox (取消注释以使用)
    # target_tool = "TextToBbox"
    # target_params = {
    #     "image": abs_path("tools_test_images/1.jpg"),
    #     "text": "red car",
    #     "top1": True,
    # }

    # 运行测试
    test_single_tool(target_tool, target_params)