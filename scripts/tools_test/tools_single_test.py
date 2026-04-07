import os
import json
from tool_server.tool_workers.tool_manager.base_manager import ToolManager

"""
Test script for the SARDualFreqDiff tool worker.
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
        print(f"当前可用的工具列表: {tm.available_tools}")
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
    # 测试工具名称
    target_tool = "TVDIAnalysis"  
    
    # 按照工具需求，准备测试参数
    target_params = {
        "ndvi_path": abs_path("tools_test_images/Sichuan_2021-07-12_NDVI.tif"), 
        "lst_path": abs_path("tools_test_images/Sichuan_2021-07-12_LST.tif"), 
        "output_path": "test_diff_result_sm.tif",
    }

    test_single_tool(target_tool, target_params)