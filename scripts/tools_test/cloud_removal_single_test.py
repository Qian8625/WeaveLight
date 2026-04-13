import json
import os

from tool_server.tool_workers.tool_manager.base_manager import ToolManager


def test_single_tool(tool_name: str, params: dict):
    print("\n=== Initializing Tool Manager ===")
    tm = ToolManager(controller_url_location=None)

    if tool_name not in tm.available_tools:
        print(f"⚠️  {tool_name} is not registered or unavailable — exiting.\n")
        print(f"当前可用的工具列表: {tm.available_tools}")
        return

    print(f"\n🔹 Testing {tool_name} with parameters:")
    print(json.dumps(params, indent=2))

    try:
        result = tm.call_tool(tool_name, params)
        if result.get("error_code") == 0:
            print(f"\n✅ {tool_name} executed successfully.")
        else:
            print(f"\n❌ {tool_name} returned an error.")
        print(f"Result:\n{json.dumps(result, indent=2)}\n")
    except Exception as e:
        print(f"\n❌ Exception while testing {tool_name}: {str(e)}\n")


if __name__ == "__main__":
    optical_image = os.environ.get("CLOUD_REMOVAL_IMAGE")
    nir_image = os.environ.get("CLOUD_REMOVAL_NIR_IMAGE")

    if not optical_image:
        raise SystemExit(
            "Please set CLOUD_REMOVAL_IMAGE before running this test.\n"
            "Example:\n"
            "  CLOUD_REMOVAL_IMAGE=/path/to/cloudy_rgb.png \\\n"
            "  CLOUD_REMOVAL_NIR_IMAGE=/path/to/optional_nir.png \\\n"
            "  python scripts/tools_test/cloud_removal_single_test.py"
        )

    target_tool = "CloudRemoval"
    target_params = {
        "image": optical_image,
        "output_path": "cloud_removal_test_result.tif",
    }
    if nir_image:
        target_params["nir_image"] = nir_image

    test_single_tool(target_tool, target_params)
