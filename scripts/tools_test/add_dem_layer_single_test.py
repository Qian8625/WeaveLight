import json
import os
import shutil
import tempfile
from tool_server.tool_workers.tool_manager.base_manager import ToolManager

"""
Test script for the AddDEMLayer tool worker.
"""


def abs_path(rel_path: str) -> str:
    """Convert relative test file paths to absolute paths."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), rel_path))


def make_working_copy(rel_path: str, scratch_dir: str, filename: str | None = None) -> str:
    """Copy a GeoPackage fixture so the test does not mutate tracked sample data."""
    source = abs_path(rel_path)
    target = os.path.join(scratch_dir, filename or os.path.basename(source))
    shutil.copy2(source, target)
    return target


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
            print(f"Result:\n{json.dumps(result, indent=2)}\n")
        else:
            print(f"\n❌ {tool_name} returned an error.")
            print(f"Result:\n{json.dumps(result, indent=2)}\n")

    except Exception as e:
        print(f"\n❌ Exception while testing {tool_name}: {str(e)}\n")


if __name__ == "__main__":
    target_tool = "AddDEMLayer"
    scratch_dir = tempfile.mkdtemp(prefix="add_dem_layer_test_")
    gpkg = make_working_copy("tools_test_images/aoi_1.gpkg", scratch_dir)

    print(f"Using scratch GeoPackage under: {scratch_dir}")

    target_params = {
        "gpkg": gpkg,
        "contour_interval_m": 20,
        "band_step_m": 100,
    }

    test_single_tool(target_tool, target_params)
