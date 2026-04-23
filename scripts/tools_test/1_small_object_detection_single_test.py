import json
import os

from tool_server.tool_workers.tool_manager.base_manager import ToolManager


def abs_path(rel_path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), rel_path))


def run_case(tm: ToolManager, tool_name: str, case_name: str, params: dict):
    print(f"\n=== Case: {case_name} ===")
    print(json.dumps(params, indent=2))
    result = tm.call_tool(tool_name, params)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print("\n=== Initializing Tool Manager ===")
    tm = ToolManager(controller_url_location=None)
    tool_name = "SmallObjectDetection"

    if tool_name not in tm.available_tools:
        print(f"⚠️  {tool_name} is not registered or unavailable — exiting.\n")
        print(f"当前可用的工具列表: {tm.available_tools}")
        raise SystemExit(0)

    image_1 = abs_path("tools_test_images/1.jpg")
    image_empty = abs_path("tools_test_images/56.png")

    cases = [
        (
            "full_image",
            {
                "image": image_1,
            },
        ),
        (
            "forced_tiling",
            {
                "image": image_1,
                "use_tiling": True,
                "tile_size": 512,
                "tile_overlap": 64,
            },
        ),
        (
            "text_filter",
            {
                "image": image_1,
                "text": "car",
            },
        ),
        (
            "empty_result",
            {
                "image": image_empty,
                "text": "ship",
            },
        ),
    ]

    summary = {}
    for case_name, params in cases:
        result = run_case(tm, tool_name, case_name, params)
        summary[case_name] = result.get("error_code")

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
