import json
import os
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tool_server.tool_workers.tool_manager.base_manager import ToolManager

"""
Test script for all registered skill workers.

Design goals:
1. Keep the calling style close to tools_single_test.py / tools_test.py.
2. Run every skill once with a representative example payload.
3. Reuse repo-local fixtures when available, and gracefully skip tests when
   required sample data is missing in the current environment.
4. Print structured results for quick manual verification.
"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
LEGACY_REPO_ROOTS = [
    os.environ.get("OPENEARTHAGENT_ROOT"),
    os.environ.get("WEAVELIGHT_ROOT"),
    "/home/ubuntu/01_Code/OpenEarthAgent",
    "/home/ubuntu/01_Code/WeaveLight",
]


def abs_path(rel_path: str) -> str:
    """Convert a path relative to this script into an absolute path."""
    return os.path.abspath(os.path.join(SCRIPT_DIR, rel_path))


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def resolve_existing_path(*candidates: Optional[str]) -> Optional[str]:
    """
    Resolve the first existing path from a list of absolute/relative candidates.

    Search order for relative paths:
      - relative to this script
      - relative to repository root
      - relative to common legacy roots
    """
    probe_paths: List[str] = []

    for candidate in candidates:
        if not candidate:
            continue
        expanded = os.path.expandvars(os.path.expanduser(str(candidate).strip()))
        if not expanded:
            continue

        probe_paths.append(expanded)
        if not os.path.isabs(expanded):
            probe_paths.append(os.path.join(SCRIPT_DIR, expanded))
            probe_paths.append(os.path.join(REPO_ROOT, expanded))
            for root in LEGACY_REPO_ROOTS:
                if root:
                    probe_paths.append(os.path.join(root, expanded))

    for path in _dedupe_keep_order(probe_paths):
        if os.path.exists(path):
            return os.path.abspath(path)

    return None


def build_scratch_path(scratch_dir: str, filename: str) -> str:
    return os.path.abspath(os.path.join(scratch_dir, filename))


def ensure_rgb_proxy_for_sar(
    tm: ToolManager,
    sar_image: Optional[str],
    preferred_rgb: Optional[str],
    scratch_dir: str,
) -> Optional[str]:
    """
    Try to prepare an RGB-like proxy image for MultConfirmSkill.

    Priority:
      1. existing preferred_rgb fixture
      2. generate via SARToRGB
      3. fallback to SARPreprocessing output
    """
    if preferred_rgb and os.path.isfile(preferred_rgb):
        return preferred_rgb
    if not sar_image or not os.path.isfile(sar_image):
        return None

    candidates: List[Tuple[str, Dict[str, Any]]] = []
    sar_to_rgb_out = build_scratch_path(scratch_dir, "sar_rgb_proxy.png")
    sar_pre_out = build_scratch_path(scratch_dir, "sar_rgb_proxy_preprocessed.png")

    if "SARToRGB" in tm.available_tools:
        candidates.append((
            "SARToRGB",
            {"image": sar_image, "output_path": sar_to_rgb_out},
        ))
    if "SARPreprocessing" in tm.available_tools:
        candidates.append((
            "SARPreprocessing",
            {"image": sar_image, "output_path": sar_pre_out},
        ))

    for tool_name, params in candidates:
        try:
            resp = tm.call_tool(tool_name, params)
            image_path = resp.get("image") or resp.get("output_path")
            if resp.get("error_code") == 0 and image_path and os.path.isfile(image_path):
                print(f"Prepared proxy RGB for SAR using {tool_name}: {image_path}")
                return image_path
            print(f"⚠️  Failed to prepare proxy RGB with {tool_name}: {resp.get('text', '')}")
        except Exception as exc:
            print(f"⚠️  Exception while preparing proxy RGB with {tool_name}: {exc}")

    return None


def discover_fixtures(tm: ToolManager, scratch_dir: str) -> Dict[str, Optional[str]]:
    """Discover sample assets used by the skill test suite."""
    rgb_image = resolve_existing_path(
        "tools_test_images/1.jpg",
        "scripts/tools_test/tools_test_images/1.jpg",
    )
    change_pre = resolve_existing_path(
        "tools_test_images/xBD_cls_1.png",
        "scripts/tools_test/tools_test_images/xBD_cls_1.png",
    )
    change_post = resolve_existing_path(
        "tools_test_images/xBD_cls_2.png",
        "scripts/tools_test/tools_test_images/xBD_cls_2.png",
    )
    geotiff = resolve_existing_path(
        "tools_test_images/S_07.tif",
        "scripts/tools_test/tools_test_images/S_07.tif",
    )

    sar_raw = resolve_existing_path(
        "scripts/tools_test/tools_test_images/sar_test_1.png",
    )
    sar_preprocessed = resolve_existing_path(
        "scripts/tools_test/tools_test_images/sar_test_1_sar_preprocessed.png",
    )
    sar_rgb = resolve_existing_path(
        "/home/ubuntu/01_Code/OpenEarthAgent/scripts/tools_test/tools_test_images/sar_to_rgb_result.png",
        "scripts/tools_test/tools_test_images/sar_to_rgb_result.png",
    )
    sar_rgb = ensure_rgb_proxy_for_sar(
        tm=tm,
        sar_image=sar_raw,
        preferred_rgb=sar_rgb,
        scratch_dir=scratch_dir,
    )

    return {
        "rgb_image": rgb_image,
        "change_pre": change_pre,
        "change_post": change_post,
        "geotiff": geotiff,
        "sar_raw": sar_raw,
        "sar_preprocessed": sar_preprocessed,
        "sar_rgb": sar_rgb,
    }


def required_paths_exist(paths: Iterable[Optional[str]]) -> Tuple[bool, List[str]]:
    missing = [p for p in paths if not p or not os.path.exists(p)]
    return len(missing) == 0, missing


def print_fixture_summary(fixtures: Dict[str, Optional[str]]) -> None:
    print("\n=== Fixture Discovery Summary ===")
    for key, value in fixtures.items():
        status = value if value else "<missing>"
        print(f"- {key}: {status}")
    print()


def build_skill_examples(fixtures: Dict[str, Optional[str]], scratch_dir: str):
    return [
        {
            "skill_name": "TargetLocateMeasureSkill",
            "required_paths": [fixtures["rgb_image"]],
            "params": {
                "image": fixtures["rgb_image"],
                "target": "red car",
                "mode": "locate",
                "detector": "text_to_bbox",
                "top1": True,
                "visualize": True,
                "max_draw": 3,
            },
        },
        {
            "skill_name": "SARTargetLocateMeasureSkill",
            "required_paths": [fixtures["sar_raw"]],
            "params": {
                "image": fixtures["sar_raw"],
                "target": "ship",
                "mode": "locate",
                "preprocess_mode": "sar_preprocess",
                "preprocessed_output_path": build_scratch_path(scratch_dir, "sar_target_preprocessed.png"),
                "top1": True,
                "visualize": True,
                "max_draw": 3,
            },
        },
        {
            "skill_name": "MultConfirmSkill",
            "required_paths": [fixtures["sar_rgb"], fixtures["sar_raw"]],
            "params": {
                "rgb_image": fixtures["sar_rgb"],
                "sar_image": fixtures["sar_raw"],
                "target": "ship",
                "task_type": "compare",
                "preprocess_mode": "sar_preprocess",
                "sar_preprocessed_output_path": build_scratch_path(scratch_dir, "multconfirm_sar_preprocessed.png"),
                "iou_threshold": 0.1,
                "top1": True,
                "visualize": True,
                "max_draw": 3,
            },
        },
        {
            "skill_name": "TargetAttributeSkill",
            "required_paths": [fixtures["rgb_image"]],
            "params": {
                "image": fixtures["rgb_image"],
                "task_type": "filter",
                "target": "car",
                "attribute": "color",
                "attribute_value": "red",
                "top1": False,
                "visualize": True,
                "max_draw": 3,
            },
        },
        {
            "skill_name": "ConditionalCountSkill",
            "required_paths": [fixtures["rgb_image"]],
            "params": {
                "image": fixtures["rgb_image"],
                "target": "car",
                "condition": "red",
                "visualize": True,
                "max_draw": 5,
                "verify_examples": True,
                "max_verify": 2,
            },
        },
        {
            "skill_name": "ChangeSummarySkill",
            "required_paths": [fixtures["change_pre"], fixtures["change_post"]],
            "params": {
                "pre_image": fixtures["change_pre"],
                "post_image": fixtures["change_post"],
                "task_type": "facility_damage_or_expansion",
            },
        },
        {
            "skill_name": "GeoTIFFPoiExploreSkill",
            "required_paths": [fixtures["geotiff"]],
            "params": {
                "geotiff": fixtures["geotiff"],
                "poi_specs": [
                    {"query": {"aeroway": "terminal"}, "layer_name": "terminals"},
                ],
                "task_type": "surrounding_description",
                "show_names": True,
                "describe_rendered": True,
            },
        },
        {
            "skill_name": "GeoTiffPoiDistanceSkill",
            "required_paths": [fixtures["geotiff"]],
            "params": {
                "geotiff": fixtures["geotiff"],
                "poi_specs": [
                    {"query": {"aeroway": "terminal"}, "layer_name": "terminals"},
                    {"query": {"aeroway": "gate"}, "layer_name": "gates"},
                ],
                "src_layer": "terminals",
                "tar_layer": "gates",
                "top": 1,
                "show_names": True,
                "render_distance_layer": True,
            },
        },
    ]


def main():
    print("\n=== Initializing Tool Manager ===")
    tm = ToolManager(controller_url_location=None)
    scratch_dir = tempfile.mkdtemp(prefix="skill_test_")
    print(f"Using scratch directory: {scratch_dir}")

    fixtures = discover_fixtures(tm, scratch_dir)
    print_fixture_summary(fixtures)

    example_calls = build_skill_examples(fixtures, scratch_dir)

    print("\n=== Running tests for all skill workers ===\n")

    skills_with_errors: List[str] = []
    skills_successful: List[str] = []
    skipped_skills: List[str] = []

    skipped = 0
    success = 0
    error = 0

    for item in example_calls:
        skill_name = item["skill_name"]
        params = item["params"]
        required_paths = item.get("required_paths", [])

        if skill_name not in tm.available_tools:
            print(f"⚠️  {skill_name} is not registered or unavailable — skipping.\n")
            skipped += 1
            skipped_skills.append(skill_name)
            continue

        ok, missing = required_paths_exist(required_paths)
        if not ok:
            print(f"⚠️  {skill_name} is skipped because required fixtures are missing:")
            for path in missing:
                print(f"   - {path}")
            print()
            skipped += 1
            skipped_skills.append(skill_name)
            continue

        print(f"🔹 Testing {skill_name} with parameters:")
        print(json.dumps(params, indent=2, ensure_ascii=False))

        try:
            result = tm.call_tool(skill_name, params)

            if result.get("error_code") == 0:
                print(f"✅ {skill_name} executed successfully.")
                print(f"Result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                success += 1
                skills_successful.append(skill_name)
            else:
                print(f"❌ {skill_name} returned an error.")
                print(f"Result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                error += 1
                skills_with_errors.append(skill_name)

        except Exception as exc:
            print(f"❌ Exception while testing {skill_name}: {exc}\n")
            error += 1
            skills_with_errors.append(skill_name)

        print("-" * 100 + "\n")
        time.sleep(0.5)

    print("\n🎯 All example skills executed in sequence.\n")
    print("📊 Summary:")
    print(f"   ✅ Successful : {success}")
    print(f"   ❌ Failed     : {error}")
    print(f"   ⚠️  Skipped    : {skipped}\n")

    if skills_successful:
        print(f"✅ Successful skills:\n   {sorted(set(skills_successful))}\n")

    if skills_with_errors:
        print(f"❌ Failed skills:\n   {sorted(set(skills_with_errors))}\n")

    if skipped_skills:
        print(f"⚠️  Skipped skills:\n   {sorted(set(skipped_skills))}\n")

    if error == 0 and skipped == 0:
        print("🎉 All skills ran successfully without errors.")


if __name__ == "__main__":
    main()
