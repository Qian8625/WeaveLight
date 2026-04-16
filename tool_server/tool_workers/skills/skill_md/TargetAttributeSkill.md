---
name: TargetAttributeSkill
executor_model: TargetAttributeSkill
version: 1
category: attribute_interpretation
description: Filter targets by attribute, compare attributes between selected targets, or describe a scene and count a target class.
required_inputs:
  - image
  - task_type
optional_inputs:
  - target
  - attribute
  - attribute_value
  - compare_mode
  - selector_a
  - selector_b
  - bbox
  - top1
  - visualize
  - max_draw
  - use_segmentation
defaults:
  visualize: true
  top1: false
  max_draw: 10
  use_segmentation: false
supported_task_types:
  - filter
  - compare
  - describe_and_count
---

# Purpose

Use this skill for tasks involving target attributes rather than only geometry.

# When to use

- The user asks for attribute-based filtering, such as `red ship` or `damaged aircraft`.
- The user asks to compare an attribute across targets.
- The user asks to describe the scene and estimate how many objects of one class are present.

# Do not use when

- The task is primarily about metric distance or area. Use `TargetLocateMeasureSkill`.
- The task requires SAR preprocessing. Use `SARTargetLocateMeasureSkill`.
- The task requires bi-temporal change analysis. Use `ChangeSummarySkill`.

# Input contract

## Required

- `image`
- `task_type`: `filter`, `compare`, or `describe_and_count`

## Optional

- `target`
- `attribute`
- `attribute_value`
- `compare_mode`
- `selector_a`
- `selector_b`
- `bbox`
- `top1`
- `visualize`
- `max_draw`
- `use_segmentation`

# Output contract

The executor should return some or all of:

- `text`
- `error_code`
- `detections`
- `image`
- `selected_pair`
- `attribute_a`
- `attribute_b`
- `same_attribute`
- `scene_description`
- `count_target`
- `count_value`
- `count_raw_text`
- `skill_trace`

# Recommended procedure

## For `filter`

1. Build a textual query such as `red ship`.
2. Call `TextToBbox`.
3. Optionally render the detections with `DrawBox`.

## For `compare`

1. Detect all targets of the requested class.
2. Select two targets, usually `smallest` vs `largest`.
3. Call `RegionAttributeDescription` for both.
4. Compare normalized attribute texts.

## For `describe_and_count`

1. Call `ImageDescription`.
2. If a target class is given, call `CountGivenObject`.
3. Return both scene description and count result.

# Fallback strategy

- If target filtering finds nothing, return an empty result with explanation.
- If comparison cannot find at least two targets, return a clear error.
- If counting fails, still return the scene description.

# Tool dependencies

- `TextToBbox`
- `DrawBox`
- `RegionAttributeDescription`
- `CountGivenObject`
- `ImageDescription`

# Example

```json
{
  "skill_name": "TargetAttributeSkill",
  "image": "img_1",
  "task_type": "compare",
  "target": "ship",
  "attribute": "color",
  "compare_mode": "smallest_vs_largest",
  "visualize": true
}