---
name: TargetLocateMeasureSkill
executor_model: TargetLocateMeasureSkill
version: 1
category: target_detection_and_measurement
description: Locate targets in a single image, optionally estimate area, perform segmentation-oriented measurement, and compute center-to-center distance.
required_inputs:
  - image
  - target
  - mode
optional_inputs:
  - reference_target
  - gsd_m_per_pixel
  - visualize
  - top1
  - detector
  - max_draw
defaults:
  visualize: true
  top1: false
  detector: auto
  max_draw: 10
supported_modes:
  - locate
  - area
  - segment
  - distance
---

# Purpose

Use this skill for recurring single-image target tasks such as locating ships, aircraft, vehicles, or facilities; estimating object area; generating segmentation-oriented measurements; and computing target-to-target distance.

# When to use

- The task is about one image.
- The user asks to find, box, segment, measure, or compare spacing between targets.
- The target can be described in natural language, such as `large ship`, `red aircraft`, or `damaged building`.
- The task can be solved by chaining standard detection and measurement tools.

# Do not use when

- The task requires SAR-specific preprocessing first. Use `SARTargetLocateMeasureSkill` instead.
- The task requires two time points. Use `ChangeSummarySkill` instead.
- The task requires RGB/SAR evidence fusion. Use `MultConfirmSkill` instead.
- The user asks for real-world distance or area but no `gsd_m_per_pixel` is available. In that case, return pixel-space results only.

# Input contract

## Required

- `image`: path to the input image.
- `target`: natural-language target description.
- `mode`: one of `locate`, `area`, `segment`, `distance`.

## Optional

- `reference_target`: used when `mode=distance` and the user wants distance to a different target class.
- `gsd_m_per_pixel`: meters per pixel for real-world conversion.
- `visualize`: whether to return an annotated image.
- `top1`: whether to keep only the top detection.
- `detector`: `auto`, `text_to_bbox`, or `object_detection`.
- `max_draw`: maximum number of boxes to render.

# Output contract

The executor should return some or all of:

- `text`
- `error_code`
- `detections`
- `image`
- `pixel_counts`
- `bbox_pixel_areas`
- `distance_px`
- `distance_m`
- `closest_pair`
- `total_area_m2`
- `skill_trace`

# Recommended procedure

1. Detect the primary target using `TextToBbox`.
2. If needed, fall back to `ObjectDetection`.
3. If `mode=locate`, return detections and optionally render boxes using `DrawBox`.
4. If `mode=area` or `mode=segment`, call `SegmentObjectPixels`.
5. If `gsd_m_per_pixel` is provided, use `Calculator` to convert pixel area to square meters.
6. If `mode=distance`, detect the reference target set and compute the minimum center-to-center distance.
7. If `gsd_m_per_pixel` is provided, use `Calculator` to convert pixel distance to meters.
8. If `visualize=true`, render the selected detections with `DrawBox`.

# Fallback strategy

- If `TextToBbox` returns no valid detections, try `ObjectDetection`.
- If segmentation fails, still return the detections and explain that only locating succeeded.
- If GSD is missing, explicitly state that distance/area are reported in pixel units only.
- If visualization fails, return measurement results without annotated imagery.

# Tool dependencies

- `TextToBbox`
- `ObjectDetection`
- `SegmentObjectPixels`
- `Calculator`
- `DrawBox`

# Example

```json
{
  "skill_name": "TargetLocateMeasureSkill",
  "image": "img_1",
  "target": "large ship",
  "mode": "distance",
  "reference_target": "small ship",
  "gsd_m_per_pixel": 0.6,
  "visualize": true
}