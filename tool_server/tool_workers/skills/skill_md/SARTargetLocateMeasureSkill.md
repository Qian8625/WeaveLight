---
name: SARTargetLocateMeasureSkill
executor_model: SARTargetLocateMeasureSkill
version: 1
category: sar_target_detection_and_measurement
description: Preprocess a SAR image and then perform locating, area estimation, segmentation-oriented measurement, or center-to-center distance estimation.
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
  - preprocess_mode
  - preprocessed_output_path
defaults:
  visualize: true
  top1: false
  detector: auto
  max_draw: 10
  preprocess_mode: sar_preprocess
supported_modes:
  - locate
  - area
  - segment
  - distance
---

# Purpose

Use this skill when the input is a SAR image and the downstream task is still target locating or measurement.

# When to use

- The input modality is SAR.
- The task is to find ships, aircraft, or other targets in SAR.
- The task requires distance or area estimation after preprocessing.
- The task can be solved by preprocessing SAR and then reusing the standard single-image locating pipeline.

# Do not use when

- The input is already a normal RGB or optical image.
- The task requires RGB/SAR comparison. Use `MultConfirmSkill`.
- The task is bi-temporal change analysis. Use `ChangeSummarySkill`.

# Input contract

## Required

- `image`: path to the SAR image.
- `target`: target description.
- `mode`: `locate`, `area`, `segment`, or `distance`.

## Optional

- `reference_target`
- `gsd_m_per_pixel`
- `visualize`
- `top1`
- `detector`
- `max_draw`
- `preprocess_mode`: `sar_preprocess` or `sar_to_rgb`
- `preprocessed_output_path`

# Output contract

The executor should return some or all of:

- `text`
- `error_code`
- `preprocessed_image`
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

1. Preprocess the SAR image using `SARPreprocessing` or `SARToRGB`.
2. Feed the processed image to `TargetLocateMeasureSkill`.
3. Reuse the same logic for locate / area / segment / distance.
4. Return both the processed image path and the downstream result.

# Fallback strategy

- If `SARPreprocessing` fails, try `SARToRGB` if allowed.
- If the SAR branch succeeds but metric conversion is impossible, return pixel-space results only.
- If visualization fails, still return measurement outputs.

# Tool dependencies

- `SARPreprocessing`
- `SARToRGB`
- `TargetLocateMeasureSkill`

# Example

```json
{
  "skill_name": "SARTargetLocateMeasureSkill",
  "image": "img_1",
  "target": "ship",
  "mode": "area",
  "preprocess_mode": "sar_preprocess",
  "gsd_m_per_pixel": 0.6
}