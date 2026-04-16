---
name: MultConfirmSkill
executor_model: MultConfirmSkill
version: 1
category: cross_modal_fusion
description: Detect the same target in RGB and SAR branches, compare their outputs, and produce a fused cross-modal conclusion.
required_inputs:
  - rgb_image
  - sar_image
  - target
optional_inputs:
  - task_type
  - preprocess_mode
  - sar_preprocessed_output_path
  - iou_threshold
  - top1
  - detector
  - visualize
  - max_draw
defaults:
  task_type: confirm
  preprocess_mode: sar_preprocess
  iou_threshold: 0.2
  top1: false
  detector: auto
  visualize: true
  max_draw: 10
supported_task_types:
  - confirm
  - compare
  - fuse
---

# Purpose

Use this skill when the user wants RGB and SAR to support, compare, or fuse evidence for the same target.

# When to use

- The task explicitly mentions both RGB and SAR.
- The user asks SAR to confirm a suspected RGB target.
- The user asks to compare differences across modalities.
- The two images are approximately aligned.

# Do not use when

- Only a single modality is available.
- The task is pure SAR analysis. Use `SARTargetLocateMeasureSkill`.
- The task is pure single-image locating. Use `TargetLocateMeasureSkill`.

# Input contract

## Required

- `rgb_image`
- `sar_image`
- `target`

## Optional

- `task_type`: `confirm`, `compare`, `fuse`
- `preprocess_mode`
- `sar_preprocessed_output_path`
- `iou_threshold`
- `top1`
- `detector`
- `visualize`
- `max_draw`

# Output contract

The executor should return some or all of:

- `text`
- `error_code`
- `rgb_detections`
- `sar_detections`
- `matched_pairs`
- `rgb_only`
- `sar_only`
- `preprocessed_sar_image`
- `rgb_image_annotated`
- `sar_image_annotated`
- `image`
- `skill_trace`

# Recommended procedure

1. Run the RGB branch using `TargetLocateMeasureSkill`.
2. Preprocess SAR using `SARPreprocessing` or `SARToRGB`.
3. Run the processed SAR branch using `TargetLocateMeasureSkill`.
4. Match detections across branches using IoU.
5. Produce one of three outputs:
   - confirmation result
   - difference comparison
   - fused conclusion

# Fallback strategy

- If RGB succeeds but SAR fails, return an RGB-only partial conclusion.
- If SAR preprocessing fails, explain that cross-modal fusion could not complete.
- If no pairs match, still report RGB-only and SAR-only candidates.

# Tool dependencies

- `TargetLocateMeasureSkill`
- `SARPreprocessing`
- `SARToRGB`

# Example

```json
{
  "skill_name": "MultConfirmSkill",
  "rgb_image": "img_rgb",
  "sar_image": "img_sar",
  "target": "ship",
  "task_type": "fuse",
  "iou_threshold": 0.2
}