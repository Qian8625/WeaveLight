---
name: ChangeSummarySkill
executor_model: ChangeSummarySkill
version: 1
category: bi_temporal_change_analysis
description: Summarize changes between two images using standardized change-detection prompts or a user-supplied query.
required_inputs:
  - pre_image
  - post_image
optional_inputs:
  - task_type
  - query
  - target
defaults:
  task_type: generic
supported_task_types:
  - generic
  - new_ships
  - disappear_or_transfer_aircraft
  - facility_damage_or_expansion
---

# Purpose

Use this skill for text-level change analysis between two chronological images.

# When to use

- The task compares two time points.
- The user asks for added ships, moved aircraft, damaged facilities, or other textual change summaries.
- Instance-level tracking is not strictly required.

# Do not use when

- The task requires exact instance IDs or trajectories.
- The task is single-image detection or measurement.
- The task requires RGB/SAR fusion rather than time comparison.

# Input contract

## Required

- `pre_image`
- `post_image`

## Optional

- `task_type`
- `query`
- `target`

# Output contract

The executor should return some or all of:

- `text`
- `error_code`
- `query_used`
- `task_type`
- `skill_trace`

# Recommended procedure

1. Build a standardized change question based on `task_type`.
2. If `query` is provided, use it directly.
3. Call `ChangeDetection`.
4. Return the textual summary with the query used.

# Fallback strategy

- If the predefined template is not appropriate, use a custom query.
- If the tool fails, return the exact failed query for debugging.

# Tool dependencies

- `ChangeDetection`

# Example

```json
{
  "skill_name": "ChangeSummarySkill",
  "pre_image": "img_t1",
  "post_image": "img_t2",
  "task_type": "new_ships"
}