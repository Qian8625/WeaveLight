---
name: ConditionalCountSkill
executor_model: ConditionalCountSkill
version: 1
category: conditional_counting
description: Count targets satisfying a visually describable condition, such as docked ships, parked aircraft, or direction-constrained objects.
required_inputs:
  - image
  - target
  - condition
optional_inputs:
  - bbox
  - visualize
  - max_draw
  - verify_examples
  - max_verify
defaults:
  visualize: true
  max_draw: 10
  verify_examples: false
  max_verify: 3
---

# Purpose

Use this skill to count objects under a condition that can be expressed visually.

# When to use

- The user asks for the number of `docked ships`, `parked aircraft`, or `targets facing east`.
- The task is a counting problem, not a full scene interpretation problem.
- The condition can be expressed in natural language.

# Do not use when

- The task requires fine-grained comparison between two specific objects. Use `TargetAttributeSkill`.
- The task requires exact metric measurements. Use `TargetLocateMeasureSkill`.
- The task is change analysis. Use `ChangeSummarySkill`.

# Input contract

## Required

- `image`
- `target`
- `condition`

## Optional

- `bbox`
- `visualize`
- `max_draw`
- `verify_examples`
- `max_verify`

# Output contract

The executor should return some or all of:

- `text`
- `error_code`
- `count`
- `query`
- `method_used`
- `detections`
- `image`
- `verification_examples`
- `skill_trace`

# Recommended procedure

1. Compose a query such as `docked ship`.
2. Try `CountGivenObject` first.
3. If the count is missing or unreliable, call `TextToBbox` and count detections directly.
4. Optionally verify some detections with `RegionAttributeDescription`.
5. Optionally render boxes with `DrawBox`.

# Fallback strategy

- If `CountGivenObject` fails, fall back to `TextToBbox`.
- If verification fails, still return the estimated count.
- If visualization fails, return the numeric result only.

# Tool dependencies

- `CountGivenObject`
- `TextToBbox`
- `RegionAttributeDescription`
- `DrawBox`

# Example

```json
{
  "skill_name": "ConditionalCountSkill",
  "image": "img_1",
  "target": "ship",
  "condition": "docked",
  "visualize": true,
  "verify_examples": true
}