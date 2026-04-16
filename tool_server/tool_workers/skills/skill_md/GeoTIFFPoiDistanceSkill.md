---
name: GeoTIFFPoiDistanceSkill
executor_model: GeoTIFFPoiDistanceSkill
version: 1
category: geotiff_external_knowledge_distance
description: Extract the AOI from a GeoTIFF, retrieve multiple POI classes, compute distance relations, and render the results back onto the GeoTIFF.
required_inputs:
  - geotiff
  - poi_specs
optional_inputs:
  - src_layer
  - tar_layer
  - buffer_m
  - top
  - show_names
  - render_distance_layer
  - layers_to_render
defaults:
  top: 1
  show_names: true
  render_distance_layer: true
---

# Purpose

Use this skill for GeoTIFF-based POI distance tasks, including nearest-neighbor measurement and visualized relation overlays.

# When to use

- The user provides a GeoTIFF and asks to visualize multiple POI types.
- The user asks to compute distance between POI classes.
- The user asks for the nearest POI relation to be annotated on the GeoTIFF.

# Do not use when

- Only POI existence or simple counting is needed. Use `GeoTIFFPoiExploreSkill`.
- The task is not based on a GeoTIFF AOI.

# Input contract

## Required

- `geotiff`
- `poi_specs`

## Optional

- `src_layer`
- `tar_layer`
- `buffer_m`
- `top`
- `show_names`
- `render_distance_layer`
- `layers_to_render`

# Output contract

The executor should return some or all of:

- `text`
- `error_code`
- `gpkg`
- `image`
- `skill_trace`

# Recommended procedure

1. Extract bbox from the GeoTIFF using `GetBboxFromGeotiff`.
2. Create the AOI boundary using `GetAreaBoundary`.
3. Add requested POI layers using `AddPoisLayer`.
4. Compute distance relations using `ComputeDistance`.
5. Render source, target, and optional distance layers back onto the GeoTIFF using `DisplayOnGeotiff`.

# Fallback strategy

- If one POI layer fails, stop and return the partial AOI result.
- If distance computation fails, return the rendered POI layers without relation overlays if possible.
- If rendering fails, still return the `gpkg` and distance text summary.

# Tool dependencies

- `GetBboxFromGeotiff`
- `GetAreaBoundary`
- `AddPoisLayer`
- `ComputeDistance`
- `DisplayOnGeotiff`

# Example

```json
{
  "skill_name": "GeoTIFFPoiDistanceSkill",
  "geotiff": "tif_1",
  "poi_specs": [
    {"query": {"tourism": "museum"}, "layer_name": "museums"},
    {"query": {"shop": "mall"}, "layer_name": "malls"}
  ],
  "top": 1,
  "show_names": true
}