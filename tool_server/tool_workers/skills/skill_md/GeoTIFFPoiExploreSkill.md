---
name: GeoTIFFPoiExploreSkill
executor_model: GeoTIFFPoiExploreSkill
version: 1
category: geotiff_external_knowledge
description: Extract the AOI from a GeoTIFF, retrieve one or more POI classes, optionally render them on the GeoTIFF, and optionally describe the rendered scene.
required_inputs:
  - geotiff
  - poi_specs
optional_inputs:
  - task_type
  - buffer_m
  - show_names
  - describe_rendered
defaults:
  task_type: visualize
  show_names: true
supported_task_types:
  - visualize
  - existence
  - count
  - surrounding_description
---

# Purpose

Use this skill for GeoTIFF-based external knowledge tasks involving POIs and AOI-level rendering.

# When to use

- The input is a GeoTIFF.
- The user asks whether POIs exist in the AOI.
- The user asks to visualize one or more POI classes.
- The user asks to count requested POI classes.
- The user asks for a rendered scene description after overlaying POIs.

# Do not use when

- The task is not GeoTIFF-based.
- The user needs distances between POI classes. Use `GeoTIFFPoiDistanceSkill`.
- The task is target detection inside the raster content rather than external POI retrieval.

# Input contract

## Required

- `geotiff`
- `poi_specs`: list of objects containing `query` and optional `layer_name`

## Optional

- `task_type`
- `buffer_m`
- `show_names`
- `describe_rendered`

# Output contract

The executor should return some or all of:

- `text`
- `error_code`
- `gpkg`
- `poi_summary`
- `total_count`
- `exists`
- `image`
- `environment_description`
- `skill_trace`

# Recommended procedure

1. Extract bbox from the GeoTIFF using `GetBboxFromGeotiff`.
2. Convert the bbox into an AOI GeoPackage using `GetAreaBoundary`.
3. Add one or more POI layers using `AddPoisLayer`.
4. If visualization is needed, render the selected layers using `DisplayOnGeotiff`.
5. If surrounding description is needed, call `ImageDescription` on the rendered output.

# Fallback strategy

- If one POI layer fails, return the successfully added layers and explain the partial failure.
- If rendering fails, still return `gpkg` and POI count summary.
- If description fails, still return visualization and POI summary.

# Tool dependencies

- `GetBboxFromGeotiff`
- `GetAreaBoundary`
- `AddPoisLayer`
- `DisplayOnGeotiff`
- `ImageDescription`

# Example

```json
{
  "skill_name": "GeoTIFFPoiExploreSkill",
  "geotiff": "tif_1",
  "poi_specs": [
    {"query": {"amenity": "hospital"}, "layer_name": "hospitals"},
    {"query": {"tourism": "museum"}, "layer_name": "museums"}
  ],
  "task_type": "surrounding_description"
}