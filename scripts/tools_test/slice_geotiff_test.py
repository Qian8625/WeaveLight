import json
import os
import tempfile

import numpy as np
import rasterio
from rasterio.transform import from_origin

from tool_server.tool_workers.tool_manager.base_manager import ToolManager

"""
Test script for the SliceGeoTIFF tool worker.
"""

def build_test_inputs(base_dir: str):
    tif_path = os.path.join(base_dir, "synthetic_scene.tif")
    meta_xml_path = os.path.join(base_dir, "synthetic_scene.meta.xml")
    output_dir = os.path.join(base_dir, "tiles")

    height = width = 1024
    data = np.zeros((3, height, width), dtype=np.uint16)
    gradient_x = np.tile(np.linspace(100, 2000, width, dtype=np.uint16), (height, 1))
    gradient_y = np.tile(np.linspace(50, 1800, height, dtype=np.uint16), (width, 1)).T

    data[0] = gradient_x
    data[1] = gradient_y
    data[2] = ((gradient_x.astype(np.uint32) + gradient_y.astype(np.uint32)) // 2).astype(np.uint16)

    transform = from_origin(120.0, 31.0, 0.0001, 0.0001)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    xml_content = """<Root>
  <ProductID>synthetic_scene</ProductID>
  <ProductLevel>L1A</ProductLevel>
  <SatelliteID>GF2</SatelliteID>
  <ImageBeginTime>2026-04-07T12:00:00Z</ImageBeginTime>
  <GroundResampleX>1.0</GroundResampleX>
  <GroundResampleY>1.0</GroundResampleY>
  <NumPixels>1024</NumPixels>
  <NumLines>1024</NumLines>
  <BandNumber>3</BandNumber>
  <SceneCenterLat>30.9488</SceneCenterLat>
  <SceneCenterLong>120.0512</SceneCenterLong>
  <UpperLeftLat>31.0</UpperLeftLat>
  <UpperLeftLong>120.0</UpperLeftLong>
  <UpperRightLat>31.0</UpperRightLat>
  <UpperRightLong>120.1024</UpperRightLong>
  <LowerLeftLat>30.8976</LowerLeftLat>
  <LowerLeftLong>120.0</LowerLeftLong>
  <LowerRightLat>30.8976</LowerRightLat>
  <LowerRightLong>120.1024</LowerRightLong>
  <Cloud CloudPercent="0.0" />
  <SolarElevation>45.0</SolarElevation>
  <SolarAzimuth>135.0</SolarAzimuth>
</Root>
"""
    with open(meta_xml_path, "w", encoding="utf-8") as file:
        file.write(xml_content)

    return tif_path, meta_xml_path, output_dir


def test_slice_geotiff():
    with tempfile.TemporaryDirectory() as tmpdir:
        tif_path, meta_xml_path, output_dir = build_test_inputs(tmpdir)
        tm = ToolManager(controller_url_location=None)
        tool_name = "SliceGeoTIFF"

        if tool_name not in tm.available_tools:
            print(f"⚠️  {tool_name} is not registered or unavailable — exiting.\n")
            print(f"当前可用的工具列表: {tm.available_tools}")
            return

        params = {
            "tif_path": tif_path,
            "meta_xml_path": meta_xml_path,
            "output_dir": output_dir,
            "tile_size": 512,
            "overlap": 0,
            "nodata_threshold": 0.2,
            "use_global_normalization": True,
        }

        print("\n🔹 Testing SliceGeoTIFF with parameters:")
        print(json.dumps(params, indent=2))

        result = tm.call_tool(tool_name, params)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    test_slice_geotiff()
