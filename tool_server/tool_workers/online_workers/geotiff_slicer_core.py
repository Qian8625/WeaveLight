import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window


def _deep_update(base: Dict, extra: Dict) -> Dict:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        elif value is not None:
            base[key] = value
    return base


class GeoTIFFMetadataParser:
    @staticmethod
    def parse_meta_xml(xml_path: str) -> Dict:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        return {
            "product_id": root.find(".//ProductID").text if root.find(".//ProductID") is not None else None,
            "product_level": root.find(".//ProductLevel").text if root.find(".//ProductLevel") is not None else None,
            "satellite_id": root.find(".//SatelliteID").text if root.find(".//SatelliteID") is not None else None,
            "image_time": root.find(".//ImageBeginTime").text if root.find(".//ImageBeginTime") is not None else None,
            "resolution": {
                "x": float(root.find(".//GroundResampleX").text) if root.find(".//GroundResampleX") is not None else None,
                "y": float(root.find(".//GroundResampleY").text) if root.find(".//GroundResampleY") is not None else None,
            },
            "image_size": {
                "width": int(root.find(".//NumPixels").text) if root.find(".//NumPixels") is not None else None,
                "height": int(root.find(".//NumLines").text) if root.find(".//NumLines") is not None else None,
            },
            "bands": int(root.find(".//BandNumber").text) if root.find(".//BandNumber") is not None else None,
            "center_coord": {
                "lat": float(root.find(".//SceneCenterLat").text) if root.find(".//SceneCenterLat") is not None else None,
                "lon": float(root.find(".//SceneCenterLong").text) if root.find(".//SceneCenterLong") is not None else None,
            },
            "corners": {
                "upper_left": {
                    "lat": float(root.find(".//UpperLeftLat").text) if root.find(".//UpperLeftLat") is not None else None,
                    "lon": float(root.find(".//UpperLeftLong").text) if root.find(".//UpperLeftLong") is not None else None,
                },
                "upper_right": {
                    "lat": float(root.find(".//UpperRightLat").text) if root.find(".//UpperRightLat") is not None else None,
                    "lon": float(root.find(".//UpperRightLong").text) if root.find(".//UpperRightLong") is not None else None,
                },
                "lower_left": {
                    "lat": float(root.find(".//LowerLeftLat").text) if root.find(".//LowerLeftLat") is not None else None,
                    "lon": float(root.find(".//LowerLeftLong").text) if root.find(".//LowerLeftLong") is not None else None,
                },
                "lower_right": {
                    "lat": float(root.find(".//LowerRightLat").text) if root.find(".//LowerRightLat") is not None else None,
                    "lon": float(root.find(".//LowerRightLong").text) if root.find(".//LowerRightLong") is not None else None,
                },
            },
            "cloud_percent": float(root.find(".//Cloud").get("CloudPercent")) if root.find(".//Cloud") is not None else None,
            "sun_elevation": float(root.find(".//SolarElevation").text) if root.find(".//SolarElevation") is not None else None,
            "sun_azimuth": float(root.find(".//SolarAzimuth").text) if root.find(".//SolarAzimuth") is not None else None,
        }

    @staticmethod
    def parse_rpb(rpb_path: str) -> Dict:
        with open(rpb_path, "r", encoding="utf-8") as file:
            content = file.read()

        def extract_value(key: str) -> Optional[float]:
            pattern = rf"{key}\s*=\s*([+-]?\d+\.?\d*[eE]?[+-]?\d*)"
            match = re.search(pattern, content)
            return float(match.group(1)) if match else None

        rpc_params = {
            "line_offset": extract_value("lineOffset"),
            "samp_offset": extract_value("sampOffset"),
            "lat_offset": extract_value("latOffset"),
            "long_offset": extract_value("longOffset"),
            "height_offset": extract_value("heightOffset"),
            "line_scale": extract_value("lineScale"),
            "samp_scale": extract_value("sampScale"),
            "lat_scale": extract_value("latScale"),
            "long_scale": extract_value("longScale"),
            "height_scale": extract_value("heightScale"),
        }
        return rpc_params

    @staticmethod
    def build_raster_metadata(tif_path: str) -> Dict:
        with rasterio.open(tif_path) as src:
            west = south = east = north = None
            if src.crs is not None:
                try:
                    west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
                except Exception:
                    west, south, east, north = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
            else:
                west, south, east, north = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top

            return {
                "product_id": Path(tif_path).stem,
                "product_level": None,
                "satellite_id": None,
                "image_time": None,
                "resolution": {
                    "x": abs(src.transform.a) if src.transform else None,
                    "y": abs(src.transform.e) if src.transform else None,
                },
                "image_size": {
                    "width": src.width,
                    "height": src.height,
                },
                "bands": src.count,
                "center_coord": {
                    "lat": (north + south) / 2 if north is not None and south is not None else None,
                    "lon": (east + west) / 2 if east is not None and west is not None else None,
                },
                "corners": {
                    "upper_left": {"lat": north, "lon": west},
                    "upper_right": {"lat": north, "lon": east},
                    "lower_left": {"lat": south, "lon": west},
                    "lower_right": {"lat": south, "lon": east},
                },
                "cloud_percent": None,
                "sun_elevation": None,
                "sun_azimuth": None,
                "crs": str(src.crs) if src.crs is not None else None,
                "bounds": {
                    "west": west,
                    "south": south,
                    "east": east,
                    "north": north,
                },
            }

    def load_metadata_for_tif(
        self,
        tif_path: str,
        meta_xml_path: Optional[str] = None,
        rpb_path: Optional[str] = None,
    ) -> Dict:
        tif_file = Path(tif_path)
        resolved_meta_xml = Path(meta_xml_path) if meta_xml_path else tif_file.with_suffix(".meta.xml")
        resolved_rpb = Path(rpb_path) if rpb_path else tif_file.with_suffix(".rpb")

        metadata = self.build_raster_metadata(tif_path)
        if resolved_meta_xml.exists():
            metadata = _deep_update(metadata, self.parse_meta_xml(str(resolved_meta_xml)))
        if resolved_rpb.exists():
            metadata["rpc_params"] = self.parse_rpb(str(resolved_rpb))
        return metadata


class GeoTIFFSlicer:
    def __init__(self, tif_path: str, metadata: Dict, tile_size: int = 512, overlap: int = 0):
        if tile_size <= 0:
            raise ValueError("tile_size must be positive.")
        if overlap < 0:
            raise ValueError("overlap must be non-negative.")
        if overlap >= tile_size:
            raise ValueError("overlap must be smaller than tile_size.")

        self.tif_path = tif_path
        self.metadata = metadata
        self.tile_size = tile_size
        self.overlap = overlap
        self.image_width = metadata["image_size"]["width"]
        self.image_height = metadata["image_size"]["height"]

    def calculate_tile_coordinates(self, tile_row: int, tile_col: int) -> Dict:
        corners = self.metadata["corners"]

        x_start = tile_col * (self.tile_size - self.overlap)
        y_start = tile_row * (self.tile_size - self.overlap)
        x_end = min(x_start + self.tile_size, self.image_width)
        y_end = min(y_start + self.tile_size, self.image_height)

        x_ratio_start = x_start / self.image_width
        y_ratio_start = y_start / self.image_height
        x_ratio_end = x_end / self.image_width
        y_ratio_end = y_end / self.image_height

        def interpolate_coord(x_ratio: float, y_ratio: float) -> Dict:
            top_lat = corners["upper_left"]["lat"] * (1 - x_ratio) + corners["upper_right"]["lat"] * x_ratio
            top_lon = corners["upper_left"]["lon"] * (1 - x_ratio) + corners["upper_right"]["lon"] * x_ratio
            bottom_lat = corners["lower_left"]["lat"] * (1 - x_ratio) + corners["lower_right"]["lat"] * x_ratio
            bottom_lon = corners["lower_left"]["lon"] * (1 - x_ratio) + corners["lower_right"]["lon"] * x_ratio
            lat = top_lat * (1 - y_ratio) + bottom_lat * y_ratio
            lon = top_lon * (1 - y_ratio) + bottom_lon * y_ratio
            return {"lat": lat, "lon": lon}

        return {
            "upper_left": interpolate_coord(x_ratio_start, y_ratio_start),
            "upper_right": interpolate_coord(x_ratio_end, y_ratio_start),
            "lower_left": interpolate_coord(x_ratio_start, y_ratio_end),
            "lower_right": interpolate_coord(x_ratio_end, y_ratio_end),
            "center": interpolate_coord((x_ratio_start + x_ratio_end) / 2, (y_ratio_start + y_ratio_end) / 2),
        }

    def _normalize_to_uint8(self, data: np.ndarray, global_stats: Optional[Dict] = None) -> np.ndarray:
        if data.dtype == np.uint8:
            return data

        if len(data.shape) == 3:
            normalized = np.zeros_like(data, dtype=np.uint8)
            for i in range(data.shape[2]):
                band_data = data[:, :, i]
                finite = band_data[np.isfinite(band_data)]
                if finite.size == 0:
                    continue

                if global_stats and i in global_stats:
                    vmin = global_stats[i]["vmin"]
                    vmax = global_stats[i]["vmax"]
                else:
                    vmin = np.percentile(finite, 2)
                    vmax = np.percentile(finite, 98)

                if vmax > vmin:
                    band_norm = np.clip((band_data - vmin) / (vmax - vmin) * 255, 0, 255)
                    normalized[:, :, i] = band_norm.astype(np.uint8)
            return normalized

        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return np.zeros_like(data, dtype=np.uint8)

        if global_stats and 0 in global_stats:
            vmin = global_stats[0]["vmin"]
            vmax = global_stats[0]["vmax"]
        else:
            vmin = np.percentile(finite, 2)
            vmax = np.percentile(finite, 98)

        if vmax > vmin:
            normalized = np.clip((data - vmin) / (vmax - vmin) * 255, 0, 255)
            return normalized.astype(np.uint8)
        return np.zeros_like(data, dtype=np.uint8)

    def _build_global_stats(self, src: rasterio.DatasetReader) -> Optional[Dict]:
        sample_factor = 10
        sample_height = max(1, src.height // sample_factor)
        sample_width = max(1, src.width // sample_factor)

        if src.count >= 3:
            sampled_data = src.read([3, 2, 1], out_shape=(3, sample_height, sample_width))
        else:
            sampled_data = src.read(out_shape=(src.count, sample_height, sample_width))

        global_stats = {}
        for i in range(sampled_data.shape[0]):
            band_data = sampled_data[i]
            finite = band_data[np.isfinite(band_data)]
            if finite.size == 0:
                global_stats[i] = {"vmin": 0.0, "vmax": 0.0}
                continue
            global_stats[i] = {
                "vmin": float(np.percentile(finite, 2)),
                "vmax": float(np.percentile(finite, 98)),
            }
        return global_stats

    def slice_geotiff(
        self,
        output_dir: str,
        nodata_threshold: float = 0.2,
        use_global_normalization: bool = True,
    ) -> List[Dict]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        step = self.tile_size - self.overlap
        n_cols = max(1, (max(self.image_width - self.overlap, 0) + step - 1) // step)
        n_rows = max(1, (max(self.image_height - self.overlap, 0) + step - 1) // step)

        tiles_info = []

        with rasterio.open(self.tif_path) as src:
            global_stats = self._build_global_stats(src) if use_global_normalization else None

            for row in range(n_rows):
                for col in range(n_cols):
                    x_start = col * step
                    y_start = row * step
                    if x_start >= src.width or y_start >= src.height:
                        continue

                    x_size = min(self.tile_size, src.width - x_start)
                    y_size = min(self.tile_size, src.height - y_start)

                    if x_size < self.tile_size or y_size < self.tile_size:
                        continue

                    window = Window(x_start, y_start, x_size, y_size)
                    if src.count >= 3:
                        tile_data = src.read([3, 2, 1], window=window)
                    else:
                        tile_data = src.read(window=window)

                    if len(tile_data.shape) == 3:
                        tile_data = np.transpose(tile_data, (1, 2, 0))

                    if len(tile_data.shape) == 3:
                        gray = tile_data.mean(axis=2)
                        nodata_mask = np.all(tile_data == src.nodata, axis=2) if src.nodata is not None else np.zeros_like(gray, dtype=bool)
                    else:
                        gray = tile_data
                        nodata_mask = tile_data == src.nodata if src.nodata is not None else np.zeros_like(gray, dtype=bool)

                    dark_mask = gray < 10
                    nodata_ratio = np.logical_or(dark_mask, nodata_mask).mean()
                    if nodata_ratio > nodata_threshold:
                        continue

                    tile_normalized = self._normalize_to_uint8(tile_data, global_stats)
                    tile_filename = f"tile_r{row:03d}_c{col:03d}.png"
                    tile_path = output_path / tile_filename

                    if len(tile_normalized.shape) == 3 and tile_normalized.shape[2] == 3:
                        Image.fromarray(tile_normalized).save(tile_path)
                    elif len(tile_normalized.shape) == 2:
                        Image.fromarray(tile_normalized, mode="L").save(tile_path)
                    else:
                        Image.fromarray(tile_normalized[:, :, 0], mode="L").save(tile_path)

                    tiles_info.append(
                        {
                            "tile_id": f"r{row:03d}_c{col:03d}",
                            "file_path": str(tile_path),
                            "pixel_bounds": {
                                "x_start": x_start,
                                "y_start": y_start,
                                "x_end": x_start + x_size,
                                "y_end": y_start + y_size,
                            },
                            "geo_coordinates": self.calculate_tile_coordinates(row, col),
                            "size": {
                                "width": x_size,
                                "height": y_size,
                            },
                            "nodata_ratio": float(nodata_ratio),
                        }
                    )

        return tiles_info


def process_single_geotiff(
    tif_path: str,
    output_dir: str,
    meta_xml_path: Optional[str] = None,
    rpb_path: Optional[str] = None,
    tile_size: int = 512,
    overlap: int = 0,
    nodata_threshold: float = 0.2,
    use_global_normalization: bool = True,
) -> Dict:
    tif_file = Path(tif_path)
    if not tif_file.is_file():
        raise FileNotFoundError(f"GeoTIFF not found: {tif_path}")

    parser = GeoTIFFMetadataParser()
    metadata = parser.load_metadata_for_tif(str(tif_file), meta_xml_path=meta_xml_path, rpb_path=rpb_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    slicer = GeoTIFFSlicer(str(tif_file), metadata, tile_size=tile_size, overlap=overlap)
    tiles_info = slicer.slice_geotiff(
        str(output_path),
        nodata_threshold=nodata_threshold,
        use_global_normalization=use_global_normalization,
    )

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    tiles_index_file = output_path / "tiles_index.json"
    with open(tiles_index_file, "w", encoding="utf-8") as file:
        json.dump(tiles_info, file, indent=2, ensure_ascii=False)

    return {
        "source_file": str(tif_file),
        "output_dir": str(output_path),
        "metadata": metadata,
        "metadata_file": str(metadata_file),
        "tiles": tiles_info,
        "tiles_index_file": str(tiles_index_file),
        "tile_count": len(tiles_info),
    }
