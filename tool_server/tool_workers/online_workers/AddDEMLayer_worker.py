import argparse
import math
import os
import re
import uuid

##### network config   ---by wys
os.environ["HTTP_PROXY"] = "http://10.31.215.24:7897"
os.environ["HTTPS_PROXY"] = "http://10.31.215.24:7897"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,192.168.0.0/16"
#####

import ee
import geemap
import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


DEFAULT_DEM_SOURCE = "USGS/SRTMGL1_003"
DEFAULT_NODATA = -32768
DEFAULT_CONTOUR_MIN_LENGTH_M = 60.0
DEFAULT_CONTOUR_SIMPLIFY_M = 15.0
SUPPORTED_DEM_SOURCES = {
    "USGS/SRTMGL1_003": {
        "ee_path": "USGS/SRTMGL1_003",
        "band": "elevation",
        "description": "SRTM 30m DEM",
    }
}


gdal.UseExceptions()

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"AddDEMLayer_worker_{worker_id}.log")


def _load_aoi_geometry(gpkg):
    area_gdf = gpd.read_file(gpkg, layer="area_boundary")
    if area_gdf.empty:
        raise ValueError("'area_boundary' layer is empty")
    if area_gdf.crs is None:
        raise ValueError("'area_boundary' layer has no CRS")
    if area_gdf.crs.to_string() != "EPSG:4326":
        area_gdf = area_gdf.to_crs("EPSG:4326")
    try:
        geom = area_gdf.geometry.union_all()
    except AttributeError:
        geom = area_gdf.geometry.unary_union
    if geom is None or geom.is_empty:
        raise ValueError("Area boundary geometry is empty")
    return geom


def _explode_lines(gdf):
    exploded = gdf.explode(index_parts=False)
    return exploded.reset_index(drop=True)


def _drop_empty_geometries(gdf):
    mask = gdf.geometry.apply(lambda geom: geom is not None and not geom.is_empty)
    return gdf[mask].copy()


def _postprocess_contours(gpkg, layer_name, min_length_m=DEFAULT_CONTOUR_MIN_LENGTH_M, simplify_m=DEFAULT_CONTOUR_SIMPLIFY_M):
    contours = gpd.read_file(gpkg, layer=layer_name)
    if contours.empty:
        return 0

    aoi_geom = _load_aoi_geometry(gpkg)
    if contours.crs is None:
        contours = contours.set_crs("EPSG:4326")
    elif contours.crs.to_string() != "EPSG:4326":
        contours = contours.to_crs("EPSG:4326")

    contours = contours.copy()
    contours["geometry"] = contours.geometry.intersection(aoi_geom)
    contours = _drop_empty_geometries(contours)
    if contours.empty:
        _delete_vector_layer(gpkg, layer_name)
        raise ValueError("Contour generation produced no linework inside the AOI")

    contours = _explode_lines(contours)
    contours = _drop_empty_geometries(contours)

    metric_crs = contours.estimate_utm_crs() or "EPSG:3857"
    contours_metric = contours.to_crs(metric_crs)
    contours_metric = contours_metric[contours_metric.geometry.length > 0].copy()
    if contours_metric.empty:
        _delete_vector_layer(gpkg, layer_name)
        raise ValueError("Contour generation produced only zero-length features")

    if min_length_m and min_length_m > 0:
        contours_metric = contours_metric[contours_metric.geometry.length >= float(min_length_m)].copy()
    if contours_metric.empty:
        _delete_vector_layer(gpkg, layer_name)
        raise ValueError("Contour generation only produced short artifacts after clipping to the AOI")

    if simplify_m and simplify_m > 0:
        contours_metric["geometry"] = contours_metric.geometry.simplify(
            float(simplify_m),
            preserve_topology=False,
        )
        contours_metric = _drop_empty_geometries(contours_metric)
        contours_metric = contours_metric[contours_metric.geometry.length > 0].copy()
    if contours_metric.empty:
        _delete_vector_layer(gpkg, layer_name)
        raise ValueError("Contour generation produced no valid features after simplification")

    contours = contours_metric.to_crs("EPSG:4326")
    contours["geometry"] = contours.geometry.intersection(aoi_geom)
    contours = _drop_empty_geometries(contours)
    contours = _explode_lines(contours)
    contours = _drop_empty_geometries(contours)
    if min_length_m and min_length_m > 0:
        final_lengths = contours.to_crs(metric_crs).geometry.length
        contours = contours.loc[final_lengths >= float(min_length_m)].copy()
    contours = contours[contours.geometry.geom_type == "LineString"].copy()
    if contours.empty:
        _delete_vector_layer(gpkg, layer_name)
        raise ValueError("Contour generation produced no LineString features after cleanup")

    if "id" in contours.columns:
        contours["id"] = pd.Series(range(1, len(contours) + 1), index=contours.index, dtype="int64")
    if "elev_m" in contours.columns:
        contours["elev_m"] = contours["elev_m"].astype(float)

    _delete_vector_layer(gpkg, layer_name)
    contours.to_file(gpkg, layer=layer_name, driver="GPKG")
    return len(contours)


def _delete_vector_layer(gpkg, layer_name):
    ds = ogr.Open(gpkg, 1)
    if ds is None:
        raise RuntimeError(f"Could not open GeoPackage for update: {gpkg}")
    try:
        for idx in range(ds.GetLayerCount()):
            layer = ds.GetLayerByIndex(idx)
            if layer is not None and layer.GetName() == layer_name:
                ds.DeleteLayer(idx)
                break
    finally:
        ds = None


def _export_dem_to_tif(gpkg, dem_source, resolution_m, tif_path):
    geom = _load_aoi_geometry(gpkg)
    aoi = ee.Geometry(geom.__geo_interface__)
    source_cfg = SUPPORTED_DEM_SOURCES[dem_source]
    dem_image = ee.Image(source_cfg["ee_path"]).select(source_cfg["band"]).clip(aoi)
    geemap.ee_export_image(
        dem_image,
        filename=tif_path,
        scale=resolution_m,
        region=aoi,
        file_per_band=False,
        crs="EPSG:4326",
        unmask_value=DEFAULT_NODATA,
        timeout=600,
        verbose=False,
    )
    if not os.path.isfile(tif_path):
        raise RuntimeError(f"DEM export failed: {tif_path} was not created")


def _read_dem_stats(tif_path):
    ds = gdal.Open(tif_path)
    if ds is None:
        raise RuntimeError(f"Could not open DEM raster: {tif_path}")
    try:
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray().astype(np.float32)
        nodata = band.GetNoDataValue()
        valid_mask = np.isfinite(arr)
        if nodata is not None:
            valid_mask &= arr != nodata
        valid = arr[valid_mask]
        if valid.size == 0:
            raise ValueError("DEM contains no valid pixels inside the AOI")
        return float(valid.min()), float(valid.max())
    finally:
        ds = None


def _create_contours(tif_path, gpkg, layer_name, contour_interval_m):
    _delete_vector_layer(gpkg, layer_name)

    raster_ds = gdal.Open(tif_path)
    if raster_ds is None:
        raise RuntimeError(f"Could not open DEM raster for contour generation: {tif_path}")

    vector_ds = ogr.Open(gpkg, 1)
    if vector_ds is None:
        raise RuntimeError(f"Could not open GeoPackage for contour output: {gpkg}")

    try:
        raster_band = raster_ds.GetRasterBand(1)
        nodata = raster_band.GetNoDataValue()
        srs = None
        projection = raster_ds.GetProjection()
        if projection:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(projection)

        contour_layer = vector_ds.CreateLayer(layer_name, srs=srs, geom_type=ogr.wkbLineString)
        contour_layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
        contour_layer.CreateField(ogr.FieldDefn("elev_m", ogr.OFTReal))

        gdal.ContourGenerate(
            raster_band,
            float(contour_interval_m),
            0.0,
            [],
            1 if nodata is not None else 0,
            float(nodata if nodata is not None else 0),
            contour_layer,
            0,
            1,
        )
        raw_count = contour_layer.GetFeatureCount()
    finally:
        vector_ds = None
        raster_ds = None

    if raw_count == 0:
        return 0
    return _postprocess_contours(gpkg, layer_name)


def _build_band_raster(dem_tif_path, band_tif_path, band_step_m):
    dem_ds = gdal.Open(dem_tif_path)
    if dem_ds is None:
        raise RuntimeError(f"Could not open DEM raster for elevation bands: {dem_tif_path}")

    try:
        dem_band = dem_ds.GetRasterBand(1)
        arr = dem_band.ReadAsArray().astype(np.float32)
        nodata = dem_band.GetNoDataValue()

        valid_mask = np.isfinite(arr)
        if nodata is not None:
            valid_mask &= arr != nodata
        valid = arr[valid_mask]
        if valid.size == 0:
            raise ValueError("DEM contains no valid pixels for elevation bands")

        base_elev = math.floor(float(valid.min()) / band_step_m) * band_step_m
        class_arr = np.zeros(arr.shape, dtype=np.int32)
        class_arr[valid_mask] = (
            np.floor((arr[valid_mask] - base_elev) / band_step_m).astype(np.int32) + 1
        )

        driver = gdal.GetDriverByName("GTiff")
        band_ds = driver.Create(
            band_tif_path,
            dem_ds.RasterXSize,
            dem_ds.RasterYSize,
            1,
            gdal.GDT_Int32,
            options=["COMPRESS=LZW"],
        )
        if band_ds is None:
            raise RuntimeError(f"Could not create temporary elevation-band raster: {band_tif_path}")

        band_ds.SetGeoTransform(dem_ds.GetGeoTransform())
        band_ds.SetProjection(dem_ds.GetProjection())
        out_band = band_ds.GetRasterBand(1)
        out_band.WriteArray(class_arr)
        out_band.SetNoDataValue(0)
        out_band.FlushCache()
        band_ds.FlushCache()
        band_ds = None

        return base_elev
    finally:
        dem_ds = None


def _polygonize_band_raster(band_tif_path, raw_gpkg_path, raw_layer_name):
    driver = ogr.GetDriverByName("GPKG")
    if os.path.exists(raw_gpkg_path):
        driver.DeleteDataSource(raw_gpkg_path)

    vector_ds = driver.CreateDataSource(raw_gpkg_path)
    if vector_ds is None:
        raise RuntimeError(f"Could not create temporary band GeoPackage: {raw_gpkg_path}")

    raster_ds = gdal.Open(band_tif_path)
    if raster_ds is None:
        raise RuntimeError(f"Could not open temporary band raster: {band_tif_path}")

    try:
        raster_band = raster_ds.GetRasterBand(1)
        srs = None
        projection = raster_ds.GetProjection()
        if projection:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(projection)

        raw_layer = vector_ds.CreateLayer(raw_layer_name, srs=srs, geom_type=ogr.wkbPolygon)
        raw_layer.CreateField(ogr.FieldDefn("band_id", ogr.OFTInteger))
        gdal.Polygonize(raster_band, None, raw_layer, 0, [], callback=None)
    finally:
        vector_ds = None
        raster_ds = None


def _create_elevation_bands(dem_tif_path, gpkg, layer_name, band_step_m, band_tif_path, raw_gpkg_path):
    base_elev = _build_band_raster(dem_tif_path, band_tif_path, band_step_m)
    raw_layer_name = "raw_elevation_bands"
    _polygonize_band_raster(band_tif_path, raw_gpkg_path, raw_layer_name)

    gdf = gpd.read_file(raw_gpkg_path, layer=raw_layer_name)
    if gdf.empty:
        raise ValueError("No elevation-band polygons were generated")

    gdf = gdf[gdf["band_id"] > 0].copy()
    gdf["band_id"] = gdf["band_id"].astype(int)
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]
    if gdf.empty:
        raise ValueError("Elevation-band polygons only contained nodata regions")

    gdf = gdf.dissolve(by="band_id", as_index=False)
    gdf["elev_min"] = base_elev + (gdf["band_id"] - 1) * band_step_m
    gdf["elev_max"] = gdf["elev_min"] + band_step_m
    gdf["band_label"] = gdf.apply(
        lambda row: f"[{int(row['elev_min'])}, {int(row['elev_max'])}) m",
        axis=1,
    )

    _delete_vector_layer(gpkg, layer_name)
    gdf.to_file(gpkg, layer=layer_name, driver="GPKG")
    return len(gdf)


def _safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
        for sidecar_path in (
            f"{path}.aux.xml",
            f"{path}.ovr",
            f"{path}-wal",
            f"{path}-shm",
            f"{path}-journal",
        ):
            if path and os.path.exists(sidecar_path):
                os.remove(sidecar_path)
    except OSError:
        logger.warning(f"Failed to remove temporary file: {path}")


def AddDEMLayer(
    gpkg,
    dem_layer_name="dem_30m",
    contour_layer_name="dem_contours_20m",
    band_layer_name="dem_bands_100m",
    resolution_m=30,
    contour_interval_m=20,
    band_step_m=100,
    dem_source=DEFAULT_DEM_SOURCE,
):
    # Kept for backward compatibility with existing callers. The DEM raster is no longer persisted.
    _ = dem_layer_name

    if dem_source not in SUPPORTED_DEM_SOURCES:
        raise ValueError(
            f"Unsupported dem_source '{dem_source}'. "
            f"Supported values: {list(SUPPORTED_DEM_SOURCES.keys())}"
        )
    if resolution_m <= 0 or contour_interval_m <= 0 or band_step_m <= 0:
        raise ValueError("resolution_m, contour_interval_m, and band_step_m must be positive numbers")

    temp_id = uuid.uuid4().hex[:8]
    work_dir = os.path.dirname(os.path.abspath(gpkg))
    dem_tif_path = os.path.join(work_dir, f"_add_dem_{temp_id}.tif")
    band_tif_path = os.path.join(work_dir, f"_add_dem_bands_{temp_id}.tif")
    raw_bands_gpkg = os.path.join(work_dir, f"_add_dem_bands_{temp_id}.gpkg")

    try:
        _export_dem_to_tif(gpkg, dem_source, resolution_m, dem_tif_path)
        elev_min, elev_max = _read_dem_stats(dem_tif_path)
        contour_count = _create_contours(dem_tif_path, gpkg, contour_layer_name, contour_interval_m)
        band_count = _create_elevation_bands(
            dem_tif_path,
            gpkg,
            band_layer_name,
            band_step_m,
            band_tif_path,
            raw_bands_gpkg,
        )

        gpkg_name = os.path.basename(gpkg)
        summary = (
            f"DEM from {dem_source} analyzed for {gpkg_name} (temporary raster not persisted)\n"
            f"Elevation range inside AOI: {elev_min:.2f} m to {elev_max:.2f} m\n"
            f"Contour layer '{contour_layer_name}' created with interval {float(contour_interval_m):g} m "
            f"({contour_count} features)\n"
            f"Elevation-band layer '{band_layer_name}' created with step {float(band_step_m):g} m "
            f"({band_count} bands)"
        )
        return {
            "text": summary,
            "created_layers": [contour_layer_name, band_layer_name],
            "elev_min_m": round(elev_min, 2),
            "elev_max_m": round(elev_max, 2),
        }
    finally:
        _safe_remove(dem_tif_path)
        _safe_remove(band_tif_path)
        _safe_remove(raw_bands_gpkg)


class AddDEMLayerWorker(BaseToolWorker):
    def __init__(
        self,
        controller_addr,
        worker_addr="auto",
        worker_id=worker_id,
        no_register=False,
        model_name="AddDEMLayer",
        host="0.0.0.0",
        port=None,
        limit_model_concurrency=2,
        model_semaphore=None,
        service_account=None,
        key_file=None,
        wait_timeout=300.0,
        task_timeout=300.0,
    ):
        self.service_account = service_account
        self.key_file = key_file
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            None,
            None,
            model_name,
            False,
            False,
            "cpu",
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

    def init_model(self):
        if not self.service_account or not self.key_file:
            txt_e = "Missing service account or key file path."
            logger.error(txt_e)
            raise ValueError(txt_e)

        if not os.path.exists(self.key_file):
            txt_e = f"Key file not found: {self.key_file}"
            logger.error(txt_e)
            raise FileNotFoundError(txt_e)

        try:
            credentials = ee.ServiceAccountCredentials(self.service_account, self.key_file)
            ee.Initialize(credentials)
            ee.Number(1).getInfo()
            logger.info("Earth Engine authenticated using service account.")
            logger.info("AddDEMLayerWorker initialization successful. Ready to run.")
        except Exception as e:
            logger.error(f"Failed to initialize Earth Engine: {e}")
            raise

    def generate(self, params):
        if "gpkg" not in params:
            txt_e = "Missing required parameter: gpkg"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        gpkg = params.get("gpkg")
        dem_layer_name = params.get("dem_layer_name", "dem_30m")
        contour_interval_m = float(params.get("contour_interval_m", 20))
        band_step_m = float(params.get("band_step_m", 100))
        resolution_m = float(params.get("resolution_m", 30))
        contour_layer_name = params.get(
            "contour_layer_name",
            f"dem_contours_{int(contour_interval_m)}m",
        )
        band_layer_name = params.get(
            "band_layer_name",
            f"dem_bands_{int(band_step_m)}m",
        )
        dem_source = params.get("dem_source", DEFAULT_DEM_SOURCE)

        if not os.path.exists(gpkg):
            txt_e = "GeoPackage not found"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 3}

        try:
            result = AddDEMLayer(
                gpkg=gpkg,
                dem_layer_name=dem_layer_name,
                contour_layer_name=contour_layer_name,
                band_layer_name=band_layer_name,
                resolution_m=resolution_m,
                contour_interval_m=contour_interval_m,
                band_step_m=band_step_m,
                dem_source=dem_source,
            )
            cleaned_txt = re.sub(r" {2,}", " ", result["text"]).strip()
            return {
                "text": cleaned_txt,
                "created_layers": result["created_layers"],
                "elev_min_m": result["elev_min_m"],
                "elev_max_m": result["elev_max_m"],
                "error_code": 0,
            }
        except Exception as e:
            txt_e = f"Error in AddDEMLayer: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": "AddDEMLayer",
                "description": (
                    "Fetch a DEM from Google Earth Engine for the AOI stored in a GeoPackage, "
                    "use it to create contour and elevation-band layers, and report elevation statistics."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gpkg": {
                            "type": "string",
                            "description": "Path to the GeoPackage. Must contain an 'area_boundary' layer.",
                        },
                        "dem_layer_name": {
                            "type": "string",
                            "description": "Deprecated and ignored. The DEM raster is only used internally and is not persisted.",
                        },
                        "contour_layer_name": {
                            "type": "string",
                            "description": "Output contour line layer name. Defaults to 'dem_contours_<interval>m'.",
                        },
                        "band_layer_name": {
                            "type": "string",
                            "description": "Output elevation-band polygon layer name. Defaults to 'dem_bands_<step>m'.",
                        },
                        "resolution_m": {
                            "type": "number",
                            "description": "DEM export resolution in meters. Default is 30.",
                        },
                        "contour_interval_m": {
                            "type": "number",
                            "description": "Contour interval in meters. Default is 20.",
                        },
                        "band_step_m": {
                            "type": "number",
                            "description": "Elevation-band step in meters. Default is 100.",
                        },
                        "dem_source": {
                            "type": "string",
                            "enum": list(SUPPORTED_DEM_SOURCES.keys()),
                            "description": f"DEM source in GEE. Default is '{DEFAULT_DEM_SOURCE}'.",
                        },
                    },
                    "required": ["gpkg"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20027)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="AddDEMLayer")
    parser.add_argument("--service_account", type=str, default="")
    parser.add_argument("--key_file", type=str, default="")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = AddDEMLayerWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_name=args.model_name,
        service_account=args.service_account,
        key_file=args.key_file,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()
