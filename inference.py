import numpy as np
from joblib import parallel_backend
import datacube
import xarray as xr
from joblib import load
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datacube.testutils.io import rio_slurp_xarray
from deafrica_tools.bandindices import calculate_indices
from deafrica_tools.classification import predict_xr
import pandas as pd
import seaborn as sns
import warnings
import logging
import inspect
import os
import traceback
from dask.distributed import Client, LocalCluster
from deafrica_tools.dask import create_local_dask_cluster
from collections import defaultdict
from deafrica_tools.datahandling import load_ard
from odc.stac import stac_load
import pystac_client
from deafrica_tools.plotting import rgb
import streamlit as st
from odc.stac import stac_load, configure_rio
from pystac_client import Client as STACClient

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

import logging
import inspect
from datacube.api import query as dc_query

_LOG = logging.getLogger(__name__)

try:
    from skimage import exposure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    _LOG.warning("scikit-image not available, some enhancements will be limited")

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    _LOG.warning("OpenCV not available, some enhancements will be limited")

# Define global parameters - SINGLE DEFINITION
buffer = 0.025  
dask_chunks = {'x': 500, 'y': 500}  # Reduced from 1000x1000 (smaller chunks)
measurements = ['blue', 'green', 'red', 'nir', 'swir_1', 'swir_2']
output_crs = 'epsg:6933'
class_labels = ['Water', 'Forestry', 'Urban', 'Barren', 'Crop', 'Other']
class_colors = {
    'Water': 'blue',
    'Forestry': 'green',
    'Urban': 'red',
    'Barren': 'yellow',
    'Crop': 'purple',
    'Other': 'grey'
}

# --- STAC aliases for GeoMAD products so returned dataset has canonical names
STAC_BAND_ALIASES = {
    'gm_ls5_ls7_annual': {
        'blue': 'SR_B1', 'green': 'SR_B2', 'red': 'SR_B3',
        'nir': 'SR_B4', 'swir_1': 'SR_B5', 'swir_2': 'SR_B7'
    },
    'gm_ls8_ls9_annual': {
        'blue': 'SR_B2', 'green': 'SR_B3', 'red': 'SR_B4',
        'nir': 'SR_B5', 'swir_1': 'SR_B6', 'swir_2': 'SR_B7'
    },
    'gm_s2_annual': {
        'blue': 'B02', 'green': 'B03', 'red': 'B04',
        'nir': 'B08', 'swir_1': 'B11', 'swir_2': 'B12'
    }
}

# Template stac_cfg we will pass to odc.stac.load (per product)
def make_stac_cfg(product):
    aliases = STAC_BAND_ALIASES.get(product, {})
    # odc.stac expects mapping from canonical -> asset name in stac_cfg under "aliases"
    return {
        'aliases': aliases,
        # hint to stac_load to not attempt heavy decoding, but keep assets lazy
        'assets': {'*': {'data_type': 'uint16', 'nodata': 0}}
    }


# Add memory optimization
os.environ.update({
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'GDAL_HTTP_MAX_RETRY': '2',
    'GDAL_HTTP_RETRY_DELAY': '1',
    'GDAL_HTTP_TIMEOUT': '30',
    'AWS_NO_SIGN_REQUEST': 'YES',
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
})

# --- lazy datacube
_dc = None

def get_datacube(app_name="prediction", config=None):
    global _dc
    if _dc is None:
        _LOG.info("Creating Datacube instance (lazy): app=%s", app_name)
        _dc = datacube.Datacube(app=app_name, config=config)
    return _dc

# --- small dask helper to be called from app.py
def init_dask_cluster(n_workers=1, threads_per_worker=1, memory_limit="32GB"):
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        processes=False
    )
    client = Client(cluster)
    return client

# --- query-compatibility helper
def _map_query_keys_for_datacube(query):
    """Make the query keys compatible with the installed datacube.Query."""
    valid_keys = getattr(dc_query.Query, "_valid_keys", None)
    if valid_keys is None:
        sig = inspect.signature(dc_query.Query.__init__)
        valid_keys = set(sig.parameters.keys()) - {"self", "kwargs", "like"}
    else:
        valid_keys = set(valid_keys)

    q = dict(query)
    if "time" in q:
        t = q["time"]
        if isinstance(t, (str, int)) and len(str(t)) == 4:
            year = str(t)
            q["time"] = (f"{year}-01-01", f"{year}-12-31")
        elif isinstance(t, str) and "-" in t and len(t.split("-")) == 3:
            y = t.split("-")[0]
            q["time"] = (f"{y}-01-01", f"{y}-12-31")

    if "time" in q and "time" not in valid_keys:
        for alt in ("time_range", "time_period", "period"):
            if alt in valid_keys:
                q[alt] = q.pop("time")
                _LOG.info("Mapped 'time' -> '%s' for this datacube version", alt)
                break
        else:
            _LOG.warning("Datacube Query does not accept 'time'. Valid keys: %s", valid_keys)

    unknown = [k for k in q.keys() if k not in valid_keys and k not in ("product",)]
    if unknown:
        _LOG.warning("Dropping unknown query keys not accepted by Query: %s", unknown)
        for k in unknown:
            q.pop(k, None)
    return q

# Load models and training data
S_model_path = 'S_model(1).joblib'
L_model_path = 'L_model(1).joblib'
training_data = "L_training_data(1).txt"

try:
    landsat_model = load(L_model_path).set_params(n_jobs=1)
    sentinel_model = load(S_model_path).set_params(n_jobs=1)
except Exception as e:
    logging.warning(f"Could not load models at import: {e}")
    landsat_model = None
    sentinel_model = None

# Initialize Datacube
dc = None

import re
_CANONICAL_BANDS = ['blue', 'green', 'red', 'nir', 'swir_1', 'swir_2']

def _harmonize_band_names(ds, dc, products):
    """
    Ensure the dataset `ds` contains canonical band names:
    ['blue','green','red','nir','swir_1','swir_2'].
    The loader (stac_load) will usually return canonical names if stac_cfg aliases
    are used; this function performs checks and fallback renames.
    """
    # if ds is None, just return
    if ds is None:
        return ds

    present = set(ds.data_vars)
    # quick path: if canonical names already present, return
    if set(_CANONICAL_BANDS).issubset(present):
        return ds

    # attempt to rename common sensor band names to canonical names
    rename_map = {}
    inv_map = {}
    # build reverse map using STAC_BAND_ALIASES for products list
    for p in products:
        aliases = STAC_BAND_ALIASES.get(p, {})
        for canon, asset_name in aliases.items():
            inv_map[asset_name] = canon
            # also accept common variations like 'SR_B4', 'B04', etc.
            inv_map[asset_name.upper()] = canon
            inv_map[asset_name.replace('SR_','').upper()] = canon

    # try to map ds.data_vars -> canonical
    for v in list(ds.data_vars):
        if v in _CANONICAL_BANDS:
            continue
        # direct mapping
        if v in inv_map:
            rename_map[v] = inv_map[v]
        # map if underlying variable contains asset name substring
        else:
            for k, canon in inv_map.items():
                if k.lower() in v.lower():
                    rename_map[v] = canon
                    break

    if rename_map:
        try:
            ds = ds.rename(rename_map)
        except Exception:
            _LOG.warning("Band renaming failed: %s", rename_map, exc_info=True)

    # final check - if still missing critical bands, log them
    present_after = set(ds.data_vars)
    missing = set(['red', 'nir']) - present_after
    if missing:
        _LOG.warning("After harmonisation still missing: %s. Present: %s", missing, present_after)

    return ds


def get_geomad_product_for_year(year):
    year = int(year)
    if 1984 <= year <= 2012:
        return 'gm_ls5_ls7_annual'
    elif 2017 <= year:
        return 'gm_s2_annual'
    else:
        return None

def feature_layers(query, model):
    """
    STAC-first feature loader that:
    - picks GeoMAD product by year
    - queries DEA STAC for items
    - uses odc.stac.stac_load with per-product aliases so returned Dataset has canonical band names (red, nir, etc.)
    - returns a lazy xarray.Dataset chunked by dask_chunks
    - has STAC-first feature loader with image enhancement for Landsat data.
    """
    dc = get_datacube(app_name='feature_layers')

    # determine year from query
    year = None
    if "time" in query:
        t = query["time"]
        if isinstance(t, (str, int)) and len(str(t)) == 4:
            year = int(t)
        elif isinstance(t, (tuple, list)) and len(t) == 2:
            year = int(str(t[0])[:4])
    if year is None:
        raise ValueError("Could not determine year from query.")

    product = get_geomad_product_for_year(year)
    if product is None:
        raise ValueError(f"No GeoMAD product available for year {year}")

    # build bbox and time arguments
    orig_query = dict(query or {})
    bbox = None
    time_arg = None
    try:
        if 'x' in orig_query and 'y' in orig_query:
            x = orig_query['x']; y = orig_query['y']
            if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
                minx, maxx = float(min(x)), float(max(x))
                miny, maxy = float(min(y)), float(max(y))
            else:
                cx = float(x if not isinstance(x, (list, tuple)) else x[0])
                cy = float(y if not isinstance(y, (list, tuple)) else y[0])
                minx, maxx = cx - buffer, cx + buffer
                miny, maxy = cy - buffer, cy + buffer
            bbox = [minx, miny, maxx, maxy]
        if 'time' in orig_query:
            t = orig_query['time']
            if isinstance(t, (list, tuple)) and len(t) == 2:
                time_arg = f"{t[0]}/{t[1]}"
            elif isinstance(t, str) and len(t) == 4 and t.isdigit():
                time_arg = f"{t}-01-01/{t}-12-31"
            elif isinstance(t, int):
                time_arg = f"{int(t)}-01-01/{int(t)}-12-31"
            else:
                time_arg = str(t)
    except Exception as e:
        _LOG.warning("Could not build bbox/time: %s", e)

    # configure rio for DEA STAC
    try:
        configure_rio(cloud_defaults=True, aws={"aws_unsigned": True}, AWS_S3_ENDPOINT="s3.af-south-1.amazonaws.com")
    except Exception:
        _LOG.debug("configure_rio failed (non-fatal)", exc_info=True)

    stac_url = "https://explorer.digitalearth.africa/stac"
    client = STACClient.open(stac_url)
    _LOG.info("Querying STAC %s collection=%s bbox=%s time=%s", stac_url, product, bbox, time_arg)

    # STAC search
    search = client.search(collections=[product], bbox=bbox, datetime=time_arg, max_items=10)
    items = list(search.get_items())
    _LOG.info("STAC search found %d items for product %s", len(items), product)
    if not items:
        raise ValueError(f"STAC search returned 0 items for collection={product} bbox={bbox} time={time_arg}")

    # HIGHER RESOLUTION: Use native resolution for each product
    if product.startswith('gm_ls'):
        resolution = 30  # Landsat native
    elif product.startswith('gm_s2'):
        resolution = 10  # Try for highest available Sentinel-2 resolution
    else:
        resolution = 10
    
    stac_cfg = make_stac_cfg(product)

    try:
        ds = stac_load(
            items,
            bands=list(STAC_BAND_ALIASES[product].keys()),
            crs=output_crs,
            resolution=resolution,
            chunks={'x': 500, 'y': 500},
            groupby=None,
            stac_cfg=stac_cfg,
            bbox=bbox,
            resampling='cubic'  # Use cubic resampling for better quality
        )
        _LOG.info("odc.stac.load returned vars: %s", list(getattr(ds, 'data_vars', {}).keys()))
    except Exception as e:
        _LOG.warning("Failed to load at %dm resolution, trying fallback resolution...", resolution)
        # Fallback to standard resolution
        resolution = 30 if product.startswith('gm_ls') else 20
        ds = stac_load(
            items,
            bands=list(STAC_BAND_ALIASES[product].keys()),
            crs=output_crs,
            resolution=resolution,
            chunks={'x': 500, 'y': 500},
            groupby=None,
            stac_cfg=stac_cfg,
            bbox=bbox,
            resampling='cubic'  # Use cubic resampling for better quality
        )

    # Harmonize band names
    ds = _harmonize_band_names(ds, dc, [product])
    _LOG.info("Bands after harmonisation: %s", list(ds.data_vars))

    # Ensure critical bands exist
    selected_bands = [b for b in _CANONICAL_BANDS if b in ds.data_vars]
    if 'red' not in ds.data_vars or 'nir' not in ds.data_vars:
        raise ValueError(f"Missing 'red' or 'nir' after harmonisation. Available: {list(ds.data_vars)}")

    ds = ds[selected_bands]

    # compute spectral indices
    satellite_mission = 's2' if model == sentinel_model else 'ls'
    da = calculate_indices(ds, index=['NDVI', 'LAI', 'MNDWI'], drop=False, satellite_mission=satellite_mission)

    # attach slope
    try:
        url_slope = "https://deafrica-input-datasets.s3.af-south-1.amazonaws.com/srtm_dem/srtm_africa_slope.tif"
        slope = rio_slurp_xarray(url_slope, geobox=ds.odc.geobox)
        slope = slope.to_dataset(name='slope').chunk({'x': 500, 'y': 500})
        result = xr.merge([da, slope], compat='override')
    except Exception:
        _LOG.warning("Could not load slope dataset; returning indices without slope")
        result = da

    # Apply image enhancement for Landsat data
    if product.startswith('gm_ls'):
        _LOG.info("Applying image enhancement for Landsat data")
        try:
            # Apply contrast stretching and sharpening to Landsat bands
            enhanced_bands = []
            for band in ['red', 'green', 'blue', 'nir']:
                if band in result.data_vars:
                    band_data = result[band]
                    # Apply histogram equalization for better contrast
                    enhanced_band = _enhance_landsat_band(band_data)
                    enhanced_bands.append((band, enhanced_band))
            
            # Replace original bands with enhanced ones
            for band_name, enhanced_data in enhanced_bands:
                result[band_name] = enhanced_data
                
        except Exception as e:
            _LOG.warning("Image enhancement failed: %s", e)

    # chunk result
    try:
        result = result.chunk({'x': 500, 'y': 500})
        _LOG.info("Feature layers data chunked as dask arrays at %dm resolution", resolution)
    except Exception as e:
        _LOG.warning("Could not chunk result as dask arrays: %s", e)

    return result.squeeze()

def _enhance_landsat_band(band_data):
    """
    Apply image enhancement to Landsat bands to improve visual quality.
    """
    try:
        # Materialize the data for processing
        if hasattr(band_data.data, 'compute'):
            data_array = band_data.data.compute()
        else:
            data_array = band_data.values
        
        # Remove outliers using percentiles
        p2, p98 = np.nanpercentile(data_array, [2, 98])
        data_clipped = np.clip(data_array, p2, p98)
        
        # Normalize to 0-1 range
        data_normalized = (data_clipped - p2) / (p98 - p2 + 1e-8)
        
        # Apply gamma correction for better contrast (gamma < 1 brightens, > 1 darkens)
        gamma = 0.8  # Slight brightening for Landsat
        data_gamma = np.power(data_normalized, gamma)
        
        # Apply unsharp masking for sharpening
        from scipy import ndimage
        blurred = ndimage.gaussian_filter(data_gamma, sigma=1)
        sharpened = data_gamma + (data_gamma - blurred) * 1.5  # Sharpening factor
        
        # Clip to valid range
        sharpened = np.clip(sharpened, 0, 1)
        
        # Convert back to original data type and range
        enhanced_data = sharpened * (p98 - p2) + p2
        
        # Create new DataArray with enhanced data
        enhanced_da = xr.DataArray(
            enhanced_data,
            coords=band_data.coords,
            dims=band_data.dims,
            attrs=band_data.attrs
        )
        
        return enhanced_da
        
    except Exception as e:
        _LOG.warning("Band enhancement failed, returning original: %s", e)
        return band_data


# PREDICTION FUNCTIONS
def predict_small_data(model, data_numpy):
    """Process small datasets in one batch (robust to different predict_proba shapes)."""
    # Flatten the data for prediction
    data_flat = []
    vars_list = list(data_numpy.data_vars)
    for var in vars_list:
        var_data = data_numpy[var].values
        # ensure we have a 2D array (y,x) or compatible
        arr = np.asarray(var_data)
        data_flat.append(arr.reshape(-1))

    # Stack features
    X = np.column_stack(data_flat)

    # Handle NaN and infinite values
    valid_mask = ~np.any(np.isnan(X), axis=1) & ~np.any(np.isinf(X), axis=1) & ~np.any(np.abs(X) > 1e10, axis=1)
    
    X_valid = X[valid_mask]

    if X_valid.shape[0] == 0:
        raise RuntimeError("No valid pixels found for prediction")

    # Log data statistics for debugging
    _LOG.info("Data statistics - Min: %f, Max: %f, Mean: %f, Std: %f", 
              np.nanmin(X), np.nanmax(X), np.nanmean(X), np.nanstd(X))
    _LOG.info("Number of invalid pixels (NaN/Inf/Extreme): %d", np.sum(~valid_mask))

    # Convert to float32 and clip extreme values to prevent overflow
    X_valid = X_valid.astype(np.float32)
    
    # Clip extreme values to safe range for float32
    X_valid = np.clip(X_valid, -1e10, 1e10)

    # Predict
    _LOG.info("Running model prediction on %d valid pixels", X_valid.shape[0])
    preds_flat = model.predict(X_valid)

    # Validate predictions
    if len(preds_flat) == 0:
        raise RuntimeError("Model returned empty predictions")

    # Ensure predictions are within valid class range
    if np.any(preds_flat < 0) or np.any(preds_flat >= len(class_labels)):
        _LOG.warning("Some predictions are outside valid class range [0, %d]", len(class_labels)-1)
        # Clip predictions to valid range
        preds_flat = np.clip(preds_flat, 0, len(class_labels)-1)

    # Number of classes according to the fitted model
    try:
        n_classes = len(model.classes_)
    except Exception:
        # fallback: infer from preds or set 2
        n_classes = int(max(preds_flat.max() + 1, 2))

    # Try to get probabilities; be defensive
    probabilities_flat = None
    try:
        prob_raw = model.predict_proba(X_valid)
        prob_arr = np.asarray(prob_raw)
        # if 1-D, turn into 2-D (N,1)
        if prob_arr.ndim == 1:
            prob_arr = prob_arr.reshape(-1, 1)
        # If shape mismatch, attempt to reconcile:
        if prob_arr.ndim == 2 and prob_arr.shape[0] == X_valid.shape[0] and prob_arr.shape[1] == n_classes:
            probabilities_flat = prob_arr
        else:
            # When shapes do not match, try to expand or warn and fallback
            _LOG.warning("predict_proba returned unexpected shape %s, expected (%d samples, %d classes). Falling back to zeros.",
                         prob_arr.shape, X_valid.shape[0], n_classes)
            probabilities_flat = np.zeros((X_valid.shape[0], n_classes), dtype=np.float32)
    except Exception as e:
        _LOG.warning("Model has no predict_proba or it failed: %s. Filling probabilities with zeros.", e)
        probabilities_flat = np.zeros((X_valid.shape[0], n_classes), dtype=np.float32)

    # Create output arrays
    # Determine pixel shape from one of the input variables (assume 2D y,x)
    sample_shape = tuple(np.asarray(data_numpy[vars_list[0]].shape))
    if len(sample_shape) == 1:
        # degenerate/1D case: treat as (n,1)
        pred_shape = (sample_shape[0], 1)
    else:
        pred_shape = sample_shape

    predictions = np.full(pred_shape, -1, dtype=np.int8)
    probabilities = np.full(pred_shape + (n_classes,), -1, dtype=np.float32)

    # Fill valid pixels
    # flatten indexing consistent with earlier reshape
    flat_pred = np.full(pred_shape[0] * (pred_shape[1] if len(pred_shape) > 1 else 1), -1, dtype=np.int8)
    flat_pred[valid_mask] = preds_flat
    predictions = flat_pred.reshape(pred_shape)

    # Fill probabilities
    flat_probs = np.full((flat_pred.size, n_classes), -1, dtype=np.float32)
    flat_probs[valid_mask, :] = probabilities_flat
    probabilities = flat_probs.reshape(pred_shape + (n_classes,))

    # Create xarray Dataset
    coords = {}
    # preserve coords where possible
    if 'y' in data_numpy.coords:
        coords['y'] = data_numpy.y
    if 'x' in data_numpy.coords:
        coords['x'] = data_numpy.x
    predicted = xr.Dataset({
        'Predictions': (['y', 'x'], predictions),
        'Probabilities': (['y', 'x', 'class'], probabilities)
    }, coords=coords)

    # Add input data (keeps metadata; doesn't duplicate big arrays)
    predicted = xr.merge([predicted, data_numpy])
    
    return predicted


def predict_large_data_chunked(model, data_xr):
    """
    Process large datasets in memory-bounded chunks.
    Slices each variable's dask array and computes only small tiles.
    """
    vars_list = list(data_xr.data_vars)
    if len(vars_list) == 0:
        raise RuntimeError("No data variables available for chunked prediction")

    # derive y,x sizes from first data variable (no compute)
    sample_da = data_xr[vars_list[0]].squeeze(drop=True)
    shape = tuple(sample_da.shape)
    if len(shape) != 2:
        # try to collapse leading singleton dims if present
        sample_da = sample_da.squeeze(drop=True)
        shape = tuple(sample_da.shape)
    n_y, n_x = int(shape[0]), int(shape[1])
    _LOG.info("Processing data with shape (y,x) = (%d,%d) in chunks", n_y, n_x)

    # chunk tuning
    chunk_size = 256

    predictions = np.full((n_y, n_x), -1, dtype=np.int8)
    try:
        n_classes = len(model.classes_)
    except Exception:
        n_classes = 2

    probabilities = np.full((n_y, n_x, n_classes), -1, dtype=np.float32)

    for i in range(0, n_y, chunk_size):
        i_end = min(i + chunk_size, n_y)
        for j in range(0, n_x, chunk_size):
            j_end = min(j + chunk_size, n_x)
            _LOG.info("Processing chunk [%d:%d, %d:%d]", i, i_end, j, j_end)

            # collect chunk slices for each variable, compute only that tile
            chunk_arrays = []
            for var in vars_list:
                try:
                    # direct dask array slicing is efficient
                    dask_arr = data_xr[var].data
                    small = dask_arr[i:i_end, j:j_end].compute()
                except Exception:
                    # fallback via xarray isel/values
                    small = data_xr[var].isel(y=slice(i, i_end), x=slice(j, j_end)).values
                chunk_arrays.append(np.asarray(small).reshape(-1))

            X_chunk = np.column_stack(chunk_arrays)
            
            # ADDED: Handle NaN, infinite, and extreme values
            valid_mask = ~np.any(np.isnan(X_chunk), axis=1) & ~np.any(np.isinf(X_chunk), axis=1) & ~np.any(np.abs(X_chunk) > 1e10, axis=1)
            
            if valid_mask.sum() == 0:
                continue

            X_valid = X_chunk[valid_mask]
            
            # ADDED: Convert to float32 and clip extreme values
            X_valid = X_valid.astype(np.float32)
            X_valid = np.clip(X_valid, -1e10, 1e10)
            
            preds_flat = model.predict(X_valid)

            # robust predict_proba handling
            try:
                prob_raw = model.predict_proba(X_valid)
                prob_arr = np.asarray(prob_raw)
                if prob_arr.ndim == 1:
                    prob_arr = prob_arr.reshape(-1, 1)
                if prob_arr.shape[0] != X_valid.shape[0] or (prob_arr.ndim == 2 and prob_arr.shape[1] != n_classes):
                    _LOG.warning("predict_proba returned unexpected shape %s for chunk; using zeros", prob_arr.shape)
                    prob_arr = np.zeros((X_valid.shape[0], n_classes), dtype=np.float32)
            except Exception as e:
                _LOG.warning("predict_proba failed on chunk: %s; using zeros", e)
                prob_arr = np.zeros((X_valid.shape[0], n_classes), dtype=np.float32)

            # reconstruct chunk outputs
            chunk_shape = (i_end - i, j_end - j)
            pred_chunk_2d = np.full(chunk_shape, -1, dtype=np.int8)
            prob_chunk_3d = np.full(chunk_shape + (n_classes,), -1, dtype=np.float32)

            valid_mask_2d = valid_mask.reshape(chunk_shape)
            pred_chunk_2d[valid_mask_2d] = preds_flat
            for cls in range(n_classes):
                prob_chunk_3d[..., cls][valid_mask_2d] = prob_arr[:, cls]

            predictions[i:i_end, j:j_end] = pred_chunk_2d
            probabilities[i:i_end, j:j_end, :] = prob_chunk_3d

    # assemble xarray Dataset with coords if available
    coords = {}
    if 'y' in data_xr.coords:
        coords['y'] = data_xr.y
    if 'x' in data_xr.coords:
        coords['x'] = data_xr.x

    predicted = xr.Dataset({
        'Predictions': (['y', 'x'], predictions),
        'Probabilities': (['y', 'x', 'class'], probabilities)
    }, coords=coords)

    predicted = xr.merge([predicted, data_xr])
    return predicted


def predict_for_location(lat, lon, year):
    """Main prediction function."""
    try:
        _LOG.info("predict_for_location called with lat=%r lon=%r year=%r", lat, lon, year)

        if isinstance(lat, dict) and lon is None and year is None:
            d = lat
            lat = d.get('lat') or d.get('y') or d.get('latitude')
            lon = d.get('lon') or d.get('x') or d.get('longitude')
            year = d.get('year') or d.get('time')

        try:
            lat = float(lat)
            lon = float(lon)
        except Exception:
            raise ValueError(f"Invalid lat/lon: lat={lat}, lon={lon}")

        try:
            year_int = int(str(year)[:4])
            year_str = str(year_int)
        except Exception:
            raise ValueError(f"Cannot parse year from {repr(year)}")

        model = landsat_model if year_int < 2017 else sentinel_model
        if model is None:
            raise RuntimeError("Model(s) not loaded")

        query = {
            'x': (lon - buffer, lon + buffer),
            'y': (lat + buffer, lat - buffer),
            'time': year_str
        }
        _LOG.info("predict_for_location: built datacube query: %s", query)

        data = feature_layers(query, model)
        if data is None or len(data.data_vars) == 0:
            raise RuntimeError(f"No feature data for {lat},{lon},{year_str}")

        expected_features = list(measurements) + ['NDVI', 'LAI', 'MNDWI', 'slope']
        present = [f for f in expected_features if f in data.data_vars]
        if ('red' not in data.data_vars) or ('nir' not in data.data_vars):
            raise RuntimeError(f"Missing critical bands red/nir in loaded data: {list(data.data_vars)}")

        features = [f for f in expected_features if f in data.data_vars]
        data = data[features]

        # Keep data lazy here. Decide path based on total pixel count (approx)
        _LOG.info("Keeping data as lazy xarray.Dataset. Will materialize only chunks during prediction.")
        data_numpy = data  # data is lazy/chunked xarray.Dataset

        # get shape from dataarray (no compute); use .shape property
        sample_da = data_numpy[list(data_numpy.data_vars)[0]]
        ds_shape = tuple(sample_da.shape)
        _LOG.info("Data lazy shape: %s", ds_shape)
        total_pixels = int(np.prod(ds_shape))

        if total_pixels == 0:
            raise RuntimeError("No pixels to process - data appears to be empty")

        _LOG.info("Total pixels to process: %d", total_pixels)
        
        if total_pixels > 1000000:
            _LOG.info("Data too large, processing in chunks...")
            result = predict_large_data_chunked(model, data_numpy)
        else:
            _LOG.info("Processing data in one batch")
            result = predict_small_data(model, data_numpy)

        # Validate the result
        if result is None or 'Predictions' not in result.data_vars:
            raise RuntimeError("Prediction failed - no results generated")

        return result

    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(f"Prediction failed for {year}: {e}\nTraceback:\n{tb}")

def _materialize_da_to_numpy(da):
    """Helper to safely convert DataArray to numpy array."""
    try:
        if hasattr(da, "data") and hasattr(da.data, "compute"):
            return da.data.compute()
        elif hasattr(da, "values") and hasattr(da.values, "compute"):
            return da.values.compute()
        else:
            return np.asarray(da)
    except Exception as e:
        _LOG.warning("Could not materialize DataArray: %s", e)
        try:
            return np.asarray(da)
        except Exception:
            return None

def plot_year_result(prediction_data, year):
    """Generate high-resolution 3-panel plot with enhanced image processing"""
    # Increase DPI for higher resolution
    fig, axes = plt.subplots(1, 3, figsize=(30, 10), dpi=300)
    
    # Ensure the data has proper 2D dimensions for plotting
    plot_data = prediction_data.copy()
    
    # Squeeze out any singleton dimensions that might cause issues
    for var in plot_data.data_vars:
        plot_data[var] = plot_data[var].squeeze()
    
    # Classified image - with higher quality
    cmap = mcolors.ListedColormap([class_colors[label] for label in class_labels])
    norm = mcolors.BoundaryNorm(range(len(class_labels) + 1), cmap.N)
    
    # Ensure Predictions is 2D
    predictions_2d = plot_data.Predictions
    if predictions_2d.ndim > 2:
        predictions_2d = predictions_2d.squeeze()
    
    # Use nearest neighbor interpolation for crisp edges in classification
    im = axes[0].imshow(predictions_2d, cmap=cmap, norm=norm, interpolation='nearest')
    cbar = fig.colorbar(im, ax=axes[0], ticks=range(len(class_labels)))
    cbar.ax.set_yticklabels(class_labels)
    axes[0].set_title(f'Classification ({year})', fontsize=14, fontweight='bold')
    
    # Enhanced True color image with advanced processing
    try:
        # Check if RGB bands exist and have proper dimensions
        rgb_bands = ['red', 'green', 'blue']
        if all(band in plot_data.data_vars for band in rgb_bands):
            # Ensure each band is 2D and compute at full resolution
            rgb_arrays = []
            
            for band in rgb_bands:
                band_data = plot_data[band]
                if band_data.ndim != 2:
                    _LOG.warning("RGB band %s has %d dimensions, expected 2", band, band_data.ndim)
                    raise ValueError("RGB bands have incorrect dimensions")
                # Materialize at full resolution and apply enhancement
                band_array = _materialize_da_to_numpy(band_data)
                if band_array is not None:
                    # Apply additional enhancement for display
                    enhanced_band = _enhance_display_band(band_array, year)
                    rgb_arrays.append(enhanced_band)
                else:
                    raise ValueError(f"Could not load {band} band")
            
            if all(arr is not None for arr in rgb_arrays):
                red, green, blue = rgb_arrays
                
                # Stack and normalize with advanced stretching
                rgb_array = np.stack([red, green, blue], axis=-1)
                
                # Advanced percentile stretching with saturation
                p1, p99 = np.nanpercentile(rgb_array, [1, 99])
                rgb_normalized = (rgb_array - p1) / (p99 - p1 + 1e-8)
                rgb_normalized = np.clip(rgb_normalized, 0, 1)
                
                # Apply additional contrast enhancement
                rgb_enhanced = _enhance_rgb_contrast(rgb_normalized)
                
                # Use high-quality interpolation
                if year < 2017:  # Landsat - use sharper interpolation
                    interpolation_method = 'hanning'
                else:  # Sentinel - use smoother interpolation
                    interpolation_method = 'lanczos'
                
                axes[1].imshow(rgb_enhanced, interpolation=interpolation_method)
                axes[1].set_title('True Color (Enhanced)', fontsize=14, fontweight='bold')
            else:
                raise ValueError("RGB data preparation failed")
        else:
            raise ValueError("Missing RGB bands")
            
    except Exception as e:
        _LOG.warning("Enhanced RGB plot failed: %s", e)
        # Fallback to basic RGB
        try:
            rgb_bands = ['red', 'green', 'blue']
            if all(band in plot_data.data_vars for band in rgb_bands):
                rgb_arrays = []
                for band in rgb_bands:
                    rgb_arrays.append(_materialize_da_to_numpy(plot_data[band]))
                
                if all(arr is not None for arr in rgb_arrays):
                    red, green, blue = rgb_arrays
                    rgb_array = np.stack([red, green, blue], axis=-1)
                    p2, p98 = np.nanpercentile(rgb_array, [2, 98])
                    rgb_normalized = (rgb_array - p2) / (p98 - p2 + 1e-8)
                    rgb_normalized = np.clip(rgb_normalized, 0, 1)
                    
                    axes[1].imshow(rgb_normalized, interpolation='bilinear')
                    axes[1].set_title('True Color', fontsize=14, fontweight='bold')
                else:
                    raise ValueError("Basic RGB failed")
            else:
                raise ValueError("Missing RGB bands")
        except Exception as fallback_error:
            _LOG.warning("Basic RGB fallback also failed: %s", fallback_error)
            # Final fallback to grayscale
            try:
                if 'red' in plot_data.data_vars:
                    red_data = _materialize_da_to_numpy(plot_data['red'])
                    if red_data is not None:
                        axes[1].imshow(red_data, cmap='gray', interpolation='nearest')
                        axes[1].set_title('True Color (Grayscale)', fontsize=14, fontweight='bold')
                    else:
                        raise ValueError("Could not load red band")
                else:
                    raise ValueError("No red band available")
            except Exception as grayscale_error:
                _LOG.warning("Grayscale fallback also failed: %s", grayscale_error)
                axes[1].text(0.5, 0.5, 'RGB data\nnot available', 
                            ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
                axes[1].set_title('True Color', fontsize=14, fontweight='bold')
    
    # Probability map - high resolution with fixed typo
    try:
        # Ensure probabilities have proper dimensions
        probs_2d = plot_data.Probabilities
        if probs_2d.ndim > 3:
            probs_2d = probs_2d.squeeze()
        
        # Take maximum probability across classes for visualization
        if probs_2d.ndim == 3:
            max_prob = probs_2d.max(dim='class')
        else:
            max_prob = probs_2d
            
        # Use high-quality colormap - FIXED: removed the invalid interpolation parameter
        prob_plot = max_prob.plot(
            ax=axes[2], 
            cmap='viridis',
            vmin=0, 
            vmax=1,
            add_labels=False, 
            add_colorbar=False
        )
        cbar = fig.colorbar(prob_plot, ax=axes[2])
        cbar.set_label('Max Probability', fontsize=12)
        axes[2].set_title('Class Probabilities (Max)', fontsize=14, fontweight='bold')
    except Exception as e:
        _LOG.warning("Probability plot failed: %s", e)
        axes[2].text(0.5, 0.5, f'Probability Error: {str(e)}', 
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Probabilities (Error)', fontsize=14, fontweight='bold')
    
    # Remove axis ticks for cleaner look
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    return fig

def _enhance_display_band(band_array, year):
    """Apply display-specific enhancement to bands"""
    try:
        # Remove NaN and infinite values
        band_clean = np.nan_to_num(band_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Different enhancement for Landsat vs Sentinel
        if year < 2017:  # Landsat
            # More aggressive enhancement for Landsat
            p0, p100 = np.percentile(band_clean[band_clean > 0], [0.5, 99.5])
            band_clipped = np.clip(band_clean, p0, p100)
            band_normalized = (band_clipped - p0) / (p100 - p0 + 1e-8)
            
            # Apply histogram equalization for Landsat
            from skimage import exposure
            band_enhanced = exposure.equalize_hist(band_normalized)
            
        else:  # Sentinel
            # Gentle enhancement for Sentinel
            p1, p99 = np.percentile(band_clean[band_clean > 0], [1, 99])
            band_clipped = np.clip(band_clean, p1, p99)
            band_normalized = (band_clipped - p1) / (p99 - p1 + 1e-8)
            band_enhanced = band_normalized
            
        return np.clip(band_enhanced, 0, 1)
        
    except Exception as e:
        _LOG.warning("Display enhancement failed: %s", e)
        # Fallback to basic normalization
        band_clean = np.nan_to_num(band_array, nan=0.0, posinf=0.0, neginf=0.0)
        p1, p99 = np.percentile(band_clean[band_clean > 0], [1, 99])
        band_normalized = (band_clean - p1) / (p99 - p1 + 1e-8)
        return np.clip(band_normalized, 0, 1)

def _enhance_rgb_contrast(rgb_array):
    """Enhance RGB contrast using CLAHE or similar methods"""
    try:
        # Convert to HSV color space for better contrast adjustment
        import cv2
        hsv = cv2.cvtColor((rgb_array * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Apply CLAHE to Value channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        hsv[:,:,2] = clahe.apply(hsv[:,:,2])
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return enhanced_rgb.astype(float) / 255.0
        
    except Exception as e:
        _LOG.warning("Advanced RGB enhancement failed: %s", e)
        # Fallback to simple gamma correction
        return np.power(rgb_array, 0.9)  # Slight brightening

def compute_transition_matrix(pred1, pred2, class_labels):
    """Compute transition matrix between two predictions"""
    pred1_flat = pred1.Predictions.values.flatten()
    pred2_flat = pred2.Predictions.values.flatten()
    
    num_classes = len(class_labels)
    transition_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for i in range(len(pred1_flat)):
        class_from = int(pred1_flat[i])
        class_to = int(pred2_flat[i])
        transition_matrix[class_from, class_to] += 1
            
    return transition_matrix

def normalize_transition_matrix(matrix):
    """Normalize transition matrix to percentages"""
    row_sums = matrix.sum(axis=1, keepdims=True)
    percentage_matrix = np.divide(
        matrix, 
        row_sums, 
        out=np.zeros_like(matrix, dtype=float), 
        where=row_sums!=0
    )
    return percentage_matrix * 100

def plot_transition_matrix(matrix, class_labels, year_from, year_to):
    """Plot transition matrix as high-resolution heatmap"""
    fig, ax = plt.subplots(figsize=(14, 10), dpi=200)  # Higher DPI
    sns.heatmap(
        matrix, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues', 
        xticklabels=class_labels, 
        yticklabels=class_labels,
        ax=ax,
        annot_kws={"size": 10}  # Larger annotation text
    )
    ax.set_title(f'Land Cover Transitions: {year_from} → {year_to}', fontsize=16, fontweight='bold')
    ax.set_xlabel(f'Class in {year_to}', fontsize=14)
    ax.set_ylabel(f'Class in {year_from}', fontsize=14)
    return fig


def predict_for_years(lat, lon, years, status_callback=None):
    """
    Main prediction function for Streamlit with robust handling of lazy arrays.
    Materializes only the small prediction arrays needed for plotting and counts.
    """
    years = sorted(years)
    predictions = {}
    figures = []
    areas_per_class = {}
    transition_matrices = {}

    def _materialize_da_to_numpy(da):
        """
        Safely convert an xarray DataArray (possibly dask-backed) to a numpy array.
        If the DataArray has a dask array under .data, compute only that.
        """
        try:
            # prefer .data.compute() when available
            if hasattr(da, "data") and hasattr(da.data, "compute"):
                result = da.data.compute()
            # fallback: .values may be a dask array
            elif hasattr(da, "values") and hasattr(da.values, "compute"):
                result = da.values.compute()
            # last resort
            else:
                result = np.asarray(da)
            
            # Ensure we return a proper numpy array, not a 0-d array
            if hasattr(result, 'shape') and result.shape == ():
                return np.array([result])
            
            # Preserve original data type for maximum precision
            return result.astype(da.dtype if hasattr(da, 'dtype') else result.dtype)
        
        except Exception as e:
            _LOG.warning("Could not materialize DataArray to numpy: %s; falling back to np.asarray", e)
            try:
                result = np.asarray(da)
                if hasattr(result, 'shape') and result.shape == ():
                    return np.array([result])
                return result
            except Exception:
                # give up gracefully
                return None

    if status_callback:
        status_callback("Starting predictions...")

    for idx, year in enumerate(years):
        try:
            if status_callback:
                status_callback(f"Predicting for year {year} ({idx+1}/{len(years)})...")
            prediction = predict_for_location(lat, lon, year)
            predictions[year] = prediction

            # check prediction presence
            if prediction is None or not hasattr(prediction, "Predictions"):
                if status_callback:
                    status_callback(f"No prediction data for year {year}. Skipping.")
                continue

            # Materialize the Predictions array to numpy (safe and small)
            pred_np = _materialize_da_to_numpy(prediction.Predictions)
            if pred_np is None:
                if status_callback:
                    status_callback(f"Could not materialize predictions for year {year}. Skipping.")
                continue

            # Ensure pred_np is 2D
            if pred_np.ndim == 0:
                # Handle scalar case - create 1x1 array
                pred_np = np.array([[pred_np]])
            elif pred_np.ndim == 1:
                # Handle 1D case - reshape to 2D (n, 1)
                pred_np = pred_np.reshape(-1, 1)
            elif pred_np.ndim > 2:
                # try to squeeze singleton dimensions
                pred_np = np.squeeze(pred_np)
                
            if pred_np.ndim != 2:
                # it's unexpected; try to reshape if possible
                try:
                    pred_np = pred_np.reshape(prediction.Predictions.shape)
                except Exception:
                    _LOG.warning("Predictions for year %s are not 2D even after squeezing; skipping", year)
                    if status_callback:
                        status_callback(f"Predictions have unexpected shape for year {year}. Skipping.")
                    continue

            # Create a shallow copy of prediction for plotting
            prediction_for_plot = prediction.copy(deep=False)
            
            # Replace Predictions with numpy-backed DataArray
            try:
                # Get original dimensions
                orig_dims = tuple(prediction.Predictions.dims) if hasattr(prediction.Predictions, 'dims') else ('y', 'x')
                
                # Create coordinates - handle both cases where coords exist and don't exist
                coords = {}
                if 'y' in orig_dims:
                    try:
                        y_coord = prediction.Predictions.coords.get('y', None)
                        if y_coord is not None:
                            y_np = _materialize_da_to_numpy(y_coord)
                            if y_np is not None and len(y_np) == pred_np.shape[0]:
                                coords['y'] = ('y', y_np)
                    except Exception:
                        pass
                    if 'y' not in coords:
                        coords['y'] = ('y', np.arange(pred_np.shape[0]))
                        
                if 'x' in orig_dims:
                    try:
                        x_coord = prediction.Predictions.coords.get('x', None)
                        if x_coord is not None:
                            x_np = _materialize_da_to_numpy(x_coord)
                            if x_np is not None and len(x_np) == pred_np.shape[1]:
                                coords['x'] = ('x', x_np)
                    except Exception:
                        pass
                    if 'x' not in coords:
                        coords['x'] = ('x', np.arange(pred_np.shape[1]))

                # Create the DataArray
                da_pred_numpy = xr.DataArray(pred_np, dims=orig_dims, coords=coords)
                prediction_for_plot = prediction_for_plot.assign(Predictions=da_pred_numpy)

            except Exception as e:
                _LOG.warning("Failed to create numpy-backed DataArray: %s", e)
                # Fallback: use original prediction but ensure it's computed
                prediction_for_plot = prediction.compute() if hasattr(prediction, 'compute') else prediction

            # In the area calculation section, add progress updates:
            if status_callback:
                status_callback(f"Computing areas for year {year}...")

            # compute areas per class using numpy preds
            area_per_pixel = 900 if year < 2017 else 400
            classes = pred_np.ravel()

            # Filter out invalid predictions (values < 0)
            valid_mask = classes >= 0
            valid_classes = classes[valid_mask]

            if len(valid_classes) == 0:
                _LOG.warning("No valid predictions found for year %s", year)
                areas_per_class[year] = {label: 0 for label in class_labels}
            else:
                # Use bincount for faster computation than np.unique
                valid_classes_int = valid_classes.astype(int)
                class_counts = np.bincount(valid_classes_int, minlength=len(class_labels))
                
                areas_per_class[year] = {}
                for i, label in enumerate(class_labels):
                    if i < len(class_counts):
                        areas_per_class[year][label] = int(class_counts[i]) * area_per_pixel
                    else:
                        areas_per_class[year][label] = 0

            if status_callback:
                status_callback(f"Area calculation completed for {year}")

            if status_callback:
                status_callback(f"Plotting results for year {year}...")
            fig = plot_year_result(prediction_for_plot, year)
            figures.append(fig)

            if status_callback:
                status_callback(f"Completed year {year}.")

        except Exception as e:
            # more detail for debugging
            _LOG.exception("Error during processing year %s: %s", year, e)
            if status_callback:
                status_callback(f"Error processing year {year}: {str(e)}")
            # raise to keep behavior consistent with previous code
            raise RuntimeError(f"Year {year} processing failed: {str(e)}")

    if not areas_per_class:
        if status_callback:
            status_callback("No successful predictions for any year. Please try a different location or year.")
        raise RuntimeError("No successful predictions for any year. Please try a different location or year.")

    # Area summary plot - higher quality
    if status_callback:
        status_callback("Generating area summary plot...")
    df_areas = pd.DataFrame(areas_per_class).T.fillna(0)
    fig_area, ax = plt.subplots(figsize=(14, 8), dpi=150)  # Higher quality
    df_areas.plot(kind='bar', stacked=True, ax=ax,
                color=[class_colors[label] for label in class_labels])
    ax.set_title('Land Cover Area Over Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Area (m²)', fontsize=14)
    ax.legend(title='Classes', bbox_to_anchor=(1.05, 1), fontsize=12)
    plt.tight_layout()
    figures.append(fig_area)

    # Only generate difference plot if we have multiple years
    if len(years) > 1:
        if status_callback:
            status_callback("Generating area difference plot...")
        df_diff = df_areas.diff().dropna()
        if not df_diff.empty:
            fig_diff, ax = plt.subplots(figsize=(12, 8))
            df_diff.plot(kind='bar', ax=ax, color=[class_colors[label] for label in class_labels])
            ax.set_title('Yearly Area Changes')
            ax.set_xlabel('Year')
            ax.set_ylabel('Area Change (m²)')
            ax.legend(title='Classes', bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            figures.append(fig_diff)
        else:
            _LOG.info("Skipping area difference plot - no data after differencing")
    else:
        _LOG.info("Skipping area difference plot - only one year of data")
        # Create a placeholder message for single year
        fig_placeholder, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Area difference plot requires\nmultiple years of data', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Area Changes (Multiple Years Required)')
        figures.append(fig_placeholder)

    # Transition matrices (for consecutive years) - only if we have multiple years
    if len(years) > 1:
        for i in range(len(years) - 1):
            year_from = years[i]
            year_to = years[i + 1]
            if year_from not in predictions or year_to not in predictions:
                continue
            if status_callback:
                status_callback(f"Computing transitions: {year_from} → {year_to}...")

            # materialize both predictions safely
            pred1_np = _materialize_da_to_numpy(predictions[year_from].Predictions)
            pred2_np = _materialize_da_to_numpy(predictions[year_to].Predictions)
            if pred1_np is None or pred2_np is None:
                _LOG.warning("Skipping transition matrix for %s-%s due to inability to materialize predictions", year_from, year_to)
                continue

            # ensure 2D shapes
            pred1_np = np.squeeze(pred1_np)
            pred2_np = np.squeeze(pred2_np)
            
            # Check if arrays are not empty
            if pred1_np.size == 0 or pred2_np.size == 0:
                _LOG.warning("Skipping transition matrix for %s-%s due to empty predictions", year_from, year_to)
                continue
                
            try:
                matrix = compute_transition_matrix(
                    xr.Dataset({'Predictions': (('y', 'x'), pred1_np)}),
                    xr.Dataset({'Predictions': (('y', 'x'), pred2_np)}),
                    class_labels
                )
                norm_matrix = normalize_transition_matrix(matrix)
                fig_trans = plot_transition_matrix(norm_matrix, class_labels, year_from, year_to)
                transition_matrices[f"{year_from}-{year_to}"] = norm_matrix
                figures.append(fig_trans)
            except Exception as e:
                _LOG.warning("Failed to compute/plot transition for %s-%s: %s", year_from, year_to, e)
    else:
        _LOG.info("Skipping transition matrices - only one year of data")
        # Create a placeholder for transition matrix
        fig_placeholder, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Transition matrices require\nmultiple years of data', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Land Cover Transitions (Multiple Years Required)')
        figures.append(fig_placeholder)

    if status_callback:
        status_callback("All processing complete.")

    return predictions, figures, areas_per_class, transition_matrices


import datetime
from matplotlib.backends.backend_pdf import PdfPages

def generate_analysis_text(areas_per_class, transition_matrices, years, lat, lon):
    """Generate automated analysis of land cover results"""
    
    if not areas_per_class:
        return "No data available for analysis."
    
    analysis_lines = []
    analysis_lines.append("LAND COVER ANALYSIS SUMMARY")
    analysis_lines.append("=" * 40)
    analysis_lines.append("")
    analysis_lines.append(f"Location: Latitude {lat:.4f}, Longitude {lon:.4f}")
    analysis_lines.append(f"Analysis Period: {min(years)} to {max(years)}")
    analysis_lines.append("")
    
    # Dominant land cover
    latest_year = max(years)
    if latest_year in areas_per_class:
        latest_areas = areas_per_class[latest_year]
        dominant_class = max(latest_areas.items(), key=lambda x: x[1])
        total_area = sum(latest_areas.values())
        
        analysis_lines.append(f"CURRENT LAND COVER ({latest_year}):")
        analysis_lines.append(f"Dominant class: {dominant_class[0]} ({dominant_class[1]:,} m²)")
        analysis_lines.append("")
        
        # Area breakdown
        analysis_lines.append("AREA BREAKDOWN:")
        for class_name, area in sorted(latest_areas.items(), key=lambda x: x[1], reverse=True):
            if area > 0:
                percentage = (area / total_area) * 100
                analysis_lines.append(f"  {class_name}: {area:,} m² ({percentage:.1f}%)")
    
    # Changes over time
    if len(years) > 1:
        analysis_lines.append("")
        analysis_lines.append("CHANGES OVER TIME:")
        for i in range(len(years)-1):
            year1, year2 = years[i], years[i+1]
            if year1 in areas_per_class and year2 in areas_per_class:
                changes = {}
                for cls in areas_per_class[year1]:
                    change = areas_per_class[year2].get(cls, 0) - areas_per_class[year1].get(cls, 0)
                    changes[cls] = change
                
                if changes:
                    analysis_lines.append(f"  {year1} → {year2}:")
                    for cls, change in sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True):
                        trend = "increased" if change > 0 else "decreased"
                        analysis_lines.append(f"    {cls}: {trend} by {abs(change):,} m²")
    
    return "\n".join(analysis_lines)

def create_prediction_pdf(predictions, figures, areas_per_class, transition_matrices, lat, lon, years):
    """Create PDF report with all prediction details and plots"""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"landcover_report_{timestamp}.pdf"
    
    # Create PDF
    with PdfPages(pdf_filename) as pdf:
        # Page 1: Title and metadata
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        
        title_text = [
            "LAND COVER PREDICTION REPORT",
            "",
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Location: Latitude {lat:.4f}, Longitude {lon:.4f}",
            f"Years Analyzed: {', '.join(map(str, years))}",
            "",
            "Report Contents:",
            "• Automated land cover analysis",
            "• Classification maps for each year", 
            "• True color satellite imagery",
            "• Probability maps",
            "• Area change analysis",
            "• Transition matrices"
        ]
        
        plt.text(0.1, 0.9, "\n".join(title_text), transform=plt.gca().transAxes, 
                fontsize=14, verticalalignment='top', fontweight='bold')
        pdf.savefig()
        plt.close()
        
        # Page 2: Automated analysis
        analysis_text = generate_analysis_text(areas_per_class, transition_matrices, years, lat, lon)
        
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.1, 0.95, "AUTOMATED ANALYSIS", transform=plt.gca().transAxes, 
                fontsize=16, fontweight='bold')
        plt.text(0.1, 0.85, analysis_text, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top')
        pdf.savefig()
        plt.close()
        
        # Add all the figures
        for i, fig in enumerate(figures):
            fig.set_size_inches(11, 8.5)
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
        
        # Final page with summary
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        summary_text = [
            "REPORT SUMMARY",
            "",
            "This automated land cover analysis report includes:",
            "",
            "✓ High-resolution classification maps",
            "✓ Enhanced true color satellite imagery", 
            "✓ Classification confidence maps",
            "✓ Area change analysis over time",
            "✓ Land cover transition analysis",
            "✓ Automated change detection",
            "",
            "Data Sources:",
            "- Landsat 5-9 (1984-2012, 2013+)",
            "- Sentinel-2 (2017+)",
            "- Digital Earth Africa",
            "",
            f"Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}"
        ]
        
        plt.text(0.1, 0.9, "\n".join(summary_text), transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top', fontweight='bold')
        pdf.savefig()
        plt.close()
    
    return pdf_filename

# Add this function to your existing prediction workflow
def predict_and_generate_pdf(lat, lon, years, status_callback=None):
    """Complete prediction workflow with PDF generation"""
    
    # Run the existing prediction
    predictions, figures, areas_per_class, transition_matrices = predict_for_years(
        lat, lon, years, status_callback
    )
    
    # Generate PDF
    if status_callback:
        status_callback("Generating PDF report...")
    
    pdf_path = create_prediction_pdf(
        predictions, figures, areas_per_class, transition_matrices, lat, lon, years
    )
    
    if status_callback:
        status_callback(f"PDF report generated: {pdf_path}")
    
    return predictions, figures, areas_per_class, transition_matrices, pdf_path