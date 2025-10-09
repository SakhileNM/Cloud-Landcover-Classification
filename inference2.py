'''
import warnings
warnings.filterwarnings("ignore")
import os
import logging, traceback
logging.basicConfig(level=logging.INFO)

# --- Dask & Deafrica cluster setup ---
from joblib import parallel_backend

# --- Core imports ---
import numpy as np
import pandas as pd
import xarray as xr
import datacube
from datacube import Datacube
from datacube.testutils.io import rio_slurp_xarray
from joblib import load
import matplotlib.pyplot as plt

from deafrica_tools.datahandling import load_ard
from deafrica_tools.bandindices import calculate_indices
from deafrica_tools.classification import predict_xr
from deafrica_tools.plotting import rgb

ODC_CONF = os.environ["DATACUBE_CONFIG_PATH"]
dc = Datacube(app="inference", config=ODC_CONF)

# --- Load your models (exact file names) ---
S_MODEL_PATH = 'S_model(1).joblib'
L_MODEL_PATH = 'L_model(1).joblib'

landsat_model = load(L_MODEL_PATH).set_params(n_jobs=1)
sentinel_model = load(S_MODEL_PATH).set_params(n_jobs=1)

# --- Constants from your notebook ---
BUFFER = 0.1
DASK_CHUNKS = {'x': 1000, 'y': 1000}
MEASUREMENTS = ['blue','green','red','nir','swir_1','swir_2']
OUTPUT_CRS = 'epsg:6933'
PRODUCTS = ['gm_s2_annual', 'gm_ls5_ls7_annual', 'gm_ls8_ls9_annual']
CLASS_LABELS = ['Water','Forestry','Urban','Barren','Crop','Other']
CLASS_COLORS = ['blue','green','red','yellow','purple','grey']

def init_dask_cluster():
    """
    On Windows you must guard cluster creation so it only runs
    when the script is launched, not when modules are imported.
    """
    import multiprocessing
    multiprocessing.freeze_support()

    from deafrica_tools.dask import create_local_dask_cluster
    from distributed import Client

    create_local_dask_cluster()
    client = Client()
    client.cluster.scale(n=4)
    return client

# --- Feature fetching exactly as you did, but unpacking args ---
def feature_layers(query, model):
    """
    Mirrors your notebook's dc.load(...) + index + slope steps,
    but avoids the unknown-arg errors by explicit parameters.
    """
    # use the pre‑configured client
    global dc

    ds = load_ard(
        dc=dc,
        products=PRODUCTS,
        measurements=MEASUREMENTS,
        x=query['x'],
        y=query['y'],
        time=query['time'],
        resolution=query['resolution'],
        output_crs=query['output_crs'],
        dask_chunks=DASK_CHUNKS
    )

    # compute indices
    mission = 's2' if model is sentinel_model else 'ls'
    da = calculate_indices(
        ds,
        index=['NDVI','LAI','MNDWI'],
        drop=False,
        satellite_mission=mission
    )

    # slope via rio_slurp_xarray
    slope = rio_slurp_xarray(
        "https://deafrica-input-datasets.s3.af-south-1.amazonaws.com/"
        "srtm_dem/srtm_africa_slope.tif",
        gbox=ds.geobox
    ).to_dataset(name='slope').chunk(DASK_CHUNKS)

    return xr.merge([da, slope], compat='override').squeeze()

# --- Single‐location, single‐year prediction (same logic) ---
def predict_for_location(location_id, coords, year):
    """
    Exactly your notebook code: choose Landsat vs Sentinel,
    build query, fetch features, then predict_xr under Dask backend.
    Returns an xarray.Dataset with Predictions & Probabilities.
    """
    lat, lon = coords
    year_int = int(year)

    if year_int < 2017:
        model = landsat_model
        resolution = (-30, 30)
    else:
        model = sentinel_model
        resolution = (-10, 10)

    query = {
        'x': (lon - BUFFER, lon + BUFFER),
        'y': (lat + BUFFER, lat - BUFFER),
        'time': year,
        'measurements': MEASUREMENTS,
        'resolution': resolution,
        'output_crs': OUTPUT_CRS
    }

    # fetch & prepare data
    data = feature_layers(query, model)
    # restrict to exactly these bands/indices+slope
    data = data[MEASUREMENTS + ['NDVI','LAI','MNDWI','slope']]

    # run prediction_xr exactly as in your notebook
    with parallel_backend('dask', wait_for_workers_timeout=60):
        result = predict_xr(
            model,
            data,
            proba=True,
            persist=True,
            clean=True,
            return_input=True
        ).compute()

    return result

# --- Batch runner that produces your maps, summary & table ---
def predict_for_years(lat, lon, years):
    """
    Loops your original per‐year logic, builds:
      - One 3‐panel figure per year (Classification / RGB / Probability)
      - A stacked‐bar summary fig
      - A pandas DataFrame of areas per class
    """
    area_tables = {}
    figs = []

    for year in years:
        ds = predict_for_location("user_loc", (lat, lon), year)

        # 3‐panel map
        fig, axes = plt.subplots(1,3, figsize=(18,6))
        cmap = plt.cm.get_cmap('tab20', len(CLASS_LABELS))
        norm = plt.Normalize(0, len(CLASS_LABELS)-1)

        # Classified
        axes[0].imshow(ds.Predictions, cmap=cmap, norm=norm)
        axes[0].set_title(f'Classified {year}')
        axes[0].axis('off')

        # True colour
        rgb(ds, bands=['red','green','blue'], ax=axes[1], percentile_stretch=(0.01,0.99))
        axes[1].set_title('True Colour')
        axes[1].axis('off')

        # Probability
        pm = ds.Probabilities.plot(
            ax=axes[2], cmap='magma', vmin=0, vmax=100, add_colorbar=True
        )
        axes[2].set_title('Probability (%)')
        axes[2].axis('off')

        figs.append(fig)

        # area calc
        pixel_area = 900 if int(year)<2017 else 100
        vals, counts = np.unique(ds.Predictions.values, return_counts=True)
        areas = {CLASS_LABELS[int(v)]: int(c)*pixel_area for v,c in zip(vals,counts)}
        area_tables[year] = {lbl: areas.get(lbl,0) for lbl in CLASS_LABELS}

    # DataFrame
    df_areas = pd.DataFrame.from_dict(area_tables, orient='index')[CLASS_LABELS]

    # summary fig
    fig2, ax2 = plt.subplots(figsize=(10,6))
    df_areas.plot(
        kind='bar', stacked=True,
        color=CLASS_COLORS, ax=ax2
    )
    ax2.set_title('Area Covered by Each Class Over Time')
    figs.append(fig2)

    return df_areas, figs
'''

