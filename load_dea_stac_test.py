# load_dea_stac_test.py
import odc.stac
import pystac_client
import xarray as xr
import logging

logging.basicConfig(level=logging.INFO)

# configure rio so rioxarray/rasterio can read DEA S3 without signing
odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})

# DEA STAC endpoint
CATALOG = "https://explorer.digitalearth.africa/stac/"

# South Africa approx bbox: [minx, miny, maxx, maxy]
SA_BBOX = [16.45, -34.84, 32.89, -22.13]

# change product, year and bands as needed
product = "gm_s2_annual"            # try one product first
year = "2019"                       # e.g. 2019
bands = ["red", "green", "blue", "nir", "swir_1", "swir_2"]

# connect to catalog
catalog = pystac_client.Client.open(CATALOG)

# search (collection == product) over bbox and year
datetime = f"{year}-01-01/{year}-12-31"
search = catalog.search(collections=[product], bbox=SA_BBOX, datetime=datetime)
items = list(search.get_items())

print(f"Found {len(items)} STAC items for {product} in {year} inside SA bbox")

if not items:
    raise SystemExit("No items found — try a different product/year or expand bbox")

# load them to an xarray.Dataset (odc-stac will handle reading assets as COGs)
# choose resolution / crs matching your pipeline (e.g. resolution=10 for S2 geomedian, 30 for Landsat)
ds = odc.stac.load(items, bands=bands, crs="epsg:6933", resolution=10)  # adjust resolution

print("Loaded dataset:")
print(ds)
print("Data variables:", list(ds.data_vars))
