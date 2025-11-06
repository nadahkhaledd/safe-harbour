import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer

# --- Open raster layers ---
with rasterio.open('slope_64.tif') as src_s:
    slope = src_s.read(1)
    transform = src_s.transform

with rasterio.open('roughness_64.tif') as src_r:
    roughness = src_r.read(1)

with rasterio.open('mars_64pix.tif') as src_e:
    elevation = src_e.read(1)

# ---Define Mars CRS (Equirectangular) ---
mars_eqc = CRS.from_wkt("""PROJCS["SIMPLE_CYLINDRICAL MARS",
    GEOGCS["GCS_MARS",
        DATUM["D_MARS",SPHEROID["MARS",3396000,0]],
        PRIMEM["Reference_Meridian",0],
        UNIT["degree",0.0174532925199433]],
    PROJECTION["Equirectangular"],
    PARAMETER["standard_parallel_1",0],
    PARAMETER["central_meridian",180],
    UNIT["metre",1]]""")

mars_geog = CRS.from_proj4("+proj=longlat +a=3396000 +b=3396000 +no_defs")

# --- Get pixel coordinates ---
rows, cols = np.indices(slope.shape)
xs, ys = rasterio.transform.xy(transform, rows, cols)
xs = np.array(xs)
ys = np.array(ys)

# --- Convert to geographic coordinates ---
transformer = Transformer.from_crs(mars_eqc, mars_geog, always_xy=True)
long_deg, lat_deg = transformer.transform(xs, ys)

# --- Convert longitude range (0–360 → -180–180) ---
long_deg = ((long_deg + 180) % 360) - 180

# ---Define suitability conditions (NASA/ESA-like) ---
slope_ok = slope <= 20       # low slope
roughness_ok = roughness < 2    # smooth surface
elevation_ok = elevation <= 1000    # below mean elevation

# ---Plot maps (basic raster view, not Mollweide) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

maps = [
    (slope_ok, 'Reds', 'Slope < 10°'),
    (roughness_ok, 'Blues', 'Surface roughness < 2 m'),
    (elevation_ok, 'Greens', 'Elevation < 0 m')
]

for ax, (data, cmap, title) in zip(axes, maps):
    im = ax.imshow(data, cmap=cmap, extent=[long_deg.min(), long_deg.max(), lat_deg.min(), lat_deg.max()])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

plt.suptitle("Mars — Suitable areas for rover landing", fontsize=14)
plt.tight_layout()
plt.show()
