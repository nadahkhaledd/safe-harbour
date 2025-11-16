import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer

with rasterio.open('slope_64.tif') as src_s:
    slope = src_s.read(1)
    transform = src_s.transform

with rasterio.open('roughness_64.tif') as src_r:
    roughness = src_r.read(1)

with rasterio.open('mars_64pix.tif') as src_e:
    elevation = src_e.read(1)


slope_ok = slope < 10
roughness_ok = roughness < 2
elevation_ok = elevation < 0

mask_all = slope_ok & roughness_ok & elevation_ok


rows, cols = np.where(mask_all)
xs, ys = rasterio.transform.xy(transform, rows, cols)
xs = np.array(xs)
ys = np.array(ys)


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

transformer = Transformer.from_crs(mars_eqc, mars_geog, always_xy=True)
lon, lat = transformer.transform(xs, ys)


plt.figure(figsize=(10,5))
plt.scatter(lon, lat, s=2, c='red', label='Suitable sites')
plt.title("Suitable Landing Areas on Mars")
plt.xlabel("Longitude [°]")
plt.ylabel("Latitude [°]")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()
