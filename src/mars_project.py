import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pyproj import CRS, Transformer

# --- Open raster layers ---
with rasterio.open('slope_64.tif') as src_s:
    slope = src_s.read(1)
    transform = src_s.transform

with rasterio.open('roughness_64.tif') as src_r:
    roughness = src_r.read(1)

with rasterio.open('mars_64pix.tif') as src_e:
    elevation = src_e.read(1)

# --- Mask NoData values ---
slope = np.where(slope == src_s.nodata, np.nan, slope)
roughness = np.where(roughness == src_r.nodata, np.nan, roughness)
elevation = np.where(elevation == src_e.nodata, np.nan, elevation)

# --- Compute slope in degrees ---
slope_deg = np.degrees(np.arctan(slope))

# --- Define suitability conditions ---
slope_ok = slope_deg <= 20
roughness_ok = roughness < 1
elevation_ok = elevation <= 1000
valid_mask = ~np.isnan(slope_deg) & ~np.isnan(roughness) & ~np.isnan(elevation)
mask_all = slope_ok & roughness_ok & elevation_ok & valid_mask

# --- Get coordinates only for suitable pixels ---
rows_ok, cols_ok = np.where(mask_all)
xs_ok, ys_ok = rasterio.transform.xy(transform, rows_ok, cols_ok)
xs_ok = np.array(xs_ok)
ys_ok = np.array(ys_ok)

# --- Convert to geographic coordinates ---
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

lon, lat = transformer.transform(xs_ok, ys_ok)
lon = ((lon + 180) % 360) - 180  # convert 0–360 → -180–180

# --- Create ranking ---
rank = np.ones(lon.shape, dtype=int)  # domyślnie range 1
range2 = (lat < -50) | (lat > 50)
rank[range2] = 2

# --- Flatten data for seaborn (już spłaszczone) ---
df = pd.DataFrame({
    'lon': lon,
    'lat': lat,
    'rank': rank
})

# --- Konwersja do radianów ---
lon_rad = np.radians(df['lon'].values)
lat_rad = np.radians(df['lat'].values)
rank = df['rank'].values

# --- Kolory dla rankingów ---
colors = np.array(['green' if r==1 else 'orange' for r in rank])

# --- Plot w Mollweide ---
patch1 = patches.Patch(color='green', label='Best')
patch2 = patches.Patch(color='orange', label='Suboptimal')

plt.figure(figsize=(12,6))
ax = plt.subplot(111, projection='mollweide')
ax.scatter(lon_rad, lat_rad, c=colors, s=2)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_title("Landing Site Ranking on Mars (Mollweide projection)")
ax.legend(handles=[patch1, patch2], loc='lower left', fontsize=10)
plt.show()

# --- Przygotowanie danych do CSV ---
df_csv = pd.DataFrame({
    'x': cols_ok,         # kolumny rasteru
    'y': rows_ok,         # wiersze rasteru
    'lon': lon,           # geograficzne longitude
    'lat': lat,           # geograficzne latitude
    'rank': rank          # 1 lub 2
})

# --- Zapis do CSV ---
df_csv.to_csv("mars_landing_sites_topo.csv", index=False)
print("Dane zapisane do mars_landing_sites_topo.csv")

# --- Plot with Seaborn ---
plt.figure(figsize=(12,6))
sns.scatterplot(data=df, x='lon', y='lat', hue='rank', palette={Best:'green',Suboptimal:'orange'}, s=5, legend='full')
plt.title("Landing Site Ranking on Mars")
plt.xlabel("Longitude [°]")
plt.ylabel("Latitude [°]")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(title='Range')
plt.show()


# # Mask NoData values for plotting
# slope_plot = np.where(np.isnan(slope_deg), np.nan, slope_ok)
# roughness_plot = np.where(np.isnan(roughness), np.nan, roughness_ok)
# elevation_plot = np.where(np.isnan(elevation), np.nan, elevation_ok)
#
# # --- Plot slope ---
# plt.figure()
# plt.imshow(slope_plot, cmap='Reds',
#            extent=[long_deg.min(), long_deg.max(), lat_deg.min(), lat_deg.max()],
#            origin='upper')
# plt.colorbar(label='Slope suitability')
# plt.title('Slope < 20°')
# plt.xlabel('Longitude (°E)')
# plt.ylabel('Latitude (°N)')
# plt.show()
#
# # --- Plot roughness ---
# plt.figure()
# plt.imshow(roughness_plot, cmap='Blues',
#            extent=[long_deg.min(), long_deg.max(), lat_deg.min(), lat_deg.max()],
#            origin='upper')
# plt.colorbar(label='Roughness suitability')
# plt.title('Surface roughness < 2 m')
# plt.xlabel('Longitude (°E)')
# plt.ylabel('Latitude (°N)')
# plt.show()
#
# # --- Plot elevation ---
# plt.figure()
# plt.imshow(elevation_plot, cmap='Greens',
#            extent=[long_deg.min(), long_deg.max(), lat_deg.min(), lat_deg.max()],
#            origin='upper')
# plt.colorbar(label='Elevation suitability')
# plt.title('Elevation < 1000 m')
# plt.xlabel('Longitude (°E)')
# plt.ylabel('Latitude (°N)')
# plt.show()
