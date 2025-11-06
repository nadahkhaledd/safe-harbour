import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np

#FOR MCD DATA
MARS_RADIUS = 3396000  # m
MARS_CIRCUMFERENCE = 2 * np.pi * MARS_RADIUS  # m
DEGREE_LENGTH_MARS = MARS_CIRCUMFERENCE / 360  # długość 1 stopnia w metrach (~59.3 km)


new_column_names = [
    'lattitude', 'longitude', 'atm pressure', 'atm density', 'temperture', 'zonal wind',
    'meridional wind', 'extvar_0', 'extvar_1', 'extvar_2', 'extvar_3', 'extvar_4', 'extvar_5',
    'extvar_6', 'extvar_7', 'extvar_8', 'extvar_9', 'extvar_10', 'extvar_11', 'extvar_12',
    'extvar_13', 'extvar_14', 'extvar_15', 'extvar_16', 'extvar_17', 'extvar_18', 'extvar_19',
    'extvar_20', 'extvar_21', 'extvar_22', 'extvar_23', 'extvar_24', 'extvar_25', 'extvar_26',
    'extvar_27', 'extvar_28', 'extvar_29', 'extvar_30', 'extvar_31', 'extvar_32', 'extvar_33',
    'extvar_34', 'extvar_35', 'extvar_36', 'extvar_37', 'extvar_38', 'extvar_39', 'extvar_40',
    'extvar_41', 'extvar_42', 'extvar_43', 'extvar_44', 'extvar_45', 'extvar_46', 'extvar_47',
    'extvar_48', 'extvar_49', 'extvar_50', 'extvar_51', 'extvar_52', 'extvar_53', 'extvar_54',
    'extvar_55', 'extvar_56', 'extvar_57', 'extvar_58', 'extvar_59', 'extvar_60', 'extvar_61',
    'extvar_62', 'extvar_63', 'extvar_64', 'extvar_65', 'extvar_66', 'extvar_67', 'extvar_68',
    'extvar_69', 'extvar_70', 'extvar_71', 'extvar_72', 'extvar_73', 'extvar_74', 'extvar_75',
    'extvar_76', 'extvar_77', 'extvar_78'
]

df = pd.read_csv(r"C:\Users\Hyperbook\Downloads\out_grid1x1deg_0h_0sollon.csv", sep=';', names=new_column_names, skiprows=[0])

# Konwersja kolumn do wartości numerycznych
df = df.apply(pd.to_numeric, errors='coerce')

# --- Obliczenia pomocnicze ---
# Wiatr
df['windspeed'] = np.sqrt(df['zonal wind']**2 + df['meridional wind']**2)

# Ustalenie rozdzielczości (co ile stopni są punkty)
unique_lat = np.sort(df['lattitude'].unique())
unique_lon = np.sort(df['longitude'].unique())

lat_res = np.min(np.diff(unique_lat))
lon_res = np.min(np.diff(unique_lon))
print(f"Rozdzielczość: {lat_res}° × {lon_res}°")

# Długość piksela (w km) – obliczamy na równiku
pixel_size_km = DEGREE_LENGTH_MARS * lon_res / 1000  # km
print(f"Rozmiar piksela na równiku: {pixel_size_km:.2f} km")

# --- Tworzenie warstwy punktowej ---
geometry = [Point(xy) for xy in zip(df['longitude'], df['lattitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="+proj=longlat +a=3396000 +b=3396000 +no_defs")

# --- Wizualizacja ---
fig, ax = plt.subplots(figsize=(12, 6))
gdf.plot(ax=ax, column='temperture', cmap='coolwarm', markersize=5, legend=True)
ax.set_title(f'Temperature map on Mars\nGrid resolution: {lat_res}° × {lon_res}°  (~{pixel_size_km:.2f} km/pixel)')
ax.set_xlabel("Longitude [°]")
ax.set_ylabel("Latitude [°]")
plt.show()
