import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np

MARS_RADIUS = 3396000  # m
MARS_CIRCUMFERENCE = 2 * np.pi * MARS_RADIUS  # m
DEGREE_LENGTH_MARS = MARS_CIRCUMFERENCE / 360  # m/degree


def load_data(file_path):
    header_file_path = "../../data/gcm/gcm_headers.txt"

    with open(header_file_path, 'r') as f:
        header_line = f.readline().strip()
        new_column_names = header_line.split(',')
    df = pd.read_csv(file_path, sep=';', names=new_column_names, skiprows=[0])

    return df

def plot_results(gdf, lat_res, lon_res, pixel_size_km):
    print("Plotting results... please wait...")
    lat_res = lat_res
    lon_res = lon_res
    fig, ax = plt.subplots(figsize=(12, 6))
    gdf.plot(ax=ax, column='temperature', cmap='coolwarm', markersize=5, legend=True)
    ax.set_title(f'Temperature map on Mars\nGrid resolution: {lat_res}° × {lon_res}°  (~{pixel_size_km:.2f} km/pixel)')
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    plt.show()

def get_analysis(df):
    df['windspeed'] = np.sqrt(df['zonal wind'] ** 2 + df['meridional wind'] ** 2)
    unique_lat = np.sort(df['latitude'].unique())
    unique_lon = np.sort(df['longitude'].unique())
    lat_res = np.min(np.diff(unique_lat))
    lon_res = np.min(np.diff(unique_lon))
    print(f"Resolution: {lat_res}° × {lon_res}°")
    pixel_size_km = DEGREE_LENGTH_MARS * lon_res / 1000  # km
    print(f"Pixel size at equator: {pixel_size_km:.2f} km")
    return lat_res, lon_res, pixel_size_km

def main():
    df = load_data("../../data/gcm/datasets/out_grid1x1deg_0h_0sollon.csv")
    df = df.apply(pd.to_numeric, errors='coerce')

    lat_res, lon_res, pixel_size_km = get_analysis(df)

    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="+proj=longlat +a=3396000 +b=3396000 +no_defs")

    plot_results(gdf, lat_res, lon_res, pixel_size_km)


if __name__ == "__main__":
    main()
