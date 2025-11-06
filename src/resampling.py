import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from pyproj import CRS

def resample_to_1deg(src_file, dst_file):
    dst_crs = CRS.from_proj4("+proj=longlat +a=3396000 +b=3396000 +no_defs")
    dst_res = 1.0
    dst_width = int(360 / dst_res)
    dst_height = int(180 / dst_res)
    dst_transform = from_origin(-180, 90, dst_res, dst_res)

    with rasterio.open(src_file) as src:
        profile = src.profile.copy()
        profile.update(
            driver='GTiff',
            dtype='float32',
            count=1,
            crs=dst_crs.to_wkt(),
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
        )
        with rasterio.open(dst_file, 'w', **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.average
            )
    print(" Saved:", dst_file)

resample_to_1deg('slope_64.tif', 'slope_64_resampled.tif')
resample_to_1deg('roughness_64.tif', 'roughness_64_resampled.tif')
resample_to_1deg('mars_64pix.tif', 'elevation_64_resampled.tif')
