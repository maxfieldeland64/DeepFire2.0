"""
Contains functions / methods for processing the data to get it in certain 
formats.
"""
import numpy as np

import rasterio
import rasterio.mask
from rasterio.plot import show
from rasterio.merge import merge
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling

from pathlib import Path
from shapely.geometry import box

def reproject_tif(file_path, new_file_path):
    """
    Input:
        file_path : path to file to reproject, must be .tif file
        new_file_path : path to file that will be written to store new 
        projection
    
    
    Output:
        new_projection : returns new projection raster

    Reproject a raster from original projection to EPSG 4326 projection. 
    Use optimal new heights and widths, apply padding as necessary 
    """
    
    dst_crs = 4326
    with rasterio.open(file_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
    
        with rasterio.open(new_file_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

def stitch(tifpaths, outpath, show = False):
    """
    Stitches together tif files to create a mosaic, then does the same thing
    with the NMAP lidar files.

    Writes everything out as a final tif file and LIDAR file.

    Should organize by date. 

    Parameters:
        :list tifpaths:     A list of paths for the tif files you want to merge.
        :str outpath:       The location for the writeout file.
        :bool show:         Whether or not the final mosaic should be shown.
    
    Returns:
        None, but begins writing out the first preprocessed data.
    """

    # Create mosaic
    rasters = [rasterio.open(str(p)) for p in tifpaths]
    mosaic, out_trans = merge(rasters)

    # Get meta
    meta = rasters[0].meta.copy()

    # Update metadata
    meta["height"] = mosaic.shape[1]
    meta["width"]  = mosaic.shape[2]
    meta["transform"] = out_trans

    with rasterio.open(outpath, 'w', **meta) as out:
        out.write(mosaic)

def normalize_array(arr):
    """
    Takes an array of values and returns the normalized version of that array.

    Parameters:
        :array arr:     The array we're generating a normalized copy of.

    Returns:
        :array norm_array:      The normalized array.
    """

    av = np.average(arr)
    std = np.std(arr)
    normer = lambda x : (x - av) / std
    vnormer = np.vectorize(normer)
    return vnormer(arr)

def crop(tifpath, outpath, bounds):
    """
    Crops a given TIF file to a set of coordinate bounds.

    Parameters:
        :str tifpath:   The filepath of the TIF that is loaded.
        :str outpath:   Where the TIF is written out to.
        :dict bounds:   The bounds of the new window. Should be a nested 
                        dictionary of lat/long positions of the lower left 
                        and upper right corners.
    
    Returns:
        None.
    """

    bounds_box = box(
                    bounds["ll"]["longitude"], 
                    bounds["ll"]["latitude"], 
                    bounds["ur"]["longitude"],
                    bounds["ur"]["latitude"]
                    )

    src = rasterio.open(tifpath, "r")
    meta = src.meta.copy()

    cropped_image, cropped_transform = rasterio.mask.mask(src, 
                                        [bounds_box], crop=True)
    
    meta.update({"driver": "GTiff",
                 "height": cropped_image.shape[1],
                 "width": cropped_image.shape[2],
                 "transform": cropped_transform})

    out = rasterio.open(outpath, "w", **meta)
    out.write(cropped_image)

