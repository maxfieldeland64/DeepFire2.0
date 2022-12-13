"""
This is the main data processing pipeline script. It utilizes functions in the
API to download raw data, but has the primary job of:

- Loading perimeters for a specific fire

- From those perimeters, finding bounding boxes for lat/long of the fire, as
well as fire start / end dates

- Scraping LANDSAT-8 spectral band data for that bounding box & interval

- Scraping National Map lidar data for that bounding box & interval

- Correcting any varied geographical systems to ensure positions are the same
between layers

- Layering all data, then cropping to the maximum fire boundaries.

- Generating samples from the data at different time periods 

- Any data augmentation.

Some things to consider about this implementation:

- The temporal resolution for the perimeters is higher than it is for the other
types of data. However, data can only be used to predict the next day, so if
there are only three instances of data for the LANDSAT-8 spectral bands, then
the model could predict a maximum of three days (one for each piece of data).

-This would require ~ 6 perimeters (one for each day used as X, and for the day
directly after that X). 

- Areas inside the perimeter would likely be labelled as on fire, and we can
assume that fires do not "go out" in this model.
"""

from api import *
from preprocess import *

import os
import time
import random
import subprocess
from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from shapely.geometry import shape

import rasterio
from rasterio.plot import show
from rasterio.merge import merge

import matplotlib.pyplot as plt

# Redirect all printed output through tqdm's "write", to ensure some better 
# printing of loading bars in combination with text.
old_print = print
def new_print(*args, **kwargs):
    try:
        tqdm.tqdm.write(*args, **kwargs)
    except:
        old_print(*args, ** kwargs)
inspect.builtins.print = new_print

# -----------------------------------

def get_raw(
    fire_name, 
    fire_year, 
    fire_code,
    usgs_username, 
    usgs_password,
    parallel):
    """
    Gets all the necessary raw data for the fire:

    - Perimeters
    - tif files
    - Lidar

    Parameters:
        :str fire_name:     Name of the fire, where it will be saved.
        :int fire_year:     The year the fire took place.
        :str fire_code:     Unique fire code for the fire. Used to find perim.
        :str usgs_username: Username for the usgs site.
        :str usgs_password: Password for the usgs site.
        :bool parallel:     Whether or not downloads should be done in parallel.

    Returns:
        :list bounds:       The bounds of the lower left and upper right coords.
    """

    # Create a home for the data
    path = Path(Path.cwd().parents[0]) / 'data' / fire_name
    if (not path.exists()):
        path.mkdir(parents = True)

    # Find perimeters
    print("Finding fire perimeters")
    geomac = Geomac(fire_year)
    perims,rbounds = geomac.perimeters(fire_code) 

    # Find date interval from perimeters
    print("Finding date interval")
    dates = [p["properties"]["perimeterdatetime"] for p in perims]
    dates = [dates[0], dates[-1]]
    dates = [x.split('|')[0] for x in dates]

    ddate = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
    earlier_date = ddate - datetime.timedelta(days = 16)
    earlier_date = datetime.datetime.strftime(earlier_date, "%Y-%m-%d")

    modified_dates = [earlier_date, dates[1]]

    # Notice of download params:
    print("\nDownloading with fire name:", fire_name)
    print("For lower left bounding corner of", rbounds["ll"]["longitude"], 
        rbounds["ll"]["latitude"])
    print("and upper right bounding corner of", rbounds["ur"]["longitude"], 
        rbounds["ur"]["latitude"])
    print("between border dates", dates[0], "and", dates[1])
    print("between USGS dates", modified_dates[0], "and", modified_dates[1])

    # Download LANDSAT-8 files
    print("\nDownloading landsat data")
    print("This download can take a while")
    usgsscrape(
        username = usgs_username,
        password = usgs_password,
        firename = fire_name,
        start_date = modified_dates[0],
        end_date = modified_dates[1],
        ll = rbounds["ll"],
        ur = rbounds["ur"],
        parallel = parallel)

    # Realistically, lidar data won't change that significantly over the time
    # period so we do not impose a date restriction.

    # Download national map lidar data.
    print("\nDownloading National Map Lidar data\n")
    nm = NMscraper()
    nm.search_products(
        firename = fire_name,
        download = True,
        bbox = [rbounds["ll"]["longitude"], rbounds["ll"]["latitude"], 
        rbounds["ur"]["longitude"], rbounds["ur"]["latitude"]],
        parallel = parallel,
        outputFormat="'GeoTIFF'",
        maximum = None)

    return perims, rbounds

def reproject_usgs(fire_name, dataset = "LANDSAT_8_C1", selection = None):
    """
    For all the downloaded tif data, selections of which spectral bands to 
    keep and which to discard are made. Any remaining tifs are repositioned
    to match a WSG 84 system. It also assumes that all data is stored in:

    data / firename / raw / LANDSAT-8 / [tifs]

    example selection = ["B4", "B3", "B2"]

    Parameters:
        :str fire_name:     The name of the fire (will tell where it's stored)
        :list selection:    Which layers to KEEP, with each layer denoted by the
                            B# at the end of its filename.

    Returns:
        None, just edits files stored on disk.
    """

    print("Reprojecting usgs TIFS and deleting unused spectral bands")
    print("This can take a while.")
    path = Path(Path.cwd()).parents[0] / "data" / fire_name / "raw" / dataset
    
    # Remove any files that are not tifs
    allpaths = set(path.rglob('*'))
    tifs = set(path.rglob('*.TIF'))

    remove = allpaths.difference(tifs)
    remove = [x for x in remove if (not x.is_dir())]

    # Find tifs that match the selection.
    match_tifs = []
    for t in tifs:
        ending = str(t).split("_")[-1]
        tiftype = ending.split(".")[0]

        if (tiftype in selection):
            match_tifs.append(t)
        else:
            remove.append(t)

    # Remove any non-tif files or tifs that are not used.
    [os.remove(x) for x in remove]

    # Reproject tifs to the same location.
    for t in tqdm(match_tifs):
        reproject_tif(str(t), str(t))

def stitch_usgs(fire_name, dataset = "LANDSAT_8_C1"):
    """
    Creates a mosaic TIF file for each spectral band, for each timestep.
    Writes out the first truly preprocessed data.

    Parameters:
        :str fire_name:     The name of the fire (where raw data is located)

    Returns:
        None, but writes out files.
    """

    path = Path(Path.cwd().parents[0]) / "data" / fire_name 
    raw = path / "raw" / dataset
    prep = path / "preprocessed" / "mosaics"

    # First, get existing suffixes (remaining spectral bands)
    suffixes = set()
    paths = raw.rglob("*.TIF")
    for p in paths:
        ending = str(p).split('_')[-1]
        suf = ending.split('.')[0]
        suffixes.add(ending.split('.')[0])

    # Then, iteratively construct TIF mosaics for each timestep.
    times = list(raw.glob('*'))
    print("Creating mosaic TIFs for each date")
    for t in tqdm(times):
        
        # For each suffix, find all TIFs across scenes within that timestep 
        # matching
        for s in suffixes:
            tifpaths = t.rglob("*" + s + ".TIF")

            outpath = prep / dataset / (t.parts[-1])
            if (not outpath.exists()):
                outpath.mkdir(parents = True)

            outpath = outpath / (s + ".TIF")

            try:
                stitch(tifpaths, outpath, show = False)
            except:
                print("Skipping date", t.parts[-1], "due to missing contents")

def reproject_nm(fire_name):
    """
    Does almost the same as the function above, but there're no unused files
    downloaded with the lidar data, so there's less to clean out.

    Parameters:
        :str fire_name:     The name of the fire we're cleaning tifs for.
    
    Returns:
        None, but modifies TIF files.
    """

    path = Path(Path.cwd().parents[0]) / "data" / fire_name / "raw" / "NMAP"
    tifpaths = list(path.rglob("*.tif"))

    print("Reprojecting National Map TIFS")
    for i in tqdm(tifpaths):
        reproject_tif(i, i)

def stitch_nm(fire_name):
    """
    Like the other stitching function, this just stiches together TIF
    files for lidar.

    Parameters:
        :str fire_name:         The name of the fire.
    """

    path = Path(Path.cwd().parents[0]) / "data" / fire_name 
    raw = path / "raw" / "NMAP"
    prep = path / "preprocessed" / "mosaics"

    if (not prep.exists()):
        prep.mkdir(parents = True)

    outpath = prep / "LIDAR.TIF"
    tifpaths = list(raw.rglob("*.tif"))

    print("Creating mosaic TIFs")
    stitch(tifpaths, outpath)

def clip_mosaics(fire_name, bounds):
    print("Cropping mosaics")
    path = Path(Path.cwd().parents[0]) / "data" / fire_name / "preprocessed"
    mosaics = path / "mosaics"

    mosaics = list(mosaics.rglob("*.TIF"))

    for m in mosaics:

        # Create parent directory if needed overwrite other paths if needed.
        clippath = Path(str(m).replace("mosaics", "clipped"))
        if (not clippath.parents[0].exists()):
            clippath.parents[0].mkdir(parents=True)

        if (clippath.exists()):
            os.remove(clippath)

        crop(m, clippath, bounds)

def rasterize_perims(fire_name, perims):
    """ 
    Rasterizes (e.g. turns into npy files) the perimeters according to the 
    smallest bounds found for any of the cropped TIF or lidar files. 
    The cropping is always a little different for each grouping of raster files.

    Raster bounds may all be equivalent, but raster pixel spaces seem to be 
    plus or minus 100 indices. For example, a B1 spectral band from two 
    different instances, cropped to the same
    perimeter may have:

    B1a.shape = (1229,1059)
    B2b.shape = (1210,1044)

    B1a.bounds = BoundingBox(left=-120.70182311012779, bottom=38.759905889948584 
                 right=-120.37646708342223, top=39.13749092943974) 

    B1b.bounds = BoundingBox(left=-120.70196997910246, bottom=38.76004817352045, 
                 right=-120.37646050034499, top=39.13731490656312)

    Therefore, when choosen a shape to project the perimeters into, 
    we choose the smallest possible pixel dimensions in the cropped dataset.

    Parameters:
        :str fire_name:     The name of the fire / where data will be located.
        :list perims:       The perimemters dictionary returned from a geomac 
                            search.

    Returns:
        :list abounds:      Artificial shape boundaries for TIFS/boundaries in 
                            pixel space.
    """

    # First, get pixel bounds.
    path = Path(Path.cwd().parents[0]) / "data" / fire_name / "preprocessed"  
    clipped = path / "clipped"
    cropped_tifs = list(clipped.rglob("*.TIF"))
    rasts = [rasterio.open(x) for x in cropped_tifs]

    abounds = list(rasts[0].shape)
    sbounds = list(rasts[0].shape)
    saff = rasts[0].transform

    for r in rasts:
        if (r.shape[0] <= sbounds[0] and r.shape[1] <= sbounds[1]):
            sbounds[0] = r.shape[0]
            sbounds[1] = r.shape[1]
            saff = r.transform

        if (r.shape[0] <= abounds[0]):
            abounds[0] = r.shape[0]

        if (r.shape[1] <= abounds[1]):
            abounds[1] = r.shape[1]

    print("Projecting perimeters to shape", abounds, "in pixel space")

    for p in perims:
        perim = path / "perims"
        
        if (not perim.exists()):
            perim.mkdir(parents=True)

        perimpath = perim / p["properties"]["perimeterdatetime"]

        shp = shape(p["geometry"])
        npy = rasterio.features.rasterize([shp], out_shape=abounds, 
            transform=saff)

        np.save(perimpath, npy)

    return abounds

def stack_layers(fire_name, dataset, scheme, order, key = None):
    """
    This method creates a tensor from the preprocessed data for each time 
    depending on the selected time interval method, according to a specified 
    order.

    Since data can be somewhat intermittent or fires can fall between rotation 
    periods for satellites, multiple schemes are available. Typically, satellite 
    data has the lowest temporal resolutution, followed by the perimeter 
    reports, then finally by weather. We only use one LIDAR report since it is 
    unlikely to change significantly over time.

    Potential schemes:
        "wrt_sat":  Where stacks are only created for times for which there is 
                    satellite imagery. E.g. if there were 5 satellite snapshots, 
                    there would be no more than 5 stacks total, even if there 
                    were more reported weather snapshots or perimeters.

        "wrt_per":  Where stacks are created for any available perimeter 
                    snapshot using the last reported satellite imagery. E.g. if 
                    there were 5 satellite snapshots, but 20 available perimeter 
                    reports, 20 stacks would be constructed using the last
                    available satellite imagery.

    Weather is always reported with the relevant perimeter date/time.

    Potential ordering items:
        - Any spectral band ('B1', 'B2' ... etc.)
        - The LIDAR layer (referred to just as 'LIDAR')
        - Land cover layer (referred to as 'LANDCOV')
        - The weather data (see available attributes list below)
        - The perims (referred to as 'PERIMS')

    Example to have a tensor of b1, atmospheric data (windspeed, humidity), 
    and perims:
        order = ['B1', 'precipitation', 'windspeed', 'PERIMS']
    
    Parameters:
        :st fire_name:  The name of the fire (where it will be located)
        :str dataset:  The name of the satellite imagery dataset
        :str scheme:    The scheme for generating stacks.
        :list order:    A list of strings denoting how the stack should be 
                        layered.
        :str key:       Optional meteostat weather api key. Only needed if 
                        weather layers are included in stacking.


    This code is not particularly pretty, but it is functional and easy enough
    to repair or understand.
    """

    valid_weather = ["temperature", "dewpoint", "humidity", "precipitation",
                    "precipitation_3", "precipitation_6", "snowdepth",
                    "windspeed", "peakgust", "winddirection", "pressure"]

    desired_weather = [x for x in order if x.lower() in valid_weather]

    path = Path(Path.cwd().parents[0]) / "data" / fire_name / "preprocessed"
    sat = path / "clipped" / dataset
    lid = path / "clipped" / "LIDAR.TIF"
    landcov = path / "clipped" / "LANDCOV.TIF"
    perim = path / "perims"

    # With respect to each satellite image
    if (scheme == "wrt_sat"):
        print("Creating stacks with respect to satellite images")

        for s in tqdm(list(sat.glob('*'))):
            date = s.parts[-1]  
            date = datetime.datetime.strptime(date, "%Y-%m-%d")

            perims_paths = list(perim.glob("*.npy"))
            perim_time_diffs = []
           
            # Find difference in time between border reports and sats 
            for p in perims_paths:
                p_date = str(p.parts[-1]).split('|')[0]
                p_date = datetime.datetime.strptime(p_date, "%Y-%m-%d")
                perim_time_diffs.append((p, abs(date - p_date)))

            # Find minimum difference in time between sat image and border report
            perim_time_diffs.sort(key = lambda x: x[1])
            perimeter = np.load(perim_time_diffs[0][0])

            stack = []
            shp = perimeter.shape

            # Get weather data, but first find time and lat/long pos.
            if (len(desired_weather) > 0):
                basesat = rasterio.open(list(sat.rglob('*.TIF'))[0])
                lon = np.average([basesat.bounds[0], basesat.bounds[2]])
                lat = np.average([basesat.bounds[1], basesat.bounds[3]])

                time = datetime.datetime.strftime(p_date, "%Y-%m-%d")
                weather = weather_average(lat, lon, time, desired_weather, key)
            else:
                weather = {}

            for i in order:
                if (i in ["B" + str(x) for x in np.arange(14)]):
                    op = str(Path(str(sat) / s / i)) + ".TIF"
                    tif = rasterio.open(op)
                    tif = tif.read()[0][0:shp[0], 0:shp[1]]
                    stack.append(normalize_array(tif))

                elif (i == "LIDAR"):
                    tif = rasterio.open(lid)
                    # The lidar has a rather irritating bug border values
                    # are padded with something approaching -inf.
                    # Since these borders are not where the fire is growing
                    # we just correct them to 0.

                    arr = tif.read()[0]
                    broken = np.where(arr < -1e5)
                    arr[broken] = 0

                    tif = arr[0:shp[0], 0:shp[1]]
                    stack.append(normalize_array(tif))

                elif (i == "LANDCOV"):
                    tif = rasterio.open(landcov)
                    tif = tif.read()[0]
                    stack.append(tif)

                elif (i == "PERIM"):
                    stack.append(perimeter)

                elif (i.lower() in desired_weather):
                    layer = np.full((shp[0], shp[1]), weather[i])
                    stack.append(layer)
                else:
                    print("Layer", i, "not found")

            stack = np.array(stack)
            outpath = path / "stacks" 
            if (not outpath.exists()):
                outpath.mkdir(parents=True)
            outpath = outpath / (s.parts[-1] + ".npy")
            np.save(outpath, stack)

    # With respect to previous satellite imagery
    elif (scheme == "wrt_per"):
        print("Creating stacks with respect to each perimeter.")

        perim_paths = sorted(list(perim.glob("*.npy")))
       
        for p in tqdm(perim_paths):

            p_date = p.parts[-1].split('.')[0] 
            p_date = datetime.datetime.strptime(p_date, "%Y-%m-%d|%H:%M:%S")

            sat_paths = list(sat.glob('*'))
            sat_times = []
            for s in sat_paths:
                t = s.parts[-1]
                t = datetime.datetime.strptime(t, "%Y-%m-%d")

                if (t <= p_date):
                    sat_times.append((s, abs(p_date - t)))

            # Sort through valid dates to find closest.
            sat_times.sort(key = lambda x: x[1])
            try:
                # Sometimes, no satellite images are older than the reported 
                # border
                satellite = sat_times[0][0]
            except:
                print("No satellite images older than", p_date, 
                    "omitting from stacks.")
                continue

            perimeter = np.load(p)
            shp = perimeter.shape

            # Get weather data, but first find time and lat/long pos.
            if (len(desired_weather) > 0):
                basesat = rasterio.open(list(sat.rglob('*.TIF'))[0])
                lon = np.average([basesat.bounds[0], basesat.bounds[2]])
                lat = np.average([basesat.bounds[1], basesat.bounds[3]])

                time = datetime.datetime.strftime(p_date, "%Y-%m-%d")
                weather = weather_average(lat, lon, time, desired_weather, key)
            else:
                weather = {}

            stack = []
            for i in order:
                if (i in ["B" + str(x) for x in np.arange(14)]):
                    op = satellite / (i + ".TIF")
                    tif = rasterio.open(op)
                    tif = tif.read()[0][0:shp[0], 0:shp[1]]
                    stack.append(normalize_array(tif))

                elif (i == "LIDAR"):
                    # The lidar has a rather irritating bug border values
                    # are padded with something approaching -inf.
                    # Since these borders are not where the fire is growing
                    # we just correct them to 0.

                    tif = rasterio.open(lid)
                    arr = tif.read()[0]
                    broken = np.where(arr < -1e5)
                    arr[broken] = 0

                    tif = arr[0:shp[0], 0:shp[1]]
                    stack.append(normalize_array(tif))

                elif (i == "LANDCOV"):
                    tif = rasterio.open(landcov)
                    tif = tif.read()[0]
                    stack.append(tif)

                elif (i == "PERIM"):
                    stack.append(perimeter)

                elif (i.lower() in desired_weather):
                    layer = np.full((shp[0], shp[1]), weather[i])
                    stack.append(layer)
                else:
                    print("Layer", i, "not found")

            stack = np.array(stack)
            outpath = path / "stacks" 
            if (not outpath.exists()):
                outpath.mkdir(parents=True)
            outpath = outpath / (p.parts[-1])
            np.save(outpath, stack)

def select_stacks(fire_name, min_diff = 24):
    """
    The fires will  occasionally generate more stacks than are really worth
    dealing with. This function thins out / restructures the stacks before
    samples are created.

    If datetimes are embedded in the stacks, it might make sense the include
    all of them. Otherwise, it might make sense to restrict them to the closest
    stack outside of a 24 hour window (e.g. all instances are at least a day
    apart).

    Parameters:
        :str fire_name:     The name of the fire.
        :int min_diff:      Minimum difference of hours between fire reports.

    Returns:
        None, just moves the unused fires into a different directory.
    """

    print("Filtering stacks to one per", min_diff, "hours.")

    path = (Path(Path.cwd().parents[0]) / "data" / fire_name / "preprocessed" /
        "stacks")

    unused = path / "unused"
    if (not unused.exists()):
        unused.mkdir(parents=True)

    stacks = sorted(list(path.glob("*.npy")))
    stack_times = [datetime.datetime.strptime(
        x.parts[-1].split('.')[0], "%Y-%m-%d|%H:%M:%S") 
    for x in stacks]

    stack_pairs = list(zip(stacks, stack_times))

    kept = []

    current_stack = 0
    next_stack = 1
    delta = datetime.timedelta(hours = min_diff)
    
    while (next_stack < len(stacks)):

        if (stack_pairs[next_stack][1]-stack_pairs[current_stack][1] >= delta):
            kept.append(stack_pairs[current_stack][0])
            current_stack = next_stack
            next_stack += 1

        else:
            next_stack += 1

    unkept = [x for x in stacks if x not in kept]
    
    print("Moving", len(unkept), "stacks to unused directory.")

    for i in unkept:
        newpath = path / "unused" / i.parts[-1]
        os.replace(i, newpath)

def generate_samples(
    fire_name, 
    height, 
    width, 
    count, 
    mask_layer,
    binary = True,
    out_height = 5,
    out_width = 5,
    parallel = False,
    cores = -1, 
    seed = None):
    """
    Generates [count] samples from each stack in the fire around a center 
    pixel from (pixel_x - height, pixel_x + height) and (pixel_y - width,
    pixel_y + width). Thus, the size of the overall sample will always be
    an odd number.

    Ex: If we sampled on a pixel width height=15, width=15, we would create
    a sample of size 31x31 centered around that pixel, with a binary outcome
    dependent on whether that position at the next time step is or is not
    on fire.

    This method generates all the samples and compresses them into a zip,
    so we don't have a bunch of high-dimension sample files just floating
    around.

    Parameters:
        :str fire_name:     The fire name.
        :int height:        Height +- of stack.
        :int width:         Int +- of stack.
        :int count:         Number of samples per stack.
        :int mask_layer:    Which layer of the stack is the fire mask.
        :bool binary:       Whether or not samples should be sampled as binary
                            pixel predictions (True), or whether they should
                            output a mask array (False). 

                            If a mask aray is desired, out height and width
                            must be specified.

        :int out_height:    Height out output mask
        :int out_width:     Width of output mask.

        :int seed:          The random seed.

    Returns:
        None, but can write out large zipfiles.

        All samples should be written out as:

        samples/firename/stackname/sample-pixel-pixel-height-width-outcome
        or 
        outcomes/firename/stackname/outcome-pixel-pixel-height-width
    """

    random.seed(seed)

    path = Path(Path.cwd().parents[0]) / "data" / fire_name
    stacks = path / "preprocessed" / "stacks"
    stacks = sorted(list(stacks.glob("*.npy")))

    if (binary):
        outpath = path / "final" / "binary"
    else:
        outpath = path / "final" / "masks"

    if (not outpath.exists()):
        outpath.mkdir(parents = True)

    if (parallel == False):
        # If sequential, just make one big file.
        print("Generating samples sequentially")

        zpath = outpath / "all-samples.zip"

        z = zipfile.ZipFile(str(zpath), "w")

        for i in tqdm(range(len(stacks) - 1)):
            stack = np.load(stacks[i])
            next_stack = np.load(stacks[i + 1])

            indices = set(np.ndindex(stack.shape[1::]))

            padstack = np.pad(stack, ((0,0), (height, height), (width,width)))

            if (not binary):
                next_padstack = np.pad(stack,((0,0),(out_height, out_height),
                    (out_width, out_width)))

            for c in tqdm(range(count), leave=False):
                pixel = random.sample(indices, 1)[0]
                indices.remove(pixel)

                # Must adjust the pixel position when sampling the padded stack.
                newpixel = list(pixel)
                newpixel[0] = pixel[0] + height
                newpixel[1] = pixel[1] + width

                sample = padstack[:, 
                                (newpixel[0]-height):(newpixel[0]+height+1),
                                (newpixel[1]-width):(newpixel[1]+width+1)]

                stack_name = stacks[i].parts[-1].split('.')[0]

                if (binary):
                    stack_outcome = next_stack[mask_layer, pixel[0], pixel[1]]
                else:
                    stack_outcome = next_padstack[mask_layer,
                    (newpixel[0]-out_height):(newpixel[0]+out_height+1),
                    (newpixel[1]-out_width):(newpixel[1]+out_width+1)]

                if (binary):
                    out_name = "sample-{}-{}-{}-{}-{}.npy".format(
                        str(pixel[0]),
                        str(pixel[1]),
                        str(height),
                        str(width),
                        str(stack_outcome))

                    # Saves useless read/write step.
                    buf = BytesIO()
                    np.save(buf, sample.T)
                    z.writestr(("samples" + 
                                '/' + 
                                fire_name +
                                '/' +
                                stack_name + 
                                '/' + 
                                out_name), buf.getvalue())

                else:
                    xname = "sample-{}-{}-{}-{}.npy".format(
                        str(pixel[0]),
                        str(pixel[1]),
                        str(height),
                        str(width))

                    buf = BytesIO()
                    np.save(buf, sample.T)
                    z.writestr(("samples" +
                                '/' + 
                                fire_name + 
                                '/' + 
                                stack_name +
                                '/' +
                                xname), buf.getvalue())

                    yname = "outcome-{}-{}-{}-{}.npy".format(
                        str(pixel[0]),
                        str(pixel[1]),
                        str(out_height),
                        str(out_width))

                    buf = BytesIO()
                    np.save(buf, stack_outcome)

                    z.writestr(("outcomes" +
                                '/' + 
                                fire_name + 
                                '/' + 
                                stack_name +
                                '/' +
                                yname), buf.getvalue())


        z.close()

    elif (parallel == True):
        # If parallel, make a zipfile for each sample's stacks, then combine
        # at the end. 
        print("Generating samples in parallel; this can take a while")
        print("Progress bar represents general progress of first n cores worth")

        # Iterating over each stack index
        def helper(i,
            stacks = stacks,
            height = height,
            width = width,
            count = count,
            binary = binary,
            out_height = out_height,
            out_width = out_width,
            outpath = outpath):

            stack = np.load(stacks[i])
            next_stack = np.load(stacks[i + 1])

            indices = set(np.ndindex(stack.shape[1::]))

            padstack = np.pad(stack, ((0,0), (height, height), (width,width)))

            if (not binary):
                next_padstack = np.pad(stack,((0,0),(out_height, out_height),
                    (out_width, out_width)))

            stack_name = stacks[i].parts[-1].split('.')[0]
            zpath = outpath / (stack_name + ".zip")
            z = zipfile.ZipFile(zpath, "w")

            for c in tqdm(range(count), leave=False):
                pixel = random.sample(indices, 1)[0]
                indices.remove(pixel)

                # Must adjust the pixel position when sampling the padded stack.
                newpixel = list(pixel)
                newpixel[0] = pixel[0] + height
                newpixel[1] = pixel[1] + width

                outpixel = list(pixel)
                outpixel[0] = pixel[0] + out_height
                outpixel[1] = pixel[1] + out_width

                sample = padstack[:, 
                                (newpixel[0]-height):(newpixel[0]+height+1),
                                (newpixel[1]-width):(newpixel[1]+width+1)]

                if (binary):
                    stack_outcome = next_stack[mask_layer, pixel[0], pixel[1]]
                else:
                    stack_outcome = next_padstack[mask_layer,
                    (outpixel[0]-out_height):(outpixel[0]+out_height+1),
                    (outpixel[1]-out_width):(outpixel[1]+out_width)+1]

                if (binary):
                    out_name = "sample-{}-{}-{}-{}-{}.npy".format(
                        str(pixel[0]),
                        str(pixel[1]),
                        str(height),
                        str(width),
                        str(stack_outcome))

                    # Saves useless read/write step.
                    buf = BytesIO()
                    np.save(buf, sample.T)
                    z.writestr(("samples" + 
                                '/' + 
                                fire_name +
                                '/' +
                                stack_name + 
                                '/' + 
                                out_name), buf.getvalue())

                else:
                    xname = "sample-{}-{}-{}-{}.npy".format(
                        str(pixel[0]),
                        str(pixel[1]),
                        str(height),
                        str(width))

                    buf = BytesIO()
                    np.save(buf, sample.T)
                    z.writestr(("samples" +
                                '/' + 
                                fire_name + 
                                '/' + 
                                stack_name +
                                '/' +
                                xname), buf.getvalue())

                    yname = "outcome-{}-{}-{}-{}.npy".format(
                        str(pixel[0]),
                        str(pixel[1]),
                        str(out_height),
                        str(out_width))

                    buf = BytesIO()
                    np.save(buf, stack_outcome)

                    z.writestr(("outcomes" +
                                '/' + 
                                fire_name + 
                                '/' + 
                                stack_name +
                                '/' +
                                yname), buf.getvalue())

            z.close()
            return zpath

        zpaths = Parallel(n_jobs = cores)(delayed(helper)(
            i = i) for i in range(len(stacks) - 1))

        # Concatenate all the zipfiles into a single file, remove individual
        # pieces.
        z = zipfile.ZipFile(str(outpath / "all-samples.zip"), "w")
        print("Concatenating parlallelized zipfiles")
        for p in tqdm(zpaths, leave=False):
            rec = zipfile.ZipFile(p, "r")

            rec_items = list(rec.namelist())
            [z.writestr(x, rec.open(x).read()) for x in rec_items]
            rec.close()
            os.remove(p)

        z.close()

def generate_splits(
    fire_name, 
    train, 
    test, 
    validate, 
    binary=True, 
    seed=None, 
    parallel=True):
    """
    Takes the all-samples.zip file and generates train / test / validate
    files to be read over by generators.

    Parameters:
        :str fire_name:         The name of the fire
        :float train:           Percentage train
        :float test:            Percentage test
        :float vallidate:       Percentage validate
        :bool binary:           Whether the data is binary pixel data or masks.
        :int seed:              A seed for the split.
    Returns:
        None, but creates train / test / splits.
    """

    path = Path(Path.cwd().parents[0]) / "data" / fire_name / "final"

    if (binary):
        path = path / "binary"
    else:
        path = path / "masks"

    z = zipfile.ZipFile(str(path / "all-samples.zip"), "r")

    np.random.seed(seed)
    zpaths = z.namelist()
    if (not binary):
        zpaths = [x for x in zpaths if "sample" in x]

    np.random.shuffle(zpaths)

    train_idx = int(len(zpaths) * train)
    validate_idx = int(len(zpaths) * (train + validate))

    train_paths = zpaths[0:train_idx]
    validate_paths = zpaths[0:validate_idx]
    test_paths = zpaths[validate_idx::]

    z.close()

    combs = zip([train_paths, validate_paths, test_paths], 
                ["train-samples.zip", "validate-samples.zip", 
                "test-samples.zip"])

    # Find output dimension; hacky solution for filenames.
    outdim = ['0','0']
    for i in z.namelist()[0:10]:
        if ("outcome" in i):
            parts = i.split('-')
            outdim[0] = parts[-2]
            outdim[1] = parts[-1].split('.')[0]

            break
        else:
            pass

    def helper(i, path = path, outdim = outdim, binary=binary):
        z = zipfile.ZipFile(str(path / "all-samples.zip"), "r")

        paths = i[0]
        zfile = zipfile.ZipFile(str(path / i[1]), "w")

        for p in paths:
            zfile.writestr(p, z.open(p).read())

            if (not binary):
                other = p.replace("sample", "outcome")
                parts = other.split('-')
                parts[-2] = outdim[0]
                parts[-1] = outdim[1] + '.npy'
                otherpath = '-'.join(parts)

                zfile.writestr(otherpath, z.open(otherpath).read())

        zfile.close()

    if (parallel == False):
        print("Generating train test split sequentially")
        for i in combs:
            helper(i)

    elif (parallel == True):
        print("Generating train test split in parallel")
        Parallel(n_jobs = -1)(delayed(helper)(
            i = i,
            path = path) for i in combs)

# --- VISUALIZATION --- #

def show_mosaics(fire_name, save = False):
    """
    Shows the preprocessed mosaics created for a gsaiven fire.

    Parameters:
        :str fire_name:         The name of the fire.

    Returns:
        None, but shows several plots.
    """

    plt.ioff()

    path = (Path(Path.cwd().parents[0]) / "data" / fire_name / "preprocessed"
        / "mosaics")

    mosaics = list(path.rglob("*.TIF"))

    print("Showing", len(mosaics), "mosaics.")
    for i in list(mosaics):
        title = i.parts[-1]
        rast = rasterio.open(str(i))

        fig = plt.figure()
        ax = show(rast, title = title)

        if (save):
            outpath = path / (title.split(".")[0] + ".png")
            plt.savefig(outpath)
