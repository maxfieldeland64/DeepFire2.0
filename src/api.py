import warnings
warnings.filterwarnings("ignore")

import os
import copy
import json
import pytz
import urllib
import inspect
import tarfile
import zipfile
import datetime
import numpy as np
from usgs import api
from tqdm import tqdm
from pathlib import Path 
from joblib import Parallel, delayed
from timezonefinder import TimezoneFinder

import matplotlib.pyplot as plt
import matplotlib.animation as ani

from shapely.geometry import shape

# Redirect all printed output through tqdm's "write", to ensure some better 
# printing of loading bars in combination with text.
old_print = print
def new_print(*args, **kwargs):
    try:
        tqdm.tqdm.write(*args, **kwargs)
    except:
        old_print(*args, ** kwargs)
inspect.builtins.print = new_print


# Gnarly helper functions I ripped off the tqdm docs. Don't worry about them.
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path, position):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url[0:10]+"...",
                             position = position,
                             leave = False) as t:
        urllib.request.urlretrieve(url, filename=output_path, 
            reporthook=t.update_to)

# -----------------------------------

def usgsscrape(
    username,
    password,
    firename,
    start_date = None,
    end_date = None,
    lat = None,
    lng = None,
    ll = None,
    ur = None,
    where = None,
    outpath = None,
    max_results = 10,
    node = "EE",
    dataset = "LANDSAT_8_C1",
    parallel = True,
    include = "LC",
    **kwargs):

    """
    Search through relevant database for relevant files.
    Saves and extracts in raw data directory. 

    Params are essentially what the usgs api tool is using, plus or minus
    a few items.

    Parameters:
        :str username:          USGS username.
        :str password:          USGS password.
        :str firename:          Name of the fire we're investigating.
        :int lat:               Latitude.
        :int lng:               Longitude
        :dict ur:               Upper right bounding box of {"latitude":..}
        :dict ll:               Lower left bounding box of {"latitude":..}
        :str start_date:        Start date of image record (YYYY-MM-DD)
        :str end_date:          End date of image record (YYYY-MM-DD)
        :dict where:            Any remaining kwargs to specify. See docs.
        :str outpath:           Output directory for the files.
        :int max_results:       Maximum alled number of return results.
        :str node:              The catalog to use (EE, CVIC, etc.)
        :str dataset:           The dataset in the catalog to use.
        :bool parallel:         Whether data should be downloaded in parallel.
        :str include:           Sometimes, extra satellite data is downloaded.
                                The include tag restricts downloads to only
                                results that include a certain substring in 
                                their ID.

    Returns:
        None; writes out to file.
    """

    print("Creating session with USGS (this can take a while)")
    api.login(username, password, catalogId=node)

    if (outpath == None):
        path = Path().cwd()
    else:
        path = Path(outpath)

    scenes = api.search(
        dataset = dataset, 
        node = node, 
        lat = lat, 
        lng = lng,
        ur = ur,
        ll = ll,
        start_date = start_date,
        end_date = end_date,
        max_results = max_results,
        where = where)

    scenes = scenes["data"]["results"]

    if (include != None):
        scenes = [x for x in scenes if include in x["entityId"]]

    scenes = list(zip(scenes, 
        np.arange(0, len(scenes))))

    print("Evaluating", len(scenes), "scenes")

    # Download and extract in parallel.
    def helper(i, path, firename, dataset, node, parallel):
        outdir = (path.parents[0] / "data" / firename / "raw" /
            str(dataset) / i[0]["acquisitionDate"]/ str(i[0]["entityId"]))

        if (not outdir.exists()):
            outdir.mkdir(parents=True)

        dl = api.download(
            dataset = dataset,
            node = node,
            entityids = str(i[0]["entityId"]))

        outpath = str(outdir / "compressed.tar.gz")

        # Only show the first download..
        if (parallel):
            if (i[1] == 0):
                download_url(
                    url = dl["data"][0]["url"], 
                    output_path = outpath,
                    position = i[1])
            else:
                urllib.request.urlretrieve(dl["data"][0]["url"], 
                    filename=outpath)
        else:
            download_url(
                    url = dl["data"][0]["url"], 
                    output_path = outpath,
                    position = i[1])

        tar = tarfile.open(outpath, 'r:gz')
        tar.extractall(path = outdir)
        tar.close()
        os.remove(outpath)

    if (parallel):
        try:
            print("Downloading in parallel")
            print("Showing download progress for first scene:")
            Parallel(n_jobs = -1)(delayed(helper)(
                i = i, 
                path = path,
                firename = firename,
                dataset = dataset,
                node = node,
                parallel = parallel)
                for i in scenes)
        except:
            print("One or more downloads errored. It is strongly suggested you")
            print("do this download with a good connection. If you are on a")
            print("good connection, you may be hitting the USGS rate limit.")
            print("If the problem persists, try downloading sequentially")
            print("instead of in parallel.")
    else:
        print("Downloading sequentially")
        for i in tqdm(scenes):
            i = list(i)
            i[1] = 1

            try:
                helper(i, 
                    path = path, 
                    firename = firename, 
                    dataset = dataset,
                    node = node,
                    parallel = parallel)
            except:
                # Sometimes the connection times out for long downloads.
                # In this case, just try to open a new session.
                api.logout()
                api.login(username, password)
                helper(i, 
                    path = path, 
                    firename = firename, 
                    dataset = dataset,
                    node = node,
                    parallel = parallel)

def weather_average(lat, lon, day, attr, key):
    """
    Returns a simple json summary of some weather characteristics for a given
    day, at the station closest to a given location. Uses 
    https://api.meteostat.net/ to collect information.

    Parametesr:
        :int lat:       The latitude.
        :int lon:       The longitude.
        :str day:       The day to view.
        :list attr:     A list of strings / attributes to keep.
        :str key:       The api key.

    Returns:
        :dict weather:      The weather dictionary returned.
    """

    # Get all weather stations, sort by distance, select the closest.
    requrl = ("https://api.meteostat.net/v1/stations/nearby?"
        "lat={}&lon={}&limit=2&key={}").format(lat,lon,key)

    req = urllib.request.urlopen(requrl)
    res = json.loads(req.read())

    stations = res["data"]
    stations.sort(key = lambda x: x["distance"])

    print("Available stations:")
    [print(x) for x in stations]

    def get_sum(idx, stations, lat, lon, day, key):
        """
        Helper function to recursively call weather to the next
        station when an originally called station does not return
        the desired data.

        Parameters:
            Same as above, with an additional:

            :int idx:       Which weather station to use.
            :list stations: The stations being iterated over.
        """

        if (idx < len(stations)):
            print("Searching station", stations[idx]["id"], "on date", day)

            requrl = ("https://api.meteostat.net/v1/history/hourly?"
            "station={}&start={}&end={}&key={}").format(
            stations[idx]["id"],day,day,key)

            req = urllib.request.urlopen(requrl)
            res = json.loads(req.read())

            if (len(res["data"]) > 0):
                summary = {}
                for i in res["data"][0].keys():
                    if (i in attr):
                        hours = [x[i] for x in res["data"] if (x[i] != None)]
                        hours = [float(x) for x in hours]

                        # If no value is reported for these attributes, 
                        # which frequently
                        # report None, report them as 0. 

                        if (len(hours) == 0 and i in ["precipitation", 
                            "precipitation_3", "precipitation_6", "snowdepth",
                            "peakgust", "pressure", "dewpoint"]):
                            hours = [0]

                        summary[i] = np.average(hours)

                return summary

            else:
                return get_sum(
                    idx = idx + 1, 
                    stations = stations,
                    lat = lat,
                    lon = lon,
                    day = day,
                    key = key)

        else:
            return "nan"

    return get_sum(
                    idx = 0, 
                    stations = stations,
                    lat = lat,
                    lon = lon,
                    day = day,
                    key = key)

class NMscraper():
    """
    The USGS provides plenty of useful data and there are numerous libraries to
    find it and download it. I have not found something comparable for the 
    national map, which maintains several of its own datsets, so I'm making
    one of my own. Relevant info can be found here:

    https://viewer.nationalmap.gov/help/documents/TNMAccessAPIDocumentation/
    TNMAccessAPIDocumentation.pdf

    The metadata is scraped through the national map website, where as the
    actual downloaded files are hosted on an amazon AWS instance.
    """

    def __init__(self):
        self.baseurl = "http://viewer.nationalmap.gov/tnmaccess/api/products?"
        self.datasets = "http://viewer.nationalmap.gov/tnmaccess/api/datasets?"

    def list_datasets(
        self,
        bbox = '',
        q = ''):
        """
        Requests datasets metadata from the national map.
        """

        datasets = urllib.request.urlopen(
            self.datasets + "bbox=&q=")
        datasets = datasets.read()

        return json.loads(datasets)

    def search_products(
        self,
        firename,
        download = False,
        outpath = None,
        bbox = [-95, 37, -94, 38],
        datasets = "National Elevation Dataset (NED) 1 arc-second",
        prodFormats = '',
        prodExtents = '',
        q = '',
        dateType = '',
        start = "2014-01-01",
        end = "2015-01-01",
        offset = '',
        maximum = 10,
        outputFormat = '',
        polyType = '',
        polyCode = '',
        dtype = "tif",
        parallel = True):

        """
        Main search method for this class. See:

        https://viewer.nationalmap.gov/help/documents/TNMAccessAPIDocumentation/
        TNMAccessAPIDocumentation.pdf

        for more information on all search fields. Note, for example, that you
        need to use 'sbDatasetTag' instead of 'title' for searches.

        Note that automatic parellelized downloads might not work for all
        datsets because they have different downloadURL
        """

        query = (self.baseurl + 
            (
            "bbox={0}&"
            "datasets={1}&"
            "prodFormats={2}&"
            "prodExtents={3}&"
            "q={4}&"
            "dateType={5}&"
            "start={6}&"
            "end={7}&"
            "offset={8}&"
            "max={9}&"
            "outputFormat={10}&"
            "polyType={11}&"
            "polyCode={12}&"
            ).format(
                ','.join(list([str(i) for i in bbox])),
                datasets,
                prodFormats,
                prodExtents,
                q,
                dateType,
                start,
                end,
                str(offset),
                str(maximum),
                outputFormat,
                polyType,
                str(polyCode)))

        # Clean query 
        query = query.replace(' ', '+')

        print("\nSearching with query:", "\n" + query, "\n")

        data = urllib.request.urlopen(query)
        data = json.loads(data.read())

        if (dtype != None):
            print("Sorting out some results based on data type")
            data["items"] = [x for x in data["items"] 
            if dtype in str(x["downloadURL"])]

        print("Returning", len(data['items']), "of", 
            str(data['total']), "found")

        # If download set to true, download and sort items.
        if (download):
            if (outpath == None):
                path = Path().cwd()
            else:
                path = Path(outpath)

            def helper(i, firename, path, parallel):
                outdir = (path.parents[0] / "data" / firename / "raw" /
                    "NMAP" / str(i[0]["sourceId"]))

                if (not outdir.exists()):
                    outdir.mkdir(parents=True)

                outpath = outdir / str("data." + dtype)

                # Only show the first download..
                if (parallel):
                    if (i[1] == 0):
                        download_url(
                            url = i[0]["downloadURL"], 
                            output_path = outpath,
                            position = i[1])
                    else:
                        urllib.request.urlretrieve(i[0]["downloadURL"], 
                            filename=outpath)
                else:
                    download_url(
                            url = i[0]["downloadURL"], 
                            output_path = outpath,
                            position = i[1])

            scenes = list(zip(data['items'], np.arange(0, len(data['items']))))

            if (parallel):
                try:
                    print("Downloading in parallel")
                    print("Showing download progress for first scene:")
                    Parallel(n_jobs = -1)(delayed(helper)(
                        i = i, 
                        path = path,
                        firename = firename,
                        parallel = parallel)
                        for i in scenes)
                except:
                    print("One or more downloads errored. It is strongly")
                    print("you perform this download on a good connection.")
                    print("If you are on a good connection and the issue")
                    print("persist, then you may be hitting the rate limit.")
                    print("In this case, try downloading sequentially.")
            else:
                print("Downloading sequentially")
                for i in tqdm(scenes):
                    i = list(i)
                    i[1] = 1
                    helper(i, 
                        path = path, 
                        firename = firename, 
                        parallel = parallel)         

        return data

class Geomac():
    """
    Basic scraping tool for geomac datasets hosted by ArcGIS. See:

    https://developers.arcgis.com/rest/services-reference/
    layer-feature-service-.html

    for information on attributes.
    """

    def __init__(self, year):
        """
        Initilize by finding meta data for the feature service, a feature layer,
        and all non-geometry data for fires in that layer.

        Parameters:
            :int year:      The year (2001-2019)
        """

        self.baseurl = (
            "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
            "/Historic_GeoMAC_Perimeters_{}/"
            "FeatureServer").format(year)

        self.feature = self.baseurl + "?f=pjson"

        self.layer = self.baseurl + "/0?f=pjson"

        self.first = self.baseurl + "/0/1?f=pjson"

        self.fires = self.baseurl + (
            "/0/query?where=1%3D1&outfields=*"
            "&returnGeometry=false&outSR=4326&f=pjson")

        self.feature = json.loads((urllib.request.urlopen(self.feature)).read())
        self.layer = json.loads((urllib.request.urlopen(self.layer)).read())
        self.first = json.loads((urllib.request.urlopen(self.first)).read())
        self.fires = json.loads((urllib.request.urlopen(self.fires)).read())

    def list_fires(self):
        """
        Returns an abbreviated list for all the fires during that year.

        Parameters:
            None

        Returns:
            :list fires:        Fires list.
        """

        fires = [] 
        for i in self.fires['features']:
            ats = i['attributes']
            fire = {}
            fire['incidentname'] = ats['incidentname']
            fire['uniquefireidentifier'] = ats['uniquefireidentifier']

            fires.append(fire)

        return fires

    def perimeters(self, uniquefireidentifier, animate = False):
        """
        Returns the geometry and date of a fire's perimeters.

        Parameters:
            :str uniquefireidentifier:      The unique fire identifier.

        Returns:
            :list perimeters:       Time/date sorted list of fire perims.
            :bool animate:          Returns an animation depicting fire.
        """

        que = (self.baseurl + 
            "/0/query?where=UNIQUEFIREIDENTIFIER%3D" +
            "'{}'".format(uniquefireidentifier) + 
            "&outFields=perimeterdatetime,state" +
            "&geometryType=esriGeometryPolygon" + 
            "&outSR=4326&f=geojson")

        print("\nSearching for perimeters with query:", "\n"+que, "\n")

        req = urllib.request.urlopen(que)
        res = json.loads(req.read())

        perims = res['features']
        perims.sort(key=lambda x: x['properties']['perimeterdatetime'])

        first = shape(perims[0]['geometry']).bounds

        # first create bounding box
        ll = {
            "longitude": first[0],
            "latitude": first[1]
        }

        ur = {
            "longitude": first[2],
            "latitude": first[3],
        }

        rbounds = {}
        rbounds["ll"] = ll
        rbounds["ur"] = ur

        for i in perims:
            # x = horiz = longitude
            # y = vert  = latitude

            shp = shape(i['geometry'])
            bounds = shp.bounds

            if (bounds[0] < ll["longitude"]):
                ll["longitude"] = bounds[0]
            if (bounds[1] < ll["latitude"]):
                ll["latitude"] = bounds[1]
            if (bounds[2] > ur["longitude"]):
                ur["longitude"] = bounds[2]
            if (bounds[3] > ur["latitude"]):
                ur["latitude"] = bounds[3]

        # Construct timezone sensitive datetimes
        dates = [] 
        for p in perims:
            dates.append(p['properties']['perimeterdatetime'])

        tf = TimezoneFinder()
        lat_av = np.average([float(ll["latitude"]), float(ur["latitude"])])
        lon_av = np.average([float(ll["longitude"]), float(ur["longitude"])])
        tz = tf.timezone_at(lng=lon_av, lat=lat_av)

        # Epoch time reported with a couple extra 0's, shave'em off.
        dates = [datetime.datetime.fromtimestamp(
                t / 1000, 
                tz = pytz.timezone(tz)) for t in dates]

        dates = [x.strftime("%Y-%m-%d|%H:%M:%S") for x in dates]
        for p,d in zip(perims, dates):
            p["properties"]["perimeterdatetime"] = d

        if (animate):

            plotter(perims)

            # Print and play
            print("Animating", len(dates), "images.")
            [print(d) for d in dates]
        
        return perims, rbounds

class Mrlc():
    """
    Simple wrapper class for the MRLC api.

    See:
    https://docs.geoserver.org/stable/en/user/services/wms/reference.html
    """

    def __init__(self):
        """
        Initializes the mrlc search; just collects the capabilities of the
        search.
        """

        self.baseurl = ("https://www.mrlc.gov/geoserver/mrlc_display/"
            "wms?service=WMS&request=GetMap&")

        self.capabilities = urllib.request.urlopen(
            ("https://www.mrlc.gov/geoserver/mrlc_display/wms?service"
                "=WMS&request=GetCapabilities")).read()

    def search(
        self,
        fire_name,
        layer,
        crs,
        height,
        width,
        outformat,
        bbox):

        """
        A quick and dirty implementation of a search / save API
        for the MRLC database for land cover.
        """

        query = (self.baseurl + 
            (
            "bbox={0}&"
            "layers={1}&"
            "epsg={2}&"
            "format={3}&"
            "height={4}&"
            "width={5}&"
            ).format(
                ','.join([str(i) for i in bbox]),
                layer,
                crs,
                outformat,
                str(height),
                str(width)
                ))

        path = Path((Path.cwd().parents[0]) / "data"/fire_name / "preprocessed"
                / "clipped")

        if (not path.exists()):
            path.mkdir(parents=True)

        outpath = path / ("LANDCOV" + ".TIF")

        print("Downloading landmap data with query:")
        print(query)

        download_url(
            url = query,
            output_path = outpath,
            position = 0)

def plotter(perims):
    """
    I think this might have given me an ucler because of the amount of
    debugging it took. If you see flickering borders, it's probably because
    they were misreported by whichever service reported them.

    Parameters:
        :dict perim:        GeoJSON reponse from API calls.
        :bool full:         Whether or not all parts of the fire are plotted.
    
    Returns:
        :matplotlib pyplot: Pyplot image.
    """

    shps = [shape(x['geometry']) for x in perims]

    plt.ioff()
    fig = plt.figure()
    ims = []

    print("Compiling animation...")
    for s in tqdm(shps):
        if (s.geometryType() == "Polygon"):
            x,y = s.exterior.xy
            ims.append(plt.plot(x,y, color='red'))
        elif (s.geometryType() == "MultiPolygon"):
            lines = []
            for geom in s.geoms:
                x,y = geom.exterior.xy
                lines += plt.plot(x,y, color='red')

            ims.append(lines)

    anim = ani.ArtistAnimation(fig, ims, interval=500, repeat_delay=2500)
    plt.show()