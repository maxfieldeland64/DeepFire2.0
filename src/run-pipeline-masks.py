from api import *
from pipeline import *

print("\n---First pass: downloading raw data---\n")

perims, bounds = get_raw(
	fire_name = "King", 
	fire_year = 2014, 
	fire_code = '2014-CAENF-023461', 
	usgs_username = "kwkaiser", 
	usgs_password = "T1ckettothefire", 
	parallel = False)

print("\n---Second pass: preprocessing raw data---\n")

reproject_usgs("King", selection=["B1", "B2", "B3", "B4", "B5", "B6"])
stitch_usgs("King")
reproject_nm("King")
stitch_nm("King")
clip_mosaics("King", bounds)
abounds = rasterize_perims("King", perims)

# The MRLC api fortunately has support for cropping / reprojecting.
# Therefore, we download it as already preprocessed.
mr = Mrlc()
mr.search(
	fire_name = "King",
	layer = "NLCD_2013_Land_Cover_L48",
	crs = "EPSG:4326",
	height = abounds[0],
	width = abounds[1],
	outformat = "image%2Fgeotiff",
	bbox = [bounds['ll']['longitude'], bounds['ll']['latitude'], 
			bounds['ur']['longitude'], bounds['ur']['latitude']])

stack_layers(
	fire_name = "King", 
	dataset = "LANDSAT_8_C1", 
	scheme = "wrt_per", 
	order = ["B4", "B2", "B3", "B5", "LIDAR", 
	"windspeed", "winddirection", "PERIM"],
	key = 'ah5NhpMw')

# Prior to sampling, filter stacks to only one per 24/hrs.
select_stacks("King", min_diff = 24)

print("\n---Third pass: sampling from layer stacks to generate data---\n")

for i in ["Tubbs", "Cascade", "County", "Rocky"]:
	generate_samples(i, height = 64, width=64, count=2500, mask_layer = -1, parallel=True, cores = -1, out_height = 32, out_width = 32, binary=False)

	generate_splits(i, 0.6, 0.2, 0.2, parallel=True, binary=False)
