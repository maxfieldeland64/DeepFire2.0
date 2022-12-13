# DeepFire2.0
Continuation of DeepFire, a CNN powered computer vision approach to predicting wildfire spread

Deep fire data autoload
Branch focused on streamlining the data pipeline while Max continues to focus on the model.
Basic data pipeline:

Scrape relevant fire shape files
Find bounding box from fire shape files
Use bounding box to scrape TIFS from landsat database, LIDAR from National Map database
Standardize position systems, match dates, stack, then crop to fire perimeter.
Sample positions from each stack, shuffle, then associate proportions with test, train, and validate.
Create / write out samples.
Read in with a generator.


Tools:
Create a conda environment from the provided pipeline.yml file. Once installed, activate that conda environment
with 'conda activate pipeline'. That should allow you to pretty easily run the pipeline. You may
need to change the $HOME variable in the yml to wherever you want the conda env to end up.