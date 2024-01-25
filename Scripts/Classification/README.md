# ALOS-LBand
RadWet-L ALOS-2 PALSAR-2 ScanSAR Classification 

These Scripts are used for classifing ALOS-2 PALSAR-2 ScanSAR tiles, as produced by the K&C project. To run these Scripts you will need the following directory Structure:

Mapping:
<br />applyXGBoostClassificationImgs_Bins.py
<br />genClassApplication_Bins.py
<br />Cycle_Dates.csv
<br />mergeOrbitCycles_V2.py
<br />&emsp;AOI:
<br />Amazon_Basin.geojson
<br />HydroData:
<br />Hand_MERIT-Amazon_COG.tif
<br />Slope-Amazon_COG.tif
<br />InputTiles 
<br />'This is where you would store ALOS Tiles for classification'
<br />LCC
<br />Combined_LCC-GFC_LY-16_Binary_COG.tif
<br />Model
<br />Scaler_Flood.pkl
<br />Scaler_Water.pkl
<br />Trained_XGBoostModel_02_Amazon_Flood.model
<br />Trained_XGBoostModel_02_Amazon_Water.model
		
The script: 'genClassApplication_Bins.py' will be used to create a bash script. Store ALOS-2 PALSAR-2 ScanSAR tiles as zipped files in the 'InputTiles' directory. The 'genClassApplication_Bins.py' script will create a python command for each of the input tiles, and write it to a shell script called: 'classJob.sh'

This shell script will contain the python call to the 'applyXGBoostClassificationImgs_Bins.py' script, with the required arguments to the script already included.

To run the classifier use the command: bash classJob.sh

Dependencies:
python3.9
pandas
xgboost
rios
sklearn
rsgislib
numba

A ready built-cross platform conda environment is provided in the file: radwet.yml








