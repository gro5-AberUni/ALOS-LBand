# ALOS-LBand
RadWet-L ALOS-2 PALSAR-2 ScanSAR Classification 

These Scripts are used for classifing ALOS-2 PALSAR-2 ScanSAR tiles, as produced by the K&C project. To run these Scripts you will need the following directory Structure:

Mapping:
<br />&emsp;applyXGBoostClassificationImgs_Bins.py
<br />&emsp;genClassApplication_Bins.py
<br />&emsp;Cycle_Dates.csv
<br />&emsp;mergeOrbitCycles_V2.py
<br />&emsp;AOI:
<br />&emsp;&emsp;&emsp;&emsp;Amazon_Basin.geojson
<br />&emsp;HydroData:
<br />&emsp;&emsp;&emsp;&emsp;Hand_MERIT-Amazon_COG.tif
<br />&emsp;&emsp;&emsp;&emsp;Slope-Amazon_COG.tif
<br />&emsp;InputTiles 
<br />&emsp;&emsp;&emsp;&emsp;'This is where you would store ALOS Tiles for classification'
<br />&emsp;LCC
<br />&emsp;&emsp;&emsp;&emsp;Combined_LCC-GFC_LY-16_Binary_COG.tif
<br />&emsp;Model
<br />&emsp;&emsp;&emsp;&emsp;Scaler_Flood.pkl
<br />&emsp;&emsp;&emsp;&emsp;Scaler_Water.pkl
<br />&emsp;&emsp;&emsp;&emsp;Trained_XGBoostModel_02_Amazon_Flood.model
<br />&emsp;&emsp;&emsp;&emsp;Trained_XGBoostModel_02_Amazon_Water.model
		
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








