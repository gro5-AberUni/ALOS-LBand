import pandas as pd
import glob
import rsgislib
from rsgislib import imageutils
import numpy as np
import logging
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO
)
logger = logging.getLogger(__name__)

rsgislib.imageutils.set_env_vars_lzw_gtiff_outs(True)

csvDates = 'Cycle_Dates_TS.csv' # Could/should pass this as an argument

try:
    datesPD = pd.read_csv(csvDates)
    logger.info(f"Loaded cycle dates:\n{datesPD}")
except FileNotFoundError:
    raise FileNotFoundError(f"CSV file '{csvDates}' not found.")

CLASS_COLOR_LUT = {
    0: "#000000",
    1: "#6CABDD",
    2: "#000080",
    4: "#004225",
    5: "#a7a7a7",
    6: "#d21255",
    7: "#FFFFFF",
}

def create_classified_mosaic(img_list, output_file, color_lut):
    if len(img_list) > 0:
        rsgislib.imageutils.create_img_mosaic(
            img_list, output_file, 0, 0, 1, 1, "GTIFF", 1
        )
        rsgislib.imageutils.define_colour_table(output_file, color_lut)
        rsgislib.imageutils.pop_thmt_img_stats(output_file, add_clr_tab=False)

data_dir = '/data/' # Could/should pass this as an argument

for index, row in datesPD.iterrows():
    orbitCycle = row['Cycle']
    orbitCycle = str(orbitCycle).zfill(3)
    logger.info(f'Starting orbitCycle {orbitCycle}')

    listOrbitRowsClassDirs = glob.glob(f'{data_dir}/ALOS-Output*-{orbitCycle}_1*/')

    if len(listOrbitRowsClassDirs) == 0:
        logger.info(f'No data found for Orbit Cycle: {orbitCycle}')
        continue # Go to next if no data for this orbit cycle

    listMergeFiles = glob.glob(f"{data_dir}/ALOS-Output*-{orbitCycle}_1*/*Classified*.tif")

    classFile = f"{data_dir}Classified_Output_Orbit-Cycle_{orbitCycle}_Total.tif"
    if len(listMergeFiles) != 0:
        create_classified_mosaic(listMergeFiles, classFile, CLASS_COLOR_LUT)

    #### Get Even Orbit Paths ####

    listRSP = [fn.split('/')[-1].split('_')[-2][:3] for fn in listMergeFiles]
    uniqueRSP = np.unique(listRSP)

    logger.info(f'All RSPs: {listRSP}')
    logger.info(f'Unique RSPs: {uniqueRSP}')

    uniqueEven = np.unique([rsp for rsp in listRSP if int(rsp) % 2 == 0])
    uniqueOdd = np.unique([rsp for rsp in listRSP if int(rsp) % 2 != 0])

    logger.info(f'Unique Even RSPs: {uniqueEven}')
    logger.info(f'Unique Odd RSPs: {uniqueOdd}')

    #### Gather all Even Files ####

    listEvenImg = []
    for evRSP in uniqueEven:

        files = glob.glob(f"{data_dir}/ALOS-Output*-{orbitCycle}_{evRSP}*/*Classified*SLopeM*.tif")
        if len(files) == 0:
            logger.info(f'No Even Images Found for RSP: {evRSP} in Orbit Cycle: {orbitCycle}')
        else:
            listEvenImg.extend(files)

    classFile = f'{data_dir}Classified_Output_Orbit-Cycle_{orbitCycle}-Dated-{row["Start"].replace("/","-")}_{row["End"].replace("/","-")}_Even-RSP_AWS.tif'

    if len(listEvenImg) != 0:
        create_classified_mosaic(listEvenImg, classFile, CLASS_COLOR_LUT)

    #### Gather all Odd Files ####

    listOddImg = []
    for oddRSP in uniqueOdd:
        files = glob.glob(f"{data_dir}/ALOS-Output*-{orbitCycle}_{oddRSP}*/*Classified*SLopeM*.tif")
        if len(files) == 0:
            logger.info(f'No Odd Images Found for RSP: {oddRSP} in Orbit Cycle: {orbitCycle}')
        else:
            listOddImg.extend(files)
    
    classFile = f'{data_dir}Classified_Output_Orbit-Cycle_{orbitCycle}-Dated-{row["Start"].replace("/","-")}_{row["End"].replace("/","-")}_Odd-RSP_AWS.tif'

    if len(listOddImg) != 0:
        create_classified_mosaic(listOddImg, classFile, CLASS_COLOR_LUT)
    
    # Clean up memory at end of each orbit cycle
    del listMergeFiles, listRSP, uniqueRSP, uniqueEven, uniqueOdd
    del listEvenImg, listOddImg, listOrbitRowsClassDirs
    gc.collect()
