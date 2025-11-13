import pandas as pd
import glob
import os
import rsgislib
from rsgislib import imageutils
import numpy as np

rsgislib.imageutils.set_env_vars_lzw_gtiff_outs(True)

csvDates = 'Cycle_Dates_TS.csv'

datesPD = pd.read_csv(csvDates)

print(datesPD)


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


# cwd=os.getcwd()
data_dir = '/data/'

for index, row in datesPD.iterrows():
    orbitCycle = row['Cycle']
    orbitCycle = str(orbitCycle).zfill(3)
    print(f'Starting orbitCycle {orbitCycle}')

    listOrbitRowsClassDirs = glob.glob(f'{data_dir}/ALOS-Output*-{orbitCycle}_1*/')

    if len(listOrbitRowsClassDirs) == 0:
        print(f'No data found for Orbit Cycle: {orbitCycle}')
        continue # Go to next if no data for this orbit cycle

    listMergeFiles = []
    for dir in listOrbitRowsClassDirs:
        os.chdir(dir)
        try:
            listMergeFiles.append(os.path.abspath(glob.glob('*Classified*.tif')[0]))
        except Exception as e:
            print('No Classified Image Found')
            print(e)
        os.chdir(data_dir)

    classFile = f"Classified_Output_Orbit-Cycle_{orbitCycle}_Total.tif"
    if len(listOrbitRowsClassDirs)!=0:
        create_classified_mosaic(listMergeFiles, classFile, CLASS_COLOR_LUT)

    #### Get Even Orbit Paths ####

    listRSP = []
    for fn in listMergeFiles:
        rsp = fn.split('/')[-1].split('_')[-2][:3]
        print(rsp)
        listRSP.append(rsp)
    uniqueRSP = np.unique(listRSP)

    print(listRSP)
    print(uniqueRSP)

    # if len(uniqueRSP) > 0:
    #     print(uniqueRSP)

    listEven = []
    listOdd = []
    for rsp in listRSP:
        rspInt = int(rsp)
        if rspInt % 2 == 0:
            listEven.append(rsp)
        else:
            listOdd.append(rsp)
    print(listEven)
    print(listOdd)

    uniqueEven = np.unique(listEven)
    uniqueOdd = np.unique(listOdd)

    print(uniqueEven)
    print(uniqueOdd)

    #### Gather all Even Files ####

    listEvenImg = []
    for evRSP in uniqueEven:
        listOrbitRowsClassDirs = glob.glob('ALOS-Output*-{0}_{1}*'.format(orbitCycle,evRSP))
        print(listOrbitRowsClassDirs)
        for dir in listOrbitRowsClassDirs:
            os.chdir(dir)
            try:
                listEvenImg.append(os.path.abspath(glob.glob('*Classified*SLopeM*.tif')[0]))
            except Exception as e:
                print('No Classified Image Found')
                print(e)
            os.chdir(data_dir)
        classFile = 'Classified_Output_Orbit-Cycle_{0}-Dated-{1}_{2}_Even-RSP_AWS.tif'.format(orbitCycle,row['Start'].replace('/','-'),row['End'].replace('/','-'))
    if len(listOrbitRowsClassDirs)!=0:
        create_classified_mosaic(listEvenImg, classFile, CLASS_COLOR_LUT)

    #### Gather all Odd Files ####

    listOddImg = []
    for oddRSP in uniqueOdd:
        listOrbitRowsClassDirs = glob.glob('ALOS-Output*-{0}_{1}*'.format(orbitCycle,oddRSP))
        for dir in listOrbitRowsClassDirs:
            os.chdir(dir)
            try:
                listOddImg.append(os.path.abspath(glob.glob('*Classified*SLopeM*.tif')[0]))
            except Exception as e:
                print('No Classified Image Found')
                print(e)
            os.chdir(data_dir)
        classFile = 'Classified_Output_Orbit-Cycle_{0}-Dated-{1}_{2}_Odd-RSP_AWS.tif'.format(orbitCycle,row['Start'].replace('/','-'),row['End'].replace('/','-'))
    if len(listOrbitRowsClassDirs)!=0:
        create_classified_mosaic(listOddImg, classFile, CLASS_COLOR_LUT)
