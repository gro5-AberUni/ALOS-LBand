import pandas as pd
import glob
import os
import rsgislib
import numpy as np

rsgislib.imageutils.set_env_vars_lzw_gtiff_outs(True)

csvDates = 'Cycle_Dates_TS.csv'

datesPD = pd.read_csv(csvDates)

print(datesPD)

# cwd=os.getcwd()
data_dir = '/data/'

for index, row in datesPD.iterrows():
    orbitCycle = row['Cycle']
    print(orbitCycle)
    if orbitCycle < 10:
        orbitCycle = '00{0}'.format(orbitCycle)

    elif orbitCycle < 100:
        orbitCycle = '{0}'.format(orbitCycle)
    print(orbitCycle)
    listOrbitRowsClassDirs = glob.glob('./ALOS-Output*-{0}_1*/'.format(orbitCycle))

    if len(listOrbitRowsClassDirs) == 0:
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

    classFile = f"Classified_Output_Orbit-Cycle_{orbitCycle,row['Start'].replace('/','-'),row['End'].replace('/','-')}_Total.tif"
    if len(listOrbitRowsClassDirs)!=0:
        rsgislib.imageutils.create_img_mosaic(listMergeFiles, classFile, 0, 0, 1,1, 'GTIFF', 1)

        clr_lut = dict()
        clr_lut[0] = '#000000'
        clr_lut[1] = '#6CABDD'
        clr_lut[2] = '#000080'
        clr_lut[4] = '#004225'
        clr_lut[5] = '#a7a7a7'
        clr_lut[6] = '#d21255'
        clr_lut[7] = '#FFFFFF'

        rsgislib.imageutils.define_colour_table(classFile, clr_lut)
        rsgislib.imageutils.pop_thmt_img_stats(classFile,add_clr_tab=False)

    #### Get Even Orbit Paths ####

    listRSP = []
    for fn in listMergeFiles:
        rsp = fn.split('/')[-1].split('_')[-2][:3]
        print(rsp)
        listRSP.append(rsp)
    uniqueRSP = np.unique(listRSP)

    print(listRSP)
    print(uniqueRSP)

    if len(uniqueRSP) > 0:
        print(uniqueRSP)

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
        rsgislib.imageutils.create_img_mosaic(listEvenImg, classFile, 0, 0, 1,1, 'GTIFF', 1)
        print(listEvenImg)

        clr_lut = dict()
        clr_lut[0] = '#000000'
        clr_lut[1] = '#6CABDD'
        clr_lut[2] = '#000080'
        clr_lut[4] = '#004225'
        clr_lut[5] = '#a7a7a7'
        clr_lut[6] = '#d21255'
        clr_lut[7] = '#FFFFFF'

        rsgislib.imageutils.define_colour_table(classFile, clr_lut)
        rsgislib.imageutils.pop_thmt_img_stats(classFile,add_clr_tab=False)

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
        rsgislib.imageutils.create_img_mosaic(listOddImg, classFile, 0, 0, 1,1, 'GTIFF', 1)

        clr_lut = dict()
        clr_lut[0] = '#000000'
        clr_lut[1] = '#6CABDD'
        clr_lut[2] = '#000080'
        clr_lut[4] = '#004225'
        clr_lut[5] = '#a7a7a7'
        clr_lut[6] = '#d21255'
        clr_lut[7] = '#FFFFFF'

        rsgislib.imageutils.define_colour_table(classFile, clr_lut)
        rsgislib.imageutils.pop_thmt_img_stats(classFile,add_clr_tab=False)
