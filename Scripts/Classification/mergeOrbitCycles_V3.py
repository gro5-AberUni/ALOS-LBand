import pandas as pd
import glob
import os
import rsgislib
from rsgislib import imageutils

rsgislib.imageutils.set_env_vars_lzw_gtiff_outs(True)

csvDates = 'Cycle_Dates_TS.csv'

aoiRast = '/data/ALOS_ENV/AOI/Valid_Area_South_America_Fill-NoData.kea'

datesPD = pd.read_csv(csvDates)

print(datesPD)

cwd=os.getcwd()

for index, row in datesPD.iterrows():
    orbitCycle = row['Cycle']
    print(orbitCycle)
    if orbitCycle < 10:
        orbitCycle = '00{0}'.format(orbitCycle)

    elif orbitCycle < 100:
        orbitCycle = '0{0}'.format(orbitCycle)
    print(orbitCycle)
    listOrbitRowsClassDirs = glob.glob('/data/ALOS-Output*C{0}*'.format(orbitCycle))
    listMergeFiles = []
    for dir in listOrbitRowsClassDirs:
        os.chdir(dir)
        try:
            listMergeFiles.append(os.path.abspath(glob.glob('*Classified*SLopeM.tif')[0]))
        except:
            print('No Classified Image Found')
        os.chdir(cwd)

    classFile = '/data/Classified_Output_Orbit-Cycle_{0}-Dated-{1}_{2}.tif'.format(orbitCycle,row['Start'].replace('/','-'),row['End'].replace('/','-'))
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
        rsp = fn.split('/')[-1].split('_')[2]
        listRSP.append(rsp)
    uniqueRSP = np.unique(listRSP)

    listEven = []
    listOdd = []
    for rsp in listRSP:
        rspInt = int(rsp[3:])
        if rspInt % 2 == 0:
            listEven.append(rsp)
        else:
            listOdd.append(rsp)
    print(listEven)
    print(listOdd)

    #### Gather all Even Files ####

    listEvenImg = []
    for evRSP in listEven:
        listOrbitRowsClassDirs = glob.glob('/data/ALOS-Output*C{0}_{1}'.format(orbitCycle,evRSP))
        for dir in listOrbitRowsClassDirs:
            os.chdir(dir)
            try:
                listEvenImg.append(os.path.abspath(glob.glob('*Classified*SLopeM.tif')[0]))
            except:
                print('No Classified Image Found')
            os.chdir(cwd)
        classFile = '/data/Classified_Output_Orbit-Cycle_{0}-Dated-{1}_{2}_Even-RSP.tif'.format(orbitCycle,row['Start'].replace('/','-'),row['End'].replace('/','-'))
    if len(listOrbitRowsClassDirs)!=0:
        rsgislib.imageutils.create_img_mosaic(listEvenImg, classFile, 0, 0, 1,1, 'GTIFF', 1)

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
    for oddRSP in listOdd:
        listOrbitRowsClassDirs = glob.glob('/data/ALOS-Output*C{0}_{1}'.format(orbitCycle,oddRSP))
        for dir in listOrbitRowsClassDirs:
            os.chdir(dir)
            try:
                listEvenImg.append(os.path.abspath(glob.glob('*Classified*SLopeM.tif')[0]))
            except:
                print('No Classified Image Found')
            os.chdir(cwd)
        classFile = '/data/Classified_Output_Orbit-Cycle_{0}-Dated-{1}_{2}_Odd-RSP.tif'.format(orbitCycle,row['Start'].replace('/','-'),row['End'].replace('/','-'))
    if len(listOrbitRowsClassDirs)!=0:
        rsgislib.imageutils.create_img_mosaic(listEvenImg, classFile, 0, 0, 1,1, 'GTIFF', 1)

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
        
        

