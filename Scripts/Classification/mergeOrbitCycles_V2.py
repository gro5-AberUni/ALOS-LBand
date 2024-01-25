import pandas as pd
import glob
import os
import rsgislib
from rsgislib import imageutils

csvDates = 'Cycle_Dates.csv'

aoiRast = '/data/gregxgb/ALOS/AOI/Valid_Area_Raster_Fill_NoData.kea'

datesPD = pd.read_csv(csvDates,delimiter=';')

print(datesPD)

cwd=os.getcwd()

for index, row in datesPD.iterrows():
    orbitCycle = row['Cycle']
    print(orbitCycle)
    listOrbitRowsClassDirs = glob.glob('ALOS-Output*{0}'.format(orbitCycle))
    listMergeFiles = []
    for dir in listOrbitRowsClassDirs:
        os.chdir(dir)
        try:
            listMergeFiles.append(os.path.abspath(glob.glob('*Classified*.tif')[0]))
        except:
            print('No Classified Image Found')
        os.chdir(cwd)

    classFile = 'Classified_Output_Orbit-Cycle_{0}-Dated-{1}_{2}.tif'.format(orbitCycle,row['Start'].replace('/','-'),row['End'].replace('/','-'))
    if len(listOrbitRowsClassDirs)!=0:
        rsgislib.imageutils.create_img_mosaic(listMergeFiles, classFile, 0, 0, 1,1, 'GTIFF', 1)

        outputFileMsk = classFile.replace('.tif','_VM.tif')

        rsgislib.imageutils.mask_img(classFile, aoiRast, outputFileMsk, 'GTIFF', 1, 7,0)

        clr_lut = dict()
        clr_lut[0] = '#000000'
        clr_lut[1] = '#6CABDD'
        clr_lut[2] = '#000080'
        clr_lut[4] = '#004225'
        clr_lut[5] = '#a7a7a7'
        clr_lut[6] = '#d21255'
        clr_lut[7] = '#FFFFFF'

        rsgislib.imageutils.define_colour_table(outputFileMsk, clr_lut)

