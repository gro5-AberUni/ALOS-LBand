import os
import glob
from osgeo import ogr
from osgeo import gdal
import argparse
import sys
import numpy as np

parser = argparse.ArgumentParser(prog='Processing of SALOS ScanSAR images')
parser.add_argument('-lnc', metavar='', type=int, help='Low Backscatter Num Clusters. Default is 20', default=20)
parser.add_argument('-mnc', metavar='', type=int, help='Main Backscatter Num Clusters. Default is 250', default=250)
parser.add_argument('-os', metavar='', type=int, help='Segmentation Object Size. Default is 5', default=5)

parser.add_argument('-ldt', metavar='', type=int, help='Low Backscatter Distance Threshold. Default is 10', default=10)
parser.add_argument('-mdt', metavar='', type=int, help='Main Backscatter Distance Threshold. Default is 10', default=10)

args = parser.parse_args()

obSize = args.os
lowBackscatterNumClumps = args.lnc
mainBackscatterNumClumps = args.mnc
lowBackscatterDT = args.ldt
mainBackscatterDT = args.mdt

cwd = os.getcwd()

listCompsRAW = glob.glob('/data/InputTiles/*.zip')

listCyclesNums = []

for compRaw in listCompsRAW:
    cycleNum = compRaw.split('/')[-1].split('_')[1]
    listCyclesNums.append(cycleNum)
uniqueCycles = np.unique(listCyclesNums)
sortedCycles = sorted(uniqueCycles)

eastingTiles = []

for compRaw in listCompsRAW:
    eastingNum = compRaw.split('/')[-1].split('_')[0][3:7]
    eastingTiles.append(eastingNum)

uniqueEastings = np.unique(eastingTiles)
sortedEastings = sorted(uniqueEastings)

northingTiles = []

for compRaw in listCompsRAW:
    northingNum = compRaw.split('/')[-1].split('_')[0][0:3]
    northingTiles.append(northingNum)

uniqueNorthing = np.unique(northingTiles)
sortedNorthing = sorted(uniqueNorthing)

processTiles = []

for cycle in sortedCycles:
    for east in sortedEastings:
        for north in sortedNorthing:
            inputFiles = glob.glob('/data/InputTiles/{0}{1}*{2}*.zip'.format(north,east,cycle))
            if len(inputFiles) > 0:
                for zip in inputFiles:
                    processTiles.append(zip)

listCmds = []
for comp in processTiles:

    compFile = os.path.abspath(comp)

    # if os.path.getsize(compFile) < 35000000:
    #     print('Error Scene: {0}'.format(compFile))
    # else:

    print(comp)
    compDate = comp.split('/')[-1].split('_')[3]

    print(compDate)


    yr = compDate[0:4]
    print(yr)
    mnth = compDate[4:6]
    print(mnth)

    #### HAND ####

    #hand = os.path.abspath('/data/ALOS_ENV/HydroData/Hand_Merit-SouthAmerica_COG.tif')
    if os.path.exists('/data/ALOS_Africa/HydroData/Africa_HAND.tif'):
        print('HAND File Found')
        hand = '/data/ALOS_Africa/HydroData/Africa_HAND.tif'
    else:
        print('HAND Image Not Found')
        print('Please Ensure the File: Africa_HAND is In the ALOS_Africa/HydroData/ Folder, mounted in the Docker Image Data Location')
        sys.exit()

    #### Slope ####

    if os.path.exists('/data/ALOS_Africa/HydroData/Africa_Slope.tif'):
        print('Slope File Found')
        slope = '/data/ALOS_Africa/HydroData/Africa_Slope.tif'
    else:
        print('Slope Image Not Found')
        print('Please Ensure the File: Africa_Slope is In the ALOS_Africa/HydroData/ Folder, mounted in the Docker Image Data Location')
        sys.exit()

    #### ScalerWater ####

    if os.path.exists('/data/ALOS_Africa/Model/Scaler_Water.pkl'):
        print('Scaler Water File Found')
        scalerWater = '/data/ALOS_Africa/Model/Scaler_Water.pkl'
    else:
        print('Scaler Water Not Found')
        print('Please Ensure the File: /data/ALOS_Africa/Model/Scaler_Water.pkl is In the ALOS_Africa/Model/ Folder, mounted in the Docker Image Data Location')
        sys.exit()

    #### Scaler Flood ####

    if os.path.exists('/data/ALOS_Africa/Model/Scaler_Flood.pkl'):
        print('Scaler Flood File Found')
        scalerFlood = '/data/ALOS_Africa/Model/Scaler_Flood.pkl'
    else:
        print('Scaler Flood Not Found')
        print('Please Ensure the File: /data/ALOS_Africa/Model/Scaler_Flood.pkl is In the ALOS_Africa/Model/ Folder, mounted in the Docker Image Data Location')
        sys.exit()

    #### Water Model ####

    if os.path.exists('/data/ALOS_Africa/Model/Trained_XGBoostModel_02_Amazon_Water.model'):
        print('Water Model File Found')
        waterModel = '/data/ALOS_Africa/Model/Trained_XGBoostModel_02_Amazon_Water.model'
    else:
        print('Water Model File Not Found')
        print('Please Ensure the File: /data/ALOS_Africa/Model/Trained_XGBoostModel_02_Amazon_Water.model is In the ALOS_Africa/Model/ Folder, mounted in the Docker Image Data Location')
        sys.exit()

    #### Water Flood ####

    if os.path.exists('/data/ALOS_Africa/Model/Trained_XGBoostModel_02_Amazon_Flood.model'):
        print('Water Model File Found')
        floodModel = '/data/ALOS_Africa/Model/Trained_XGBoostModel_02_Amazon_Flood.model'
    else:
        print('Water Model File Not Found')
        print('Please Ensure the File: /data/ALOS_Africa/Model/Trained_XGBoostModel_02_Amazon_Flood.model is In the ALOS_Africa/Model/ Folder, mounted in the Docker Image Data Location')
        sys.exit()

    #### LCC Mask ####

    if os.path.exists('/data/ALOS_Africa/LCC/Africa_Blank_Mask.tif'):
        print('Mask Image Found')
        lcc = '/data/ALOS_Africa/LCC/Africa_Blank_Mask.tif'
    else:
        print('Mask Image Not Found')
        print('Please Ensure the File: /data/ALOS_Africa/LCC/Africa_Blank_Mask.tif is In the ALOS_Africa/LCC/ Folder, mounted in the Docker Image Data Location')
        sys.exit()

    #### AOI ####

    if os.path.exists('/data/ALOS_Africa/AOI/Africa_AOI.geojson'):
        print('AOI File Found')
        aoi = '/data/ALOS_Africa/AOI/Africa_AOI.geojson'
    else:
        print('AOI File Not Found')
        print('Please Ensure the File: /data/ALOS_Africa/AOI/Africa_AOI.geojson is In the ALOS_Africa/AOI/ Folder, mounted in the Docker Image Data Location')
        sys.exit()

    # cmd = 'python3 applyXGBoostClassificationImgs_Bins.py -i {0} -j 1 -hd {1} -sl {2} -skw {3} -skf {4} -cmw {5} -cmf {6} -lc {7} -v {8}\n'.format(compFile,hand,slope,scalerWater,scalerFlood,waterModel,floodModel,lcc,aoi)
    #cmd = 'python applyXGBoostClassificationImgs_Bins.py -i {0} -j 1 -hd {1} -sl {2} -skw {3} -skf {4} -cmw {5} -cmf {6} -lc {7} -v {8} -lnc {9} -mnc {10} -os {11}\n'.format(compFile,hand,slope,scalerWater,scalerFlood,waterModel,floodModel,lcc,aoi,lowBackscatterNumClumps,mainBackscatterNumClumps,obSize)
    cmd = 'python applyXGBoostClassificationImgs_Bins.py -i {0} -j 1 -hd {1} -sl {2} -skw {3} -skf {4} -cmw {5} -cmf {6} -lc {7} -v {8} -lnc {9} -mnc {10} -os {11} -ldt {12} -mdt {13}\n'.format(compFile,hand,slope,scalerWater,scalerFlood,waterModel,floodModel,lcc,aoi,lowBackscatterNumClumps,mainBackscatterNumClumps,obSize,lowBackscatterDT,mainBackscatterDT)
    listCmds.append(cmd)

    os.chdir(cwd)
os.chdir(cwd)

with open('classJob.sh','w') as f:
    for cmd in listCmds:
        f.write(cmd)
