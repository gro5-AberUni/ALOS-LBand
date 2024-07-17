import os
import glob
from osgeo import ogr
from osgeo import gdal
import argparse
import sys

parser = argparse.ArgumentParser(prog='Processing of SALOS ScanSAR images')
parser.add_argument('-lnc', metavar='', type=int, help='Low Backscatter Num Clusters. Default is 20', default=20)
parser.add_argument('-mnc', metavar='', type=int, help='Main Backscatter Num Clusters. Default is 250', default=250)
parser.add_argument('-os', metavar='', type=int, help='Segmentation Object Size. Default is 5', default=5)
args = parser.parse_args()

obSize = args.os
lowBackscatterNumClumps = args.lnc
mainBackscatterNumClumps = args.mnc


cwd = os.getcwd()

listComps = glob.glob('/data/InputTiles/*.zip')

listCmds = []
for comp in listComps:

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
    if os.path.exists('/data/ALOS_ENV/HydroData/Hand_Merit-SouthAmerica_COG.tif'):
        print('HAND File Found')
        hand = '/data/ALOS_ENV/HydroData/Hand_Merit-SouthAmerica_COG.tif'
    else:
        print('HAND Image Not Found')
        print('Please Ensure the File: Hand_Merit-SouthAmerica_COG.tif is In the ALOS_ENV/HydroData/ Folder, mounted in the Docker Image Data Location')
        sys.exit()

    #### Slope ####

    if os.path.exists('/data/ALOS_ENV/HydroData/Slope_SouthAmerica_COG.tif'):
        print('Slope File Found')
        slope = '/data/ALOS_ENV/HydroData/Slope_SouthAmerica_COG.tif'
    else:
        print('Slope Image Not Found')
        print('Please Ensure the File: Slope_SouthAmerica_COG.tif is In the ALOS_ENV/HydroData/ Folder, mounted in the Docker Image Data Location')
        sys.exit()

    #### ScalerWater ####

    if os.path.exists('/data/ALOS_ENV/Model/Scaler_Water.pkl'):
        print('Scaler Water File Found')
        scalerWater = '/data/ALOS_ENV/Model/Scaler_Water.pkl'
    else:
        print('Scaler Water Not Found')
        print('Please Ensure the File: /data/ALOS_ENV/Model/Scaler_Water.pkl is In the ALOS_ENV/Model/ Folder, mounted in the Docker Image Data Location')
        sys.exit()

    #### Scaler Flood ####

    if os.path.exists('/data/ALOS_ENV/Model/Scaler_Flood.pkl'):
        print('Scaler Flood File Found')
        scalerFlood = '/data/ALOS_ENV/Model/Scaler_Flood.pkl'
    else:
        print('Scaler Flood Not Found')
        print('Please Ensure the File: /data/ALOS_ENV/Model/Scaler_Flood.pkl is In the ALOS_ENV/Model/ Folder, mounted in the Docker Image Data Location')
        sys.exit()

    #### Water Model ####

    if os.path.exists('/data/ALOS_ENV/Model/Trained_XGBoostModel_02_Amazon_Water.model'):
        print('Water Model File Found')
        waterModel = '/data/ALOS_ENV/Model/Trained_XGBoostModel_02_Amazon_Water.model'
    else:
        print('Water Model File Not Found')
        print('Please Ensure the File: /data/ALOS_ENV/Model/Trained_XGBoostModel_02_Amazon_Water.model is In the ALOS_ENV/Model/ Folder, mounted in the Docker Image Data Location')
        sys.exit()

    #### Water Flood ####

    if os.path.exists('/data/ALOS_ENV/Model/Trained_XGBoostModel_02_Amazon_Flood.model'):
        print('Water Model File Found')
        floodModel = '/data/ALOS_ENV/Model/Trained_XGBoostModel_02_Amazon_Flood.model'
    else:
        print('Water Model File Not Found')
        print('Please Ensure the File: /data/ALOS_ENV/Model/Trained_XGBoostModel_02_Amazon_Flood.model is In the ALOS_ENV/Model/ Folder, mounted in the Docker Image Data Location')
        sys.exit()

    #### LCC Mask ####

    if os.path.exists('/data/ALOS_ENV/LCC/Combined_LCC-GFC_LY-16_Binary_COG_Blank_COG.tif'):
        print('Mask Image Found')
        lcc = '/data/ALOS_ENV/LCC/Combined_LCC-GFC_LY-16_Binary_COG_Blank_COG.tif'
    else:
        print('Mask Image Not Found')
        print('Please Ensure the File: /data/ALOS_ENV/LCC/Combined_LCC-GFC_LY-16_Binary_COG_Blank_COG.tif is In the ALOS_ENV/LCC/ Folder, mounted in the Docker Image Data Location')
        sys.exit()

    #### AOI ####

    if os.path.exists('/data/ALOS_ENV/AOI/South_America_AOI.geojson'):
        print('AOI File Found')
        aoi = '/data/ALOS_ENV/AOI/South_America_AOI.geojson'
    else:
        print('AOI File Not Found')
        print('Please Ensure the File: /data/ALOS_ENV/AOI/South_America_AOI.geojson is In the ALOS_ENV/AOI/ Folder, mounted in the Docker Image Data Location')
        sys.exit()

    # cmd = 'python3 applyXGBoostClassificationImgs_Bins.py -i {0} -j 1 -hd {1} -sl {2} -skw {3} -skf {4} -cmw {5} -cmf {6} -lc {7} -v {8}\n'.format(compFile,hand,slope,scalerWater,scalerFlood,waterModel,floodModel,lcc,aoi)
    cmd = 'python applyXGBoostClassificationImgs_Bins.py -i {0} -j 1 -hd {1} -sl {2} -skw {3} -skf {4} -cmw {5} -cmf {6} -lc {7} -v {8} -lnc {9} -mnc {10} -os {11}\n'.format(compFile,hand,slope,scalerWater,scalerFlood,waterModel,floodModel,lcc,aoi,lowBackscatterNumClumps,mainBackscatterNumClumps,obSize)
    listCmds.append(cmd)

    os.chdir(cwd)
os.chdir(cwd)

with open('classJob.sh','w') as f:
    for cmd in listCmds:
        f.write(cmd)
