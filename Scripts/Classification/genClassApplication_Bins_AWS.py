import os
import glob
from osgeo import ogr
from osgeo import gdal
import argparse
import sys
import numpy as np
import pandas as pd
import datetime

parser = argparse.ArgumentParser(prog='Processing of SALOS ScanSAR images')
parser.add_argument('-lnc', metavar='', type=int, help='Low Backscatter Num Clusters. Default is 20', default=10)
parser.add_argument('-mnc', metavar='', type=int, help='Main Backscatter Num Clusters. Default is 250', default=25)
parser.add_argument('-os', metavar='', type=int, help='Segmentation Object Size. Default is 5', default=15)

parser.add_argument('-ldt', metavar='', type=int, help='Low Backscatter Distance Threshold. Default is 10', default=10)
parser.add_argument('-mdt', metavar='', type=int, help='Main Backscatter Distance Threshold. Default is 10', default=10)

parser.add_argument('-c', metavar='', type=str, help='Orbit Cycle to Process AWS')


args = parser.parse_args()

obSize = args.os
lowBackscatterNumClumps = args.lnc
mainBackscatterNumClumps = args.mnc
lowBackscatterDT = args.ldt
mainBackscatterDT = args.mdt
c = args.c
cwd = os.getcwd()

listCmds = []

#local = /data



cycleDatesFile = '/data/ALOS_ENV/Cycle_Dates_TS.csv'
cycleLinkDF = pd.read_csv(cycleDatesFile)
print(cycleLinkDF)

alosScenes = '/data/ALOS_ENV/Orbit_Cycles/ALOS-2_PALSAR-2_Central_Amazon_File_List_Orbit_Cycle-{0}_List.csv'.format(c)

dataLoc = '/data/InputTiles/'

alosDF = pd.read_csv(alosScenes)

for index, row in alosDF.iterrows():

    basename = row['Scene ID']
    rsp = row['Path']

    hh = os.path.abspath('{0}{1}_WBDR2.2GUD_HH_SLP.tif'.format(dataLoc,basename))
    print(hh)
    print(os.path.exists(hh))

    if os.path.exists(hh):
        hv = os.path.abspath('{0}{1}_WBDR2.2GUD_HV_SLP.tif'.format(dataLoc,basename))
        lin = os.path.abspath('{0}{1}_WBDR2.2GUD_LIN.tif'.format(dataLoc,basename))

        comp = hh

        print(comp)
        fileID = comp.split('/')[-1]
        print(fileID)

        #rsp = str(rspDataLinkDF.loc[rspDataLinkDF['FilePath'] == comp.split('/')[-1]]['RSP'].values[0])
        print(rsp)
        #.split('_')[0]
        # .split('_')[0]
        print(rsp)
        compFile = os.path.abspath(comp)
        print(comp)
        compDate = comp.split('/')[-1].split('-')[1].split('_')[0]
        print(compDate)

        yr = int('20{0}'.format(compDate[0:2]))
        print(yr)
        mnth = int(compDate[2:4])
        day = int(compDate[4:6])
        print(mnth)
        print(day)

        #### Get Orbit Cycle ####

        date = datetime.date(yr, mnth, day)
        print(date)

        orbitCycle = None

        for index, row in cycleLinkDF.iterrows():
            # print(row)
            startStr = row['Start']
            # print(startStr)

            day, mnth, yr = startStr.split('/')
            startDTObj = datetime.date(int('20{0}'.format(yr)),int(mnth),int(day))
            #print(startDTObj)



            endStr = row['End']
            #print(endStr)
            day, mnth, yr = endStr.split('/')
            endDTObj = datetime.date(int('20{0}'.format(yr)),int(mnth),int(day))
            #print(endDTObj)


            if startDTObj <= date <= endDTObj:
                orbitCycle = row['Cycle']
        if orbitCycle is None:
            print(f'No Orbit Cycle Found for Date: {date}')
            sys.exit()
        else:
            print(f"Orbit Cycle Found: {orbitCycle}")

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

        if os.path.exists('/opt/ALOS-LBand/Scripts/Model/Scaler_Water.pkl'):
            print('Scaler Water File Found')
            scalerWater = '/opt/ALOS-LBand/Scripts/Model/Scaler_Water.pkl'
        else:
            print('Scaler Water Not Found')
            print('Please Ensure the File: /opt/ALOS-LBand/Scripts/Model/Scaler_Water.pkl is In the ALOS_ENV/Model/ Folder, mounted in the Docker Image Data Location')
            sys.exit()

        #### Scaler Flood ####

        if os.path.exists('/opt/ALOS-LBand/Scripts/Model/Scaler_Flood.pkl'):
            print('Scaler Flood File Found')
            scalerFlood = '/opt/ALOS-LBand/Scripts/Model/Scaler_Flood.pkl'
        else:
            print('Scaler Flood Not Found')
            print('Please Ensure the File: /opt/ALOS-LBand/Scripts/Model/Scaler_Flood.pkl is In the ALOS_ENV/Model/ Folder, mounted in the Docker Image Data Location')
            sys.exit()

        #### Water Model ####

        if os.path.exists('/opt/ALOS-LBand/Scripts/Model/Trained_XGBoostModel_02_Amazon_Water.model'):
            print('Water Model File Found')
            waterModel = '/opt/ALOS-LBand/Scripts/Model/Trained_XGBoostModel_02_Amazon_Water.model'
        else:
            print('Water Model File Not Found')
            print('Please Ensure the File: /opt/ALOS-LBand/Scripts/Model/Trained_XGBoostModel_02_Amazon_Water_V5_DevLvl2-2.model is In the ALOS_ENV/Model/ Folder, mounted in the Docker Image Data Location')
            sys.exit()

        #### Water Flood ####

        if os.path.exists('/opt/ALOS-LBand/Scripts/Model/Trained_XGBoostModel_02_Amazon_Flood.model'):
            print('Water Model File Found')
            floodModel = '/opt/ALOS-LBand/Scripts/Model/Trained_XGBoostModel_02_Amazon_Flood.model'
        else:
            print('Water Model File Not Found')
            print('Please Ensure the File: /opt/ALOS-LBand/Scripts/Model/Trained_XGBoostModel_02_Amazon_Flood_V5_DevLvl2-2.model is In the ALOS_ENV/Model/ Folder, mounted in the Docker Image Data Location')
            sys.exit()

        #### LCC Mask ####

        if os.path.exists('/data/ALOS_ENV/LCC/Combined_LCC-GFC_LY-16_Binary_COG_Blank_COG.tif'):
            print('Mask Image Found')
            lcc = '/data/ALOS_ENV/LCC/Combined_LCC-GFC_LY-16_Binary_COG_Blank_COG.tif'
        else:
            print('Mask Image Not Found')
            print('Please Ensure the File: /home/greg/Documents/ALOS_ENV/LCC/Combined_LCC-GFC_LY-16_Binary_COG_Blank_COG.tif is In the ALOS_ENV/LCC/ Folder, mounted in the Docker Image Data Location')
            sys.exit()

        #### AOI ####

        if os.path.exists('/data/ALOS_ENV/AOI/South_America_AOI.geojson'):
            print('AOI File Found')
            aoi = '/data/ALOS_ENV/AOI/South_America_AOI.geojson'
        else:
            print('AOI File Not Found')
            print('Please Ensure the File: /home/greg/Documents/ALOS_ENV/AOI/South_America_AOI.geojson is In the ALOS_ENV/AOI/ Folder, mounted in the Docker Image Data Location')

        if os.path.exists('/data/ALOS_ENV/Dryland_Forest_Mask/DrylandForest-Mask_bin_WC-RWLv1.tiff'):
            print('Ref Data Found')
            ref = '/data/ALOS_ENV/Dryland_Forest_Mask/DrylandForest-Mask_bin_WC-RWLv1.tiff'
        else:
            print('Ref Dataset Not Found')

        cmd = 'python applyXGBoostClassificationImgs_Bins_L2-2_AWS.py -hh {0} -hv {1} -lin {2} -j 1 -hd {3} -sl {4} -skw {5} -skf {6} -cmw {7} -cmf {8} -lc {9} -v {10} -lnc {11} -mnc {12} -os {13} -ldt {14} -mdt {15} -c {16} -rsp {17} -ref {18}\n'.format(hh,hv,lin,hand,slope,scalerWater,scalerFlood,waterModel,floodModel,lcc,aoi,lowBackscatterNumClumps,mainBackscatterNumClumps,obSize,lowBackscatterDT,mainBackscatterDT,orbitCycle,rsp,ref)
        listCmds.append(cmd)

    os.chdir(cwd)
os.chdir(cwd)

with open('classJob.sh','w') as f:
    for cmd in listCmds:
        f.write(cmd)
