import os
import glob
from osgeo import ogr
from osgeo import gdal


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

    hand = os.path.abspath('./HydroData/Hand_MERIT-Amazon_COG.tif')
    slope = os.path.abspath('./HydroData/Slope-Amazon_COG.tif')

    scalerWater = os.path.abspath('./Model/Scaler_Water.pkl')
    scalerFlood = os.path.abspath('./Model/Scaler_Flood.pkl')
    waterModel = os.path.abspath('./Model/Trained_XGBoostModel_02_Amazon_Water.model')
    floodModel = os.path.abspath('./Model/Trained_XGBoostModel_02_Amazon_Flood.model')
    lcc = os.path.abspath('./LCC/Combined_LCC-GFC_LY-16_Binary_COG.tif')
    aoi = os.path.abspath('./AOI/Amazon_Basin.geojson')
    
    cmd = 'python3 applyXGBoostClassificationImgs_Bins.py -i {0} -j 1 -hd {1} -sl {2} -skw {3} -skf {4} -cmw {5} -cmf {6} -lc {7} -v {8}\n'.format(compFile,hand,slope,scalerWater,scalerFlood,waterModel,floodModel,lcc,aoi)
    listCmds.append(cmd)

    os.chdir(cwd)
os.chdir(cwd)

with open('classJob.sh','w') as f:
    for cmd in listCmds:
        f.write(cmd)
