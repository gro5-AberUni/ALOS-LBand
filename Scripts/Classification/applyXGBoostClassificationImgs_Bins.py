import os
import pandas as pd
import numpy as np
import xgboost as xgb
import rios
from rios import rat
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
from sklearn import preprocessing
import rsgislib
from rsgislib import imageutils
from rsgislib import rastergis
from rsgislib.segmentation import shepherdseg
from rsgislib import imagefilter
from pickle import load
import argparse
import shutil
import glob
import multiprocessing
from numba import jit
import math
import gc
import subprocess
import zipfile
import sys

def generateSubArrays(mainArray,windowSize):
    h, w = mainArray.shape

    return mainArray.reshape(h // windowSize, windowSize, -1, windowSize).swapaxes(1, 2).reshape(-1, windowSize, windowSize)

@jit(nopython=True)
def get_bin_edges(a, bins):
    bin_edges = np.zeros((bins+1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges

@jit(nopython=True)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin

@jit(nopython=True)
def numba_histogram(a, bins):
    hist = np.zeros((bins,), dtype=np.intp)
    bin_edges = get_bin_edges(a, bins)

    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist, bin_edges

@jit(nopython=True)
def calculateCVR(subArrays,globalMean):
    listStats = []
    for array in subArrays:
        try:
            cv = np.nanstd(array)/np.nanmean(array)
            rval = np.nanmean(array)/globalMean
            listStats.append([cv,rval])
        except:
            listStats.append([np.nan, np.nan])
    return listStats

@jit(nopython=True)
def getThresholds(subArrays,meanCV,meanR,std2CV,std2R,listStats):

    listThresholds = []

    counter = 0
    for array in subArrays:
        if np.isnan(listStats[counter][0]) != True:
            cv = listStats[counter][0]
            rval = listStats[counter][1]

            meanDist = math.sqrt((cv - meanCV)**2+(rval - meanR)**2)
            # stdDist = math.sqrt(((cv - std2CV) **2) + ((rval - std2R) **2))
            meanStdDist = math.sqrt(((meanCV - std2CV) **2) + ((meanR - std2R) **2))

            if meanDist > meanStdDist:
                ## Variable Sub Tile ##

                ## Get Pixel Values, Otsu and get Thresholds ##

                pixelImg = array.ravel()

                pixelImg = pixelImg[~np.isnan(pixelImg)]

                hist, bin_edges = numba_histogram(pixelImg, 300)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

                # class probabilities for all possible thresholds
                weight1 = np.cumsum(hist)
                weight2 = np.cumsum(hist[::-1])[::-1]
                # class means for all possible thresholds
                mean1 = np.cumsum(hist * bin_centers) / weight1
                mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

                # Clip ends to align class 1 and class 2 variables:
                # The last value of ``weight1``/``mean1`` should pair with zero values in
                # ``weight2``/``mean2``, which do not exist.
                variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

                idx = np.argmax(variance12)
                otsuThreshold = bin_centers[:-1][idx]

            listThresholds.append(otsuThreshold)

        counter += 1

    return listThresholds

def autoThresholdWrapper(s1Array,windowSize,globalImgMean):

    ## Moving Window over image to calculate cv and r for each window ##

    ## cv = Window Std Dev / Sample Mean ##
    ## rVal = Sample Mean / Global Mean

    imgWidth = s1Array.shape[0]

    nTiles = imgWidth//windowSize

    tiledWidthPxls = nTiles*windowSize

    imgHeight = s1Array.shape[1]

    nTiles = imgHeight//windowSize

    tiledHeightPxls = nTiles*windowSize

    s1ArraySub = np.ascontiguousarray(s1Array[0:tiledWidthPxls,0:tiledHeightPxls])

    del s1Array

    # print("Generating Sub Arrays...")

    tiledArrays = generateSubArrays(s1ArraySub,windowSize)

    # print("Sub Arrays Found")
    del s1ArraySub

    # print("Calculating Tile Stats...")

    listStats = calculateCVR(tiledArrays,globalImgMean)

    # print("Tile Stats Found")

    listCV = []
    listR = []

    for values in listStats:
        if np.isnan(values[0]) != True:
            listCV.append(values[0])
            listR.append(values[1])


    meanCV = np.nanmean(listCV)
    meanR = np.nanmean(listR)

    std2CV = (np.std(listCV)) * 3
    std2R = (np.std(listR)) * 3

    # print('meanCV: ',meanCV)
    # print('meanR: ',meanR)
    # print('std2CV: ',std2CV)
    # print('std2R: ',std2R)

    thresholds = getThresholds(tiledArrays,meanCV,meanR,std2CV,std2R,np.array(listStats))
    del listStats
    del tiledArrays

    return thresholds

def gdalSave(refimg,listOutArray,outputfile,form):

    # print('Ref Image: ',refimg)

    ds = gdal.Open(refimg)
    refArray = (np.array(ds.GetRasterBand(1).ReadAsArray()))
    refimg = ds
    arrayshape = refArray.shape
    x_pixels = arrayshape[1]
    y_pixels = arrayshape[0]
    # print(x_pixels,y_pixels)
    GeoT = refimg.GetGeoTransform()
    Projection = osr.SpatialReference()
    Projection.ImportFromWkt(refimg.GetProjectionRef())
    driver = gdal.GetDriverByName(form)
    numBands = len(listOutArray)
    # print(numBands)
    dataset = driver.Create(outputfile, x_pixels, y_pixels, numBands, gdal.GDT_Float32)
    dataset.SetGeoTransform(GeoT)
    dataset.SetProjection(Projection.ExportToWkt())
    counter = 1
    for array in listOutArray:
        dataset.GetRasterBand(counter).WriteArray(array)
        counter+=1
    dataset.FlushCache()
    del listOutArray
    del refArray
    gc.collect()


def extractTraining(args):
    print(args)

    clumpsImg = args[0]

    if clumpsImg != None:

        ## HH ##

        bs = []
        bs.append(rastergis.BandAttStats(band=1, mean_field='HHMean',std_dev_field='HHstdDev'))
        rastergis.populate_rat_with_stats(args[1], clumpsImg, bs)

        ## HV ##

        bs = []
        bs.append(rastergis.BandAttStats(band=2, mean_field='HVMean', std_dev_field='HVstdDev'))
        rastergis.populate_rat_with_stats(args[1], clumpsImg, bs)

        ## Cross Pol ##

        bs = []
        bs.append(rastergis.BandAttStats(band=3, mean_field='NDPIMean', std_dev_field='NDPIstdDev'))
        rastergis.populate_rat_with_stats(args[1], clumpsImg, bs)

        ## Inc Angle ##

        bs = []
        bs.append(rastergis.BandAttStats(band=4, mean_field='IncMean', std_dev_field='IncstdDev'))
        rastergis.populate_rat_with_stats(args[1], clumpsImg, bs)


        ## Slope ##

        bs = []
        bs.append(rastergis.BandAttStats(band=1, mean_field='SlopeMean', std_dev_field='SlopestdDev'))
        rastergis.populate_rat_with_stats(args[2], clumpsImg, bs)

        ## Hand ##

        bs = []
        bs.append(rastergis.BandAttStats(band=1, mean_field='HandMean', std_dev_field='HandstdDev'))
        rastergis.populate_rat_with_stats(args[3], clumpsImg, bs)


    return clumpsImg

def segment(args):

    clumpsImg = None

    inImg = args[0]
    clusters = args[1]
    obSize = args[2]
    dt = args[3]
    stretch = args[4]
    samp = args[5]
    print()
    print("Segmenting: ",inImg)
    print()



    rsgislib.imageutils.pop_img_stats(inImg, use_no_data=True, no_data_val=0)
    clumpsImgTemp = inImg.replace('.tif','_Clumps.kea')
    shepherdseg.run_shepherd_segmentation(inImg,
                                     clumpsImgTemp,
                                     tmp_dir='./',
                                     no_stretch = stretch,
                                     num_clusters=clusters,
                                     min_n_pxls=obSize,
                                     dist_thres=dt, bands=[1,2,3],
                                     sampling=samp, km_max_iter=50,
                                     process_in_mem=True)

    # tmpDir = inImg.replace('.tif','_tmpDir')

    # tiledsegsingle.perform_tiled_segmentation(inImg,
    #                                  clumpsImgTemp,
    #                                  tmp_dir='./',
    #                                  tile_width=2250,
    #                                  tile_height=2250,
    #                                  num_clusters=clusters,
    #                                  min_pxls=obSize,
    #                                  dist_thres=dt,
    #                                  sampling=1, km_max_iter=2000)


    gc.collect()

    clumpsImgTempRe = clumpsImgTemp.replace('.kea','_Re.kea')

    # rsgislib.segmentation.rm_small_clumps(clumpsImgTemp, clumpsImgTempRe, obSize, 'KEA')

    #rsgislib.segmentation.rm_small_clumps_stepwise(inImg, clumpsImgTemp, clumpsImgTempRe, 'KEA', False,'', False, True, obSize,100)

    rsgislib.rastergis.pop_rat_img_stats(clumpsImgTemp, True, True,True, 1)


    clumpsImg = clumpsImgTemp

    return clumpsImg

def resample(args):

    processingImg= args[0]
    refImg = args[1]
    resampImg = args[2]

    imageutils.resample_img_to_match(refImg, processingImg, resampImg, 'GTIFF', 2, 9,False)

    return resampImg

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Processing of Sentinel 1 images')

    parser.add_argument('-i',metavar='',type=str,help='Input ALOS PALSAR Image to be classified')
    parser.add_argument('-j',metavar='',type=int,help='Maximum Number of threads to use. Default is 2.',default=2)
    parser.add_argument('-sl', metavar='', type=str, help='Slope')
    parser.add_argument('-hd', metavar='', type=str, help='HAND')
    parser.add_argument('-skw', metavar='', type=str, help='Classification Scaler Water')
    parser.add_argument('-skf', metavar='', type=str, help='Classification Scaler Flood')
    parser.add_argument('-cmw', metavar='', type=str, help='Classification Model Water')
    parser.add_argument('-cmf', metavar='', type=str, help='Classification Model Flood')
    parser.add_argument('-lc', metavar='', type=str, help='Landcover')
    parser.add_argument('-v', metavar='', type=str, help='Basin AOI')

    parser.add_argument('-lnc', metavar='', type=int, help='Low Backscatter Num Clusters. Default is 20', default=20)
    parser.add_argument('-mnc', metavar='', type=int, help='Main Backscatter Num Clusters. Default is 250', default=250)

    parser.add_argument('-bnc', metavar='', type=int, help='Main Backscatter Backup Num Clusters. Default is 50', default=50)
    
    parser.add_argument('-ldt', metavar='', type=int, help='Low Backscatter Distance Threshold. Default is 10', default=10)
    parser.add_argument('-mdt', metavar='', type=int, help='Main Backscatter Distance Threshold. Default is 10', default=10)
    
    parser.add_argument('-os', metavar='', type=int, help='Segmentation Object Size. Default is 5', default=5)


    args = parser.parse_args()

    cwd = "/data"#os.getcwd()

    cores = args.j

    scalerPKLWater = args.skw
    scalerPKLFlood = args.skf

    modelCMW = args.cmw
    modelCMF = args.cmf

    backupClusterCentres = args.bnc

    obSize = args.os

    lowBackscatterNumClumps = args.lnc
    mainBackscatterNumClumps = args.mnc

    lowBackscatterDist = args.ldt
    mainBackscatterDist = args.mdt
    
    form = 'KEA'
    dtype = rsgislib.TYPE_32FLOAT

    alosEpoch = args.i.split('/')[-1]
    print(alosEpoch)
    epochDate = alosEpoch.split('_')[3]
    print(epochDate)
    filt=alosEpoch.split('_')[-1].split('.')[0]
    print(filt)

    tileLoc = alosEpoch.split('_')[0]

    epochDate = epochDate+'-'+tileLoc
    print(epochDate)

    orbitCycle = alosEpoch.split('_')[1]

    print(orbitCycle)

    if os.path.exists('{0}/ALOS-processing_{1}/'.format(cwd,epochDate)) == True:
        shutil.rmtree('{0}/ALOS-processing_{1}/'.format(cwd,epochDate))
        print('Processing Deleted')
        os.mkdir('{0}/ALOS-processing_{1}/'.format(cwd,epochDate))
    else:
        os.mkdir('{0}/ALOS-processing_{1}/'.format(cwd,epochDate))

    if os.path.exists('{0}/ALOS-Output_{1}-{2}/'.format(cwd,epochDate,orbitCycle)) == True:
        shutil.rmtree('{0}/ALOS-Output_{1}-{2}/'.format(cwd,epochDate,orbitCycle))
        os.mkdir('{0}/ALOS-Output_{1}-{2}/'.format(cwd,epochDate,orbitCycle))
    else:
        os.mkdir('{0}/ALOS-Output_{1}-{2}/'.format(cwd,epochDate,orbitCycle))

    workspace = '{0}/ALOS-processing_{1}/'.format(cwd,epochDate)

    outputDir = '{0}/ALOS-Output_{1}-{2}'.format(cwd,epochDate,orbitCycle)

    shutil.copy(args.i, workspace)

    os.chdir(workspace)

    #### Process Zip ####

    if '.zip' in args.i:
        print('Zip File')

        zipFN = glob.glob('*.zip')[0]

        with zipfile.ZipFile(zipFN, 'r') as zip_ref:
            targetDir = zipFN.replace('.zip', '')

            print(targetDir)

            zip_ref.extractall(targetDir)

        zipDir = targetDir

        os.chdir(zipDir)

        hhImg = glob.glob('*HH.tif')[0]
        hvImg = glob.glob('*HV.tif')[0]

        inc = glob.glob('*linci*')[0]

        ds = gdal.Open(inc)
        incArray = (np.array(ds.GetRasterBand(1).ReadAsArray()))
        del ds

        ds = gdal.Open(hhImg)
        hhArray = (np.array(ds.GetRasterBand(1).ReadAsArray()))
        del ds

        hhArrayNanCor = np.where(hhArray == 0, np.nan, hhArray)
        hhArraydB = (10 * np.log10(hhArrayNanCor * hhArrayNanCor)) - 83
        hhArraydBNanCor = np.where(hhArraydB == -83, np.nan, hhArraydB)

        ds = gdal.Open(hvImg)
        hvArray = (np.array(ds.GetRasterBand(1).ReadAsArray()))
        del ds

        hvArrayNanCor = np.where(hvArray == 0, np.nan, hvArray)
        hvArraydB = (10 * np.log10(hvArrayNanCor * hvArrayNanCor)) - 83
        hvArraydBNanCor = np.where(hvArraydB == -83, np.nan, hvArraydB)

        ndpiArray = (hhArraydBNanCor - hvArraydBNanCor) / (hhArraydBNanCor + hvArraydBNanCor)

        outputTile = zipDir + '.tif'

        gdalSave(hhImg, [hhArraydBNanCor, hvArraydBNanCor, ndpiArray], outputTile, 'GTIFF')

        windowSize = 3

        filtImg = outputTile.replace('.tif', '_Filt-{0}-Window.tif'.format(windowSize))

        rsgislib.imagefilter.apply_lee_filter(outputTile, filtImg, windowSize, windowSize, 'GTIFF',
                                              rsgislib.TYPE_32FLOAT)

        incImg = filtImg.replace('.tif', '_Inc.kea')

        rsgislib.imageutils.stack_img_bands([filtImg, inc], None, incImg, 0, 0, 'KEA', rsgislib.TYPE_32FLOAT)

        shutil.copy(incImg,'..')

        os.chdir(workspace)

        alosEpoch = incImg

    else:
        alosEpoch = glob.glob('*.kea')[0]

    #### Check that Raster Tile Overlaps AOI ####

    aoi = args.v

    raster = gdal.Open(alosEpoch)
    vector = ogr.Open(aoi)

    # Get raster geometry
    transform = raster.GetGeoTransform()
    pixelWidth = transform[1]
    pixelHeight = transform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    xLeft = transform[0]
    yTop = transform[3]
    xRight = xLeft + cols * pixelWidth
    yBottom = yTop - rows * pixelHeight

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(xLeft, yTop)
    ring.AddPoint(xLeft, yBottom)
    ring.AddPoint(xRight, yTop)
    ring.AddPoint(xRight, yBottom)
    ring.AddPoint(xLeft, yTop)
    rasterGeometry = ogr.Geometry(ogr.wkbPolygon)
    rasterGeometry.AddGeometry(ring)

    layer = vector.GetLayer()
    feature = layer.GetFeature(0)
    vectorGeometry = feature.GetGeometryRef()

    # if rasterGeometry.Intersect(vectorGeometry) == False:
    #     shutil.rmtree(workspace)
    #     shutil.rmtree(outputDir)
    #     sys.exit()
    #
    resampCmds = []

    #### Gather Ancillary Datasets to workspace ####

    ## Slope ##

    slope = args.sl

    slopeRe = slope.split('/')[-1].replace('.tif', '_Re.tif')

    cmd = [slope, alosEpoch, slopeRe]
    resampCmds.append(cmd)

    ## Hand ##

    hand = args.hd

    handRe = hand.split('/')[-1].replace('.tif', '_Re.tif')

    cmd = [hand, alosEpoch, handRe]
    resampCmds.append(cmd)

    ## GlobCover ##

    lccMask = args.lc

    lccMaskRe = lccMask.split('/')[-1].replace('.tif', '_Re.tif')

    cmd = [lccMask, alosEpoch, lccMaskRe]
    resampCmds.append(cmd)

    print(resampCmds)

    print(cores)

    if cores == 1:
        results = []

        for cmd in resampCmds:
            results.append(resample(cmd))

    else:

        pool = multiprocessing.Pool(cores)

        results = pool.map(resample, resampCmds)

        pool.close()

        pool.join()

    # print()

    slopeRe = results[0]
    hand = results[1]
    lccMaskRe = results[2]

    ds = gdal.Open(slopeRe)
    slopeArr = (np.array(ds.GetRasterBand(1).ReadAsArray()))
    ds = None

    slopeArrLow = np.where(slopeArr>10,1,0)

    slopeMask = 'SlopeMask.tif'

    gdalSave(slopeRe,[slopeArrLow],slopeMask,'GTIFF')

    #### Apply LCC ####

    alosEpochLCCM = alosEpoch.replace('.kea','_LCC.tif')
    rsgislib.imageutils.mask_img(alosEpoch, lccMaskRe, alosEpochLCCM, 'GTIFF', rsgislib.TYPE_32FLOAT, 0, 1)

    alosEpochSlopeM = alosEpoch.replace('.kea','_SLopeM.tif')
    rsgislib.imageutils.mask_img(alosEpochLCCM, slopeMask, alosEpochSlopeM, 'GTIFF', rsgislib.TYPE_32FLOAT, 0, 1)

    alosEpoch = alosEpochSlopeM

    shutil.copy(alosEpoch,outputDir)

    alosEpochVM = alosEpochLCCM.replace('.tif','_Valid_Mask.tif')

    rsgislib.imageutils.gen_finite_mask(alosEpochLCCM, alosEpochVM,'GTIFF')

    ## Begin SBT for HV ##

    ds = gdal.Open(alosEpoch)
    hhArray = (np.array(ds.GetRasterBand(1).ReadAsArray()))
    ds = None

    ds = gdal.Open(alosEpoch)
    hvArray = (np.array(ds.GetRasterBand(2).ReadAsArray()))
    ds = None

    ds = gdal.Open(alosEpoch)
    ndpiArray = (np.array(ds.GetRasterBand(3).ReadAsArray()))
    ds = None

    lowBackscatterHV_HH = np.where(hvArray < -15, hhArray, 0)
    lowBackscatterHV_HV = np.where(hvArray < -15, hvArray, 0)
    lowBackscatterHV_NDPI = np.where(hvArray < -15, ndpiArray, 0)

    lowBSImgHV = 'Low_Backscatter_Image_HV-Based.tif'

    gdalSave(alosEpoch, [lowBackscatterHV_HH,lowBackscatterHV_HV,lowBackscatterHV_NDPI], lowBSImgHV, 'GTIFF')

    waterValid = 0

    try:

        segmentedImageLowBS = segment([lowBSImgHV, lowBackscatterNumClumps, obSize, lowBackscatterDist, False,1])
        waterValid = 1

    except:

        errFileName = '/data/{0}'.format(alosEpoch.split('/')[-1].replace('.tif','_Warn.txt'))
        print(errFileName)

        ds = gdal.Open(lowBSImgHV)
        readLow = np.array(ds.GetRasterBand(1).ReadAsArray())
        uniquePxls = np.unique(readLow)

        with open(errFileName,'w') as f:
            f.write('There has been an error processing this tile in the segmentation phase to find areas of low backscater.\nUnique Pixels: {0}\nNum Clusters: {1}\nDistanceThreshold: {2}\nImage Segmentation is performed on the whole input PALSAR tile.'.format(len(uniquePxls),lowBackscatterNumClumps,lowBackscatterDist))

    try:
        segmentedImageALOS = segment([alosEpoch, mainBackscatterNumClumps, obSize, mainBackscatterDist, False,100])

    except:
        try:
            segmentedImageALOS = segment([alosEpoch, mainBackscatterNumClumps, obSize, mainBackscatterDist, False,1])
        except:
            try:
                segmentedImageALOS = segment([alosEpoch, backupClusterCentres, obSize, mainBackscatterDist, False,1])
                errFileName ='/data/{0}'.format(alosEpoch.split('/')[-1].replace('.tif','_Warn.txt'))
                with open(errFileName,'w') as f:
                    f.write('Image Segmentation is performed on the whole input PALSAR tile, with: {0} cluster centres.'.format(backupClusterCentres))
            except:
                errFileName ='/data/{0}'.format(alosEpoch.split('/')[-1].replace('.tif','_Error.txt'))
                with open(errFileName,'w') as f:
                    f.write('There has been an error processing this tile in the segmentation phase - Processing Could not be performed, even with no pixel sampling')
                shutil.rmtree(workspace)
                sys.exit()
    # except:

    #     errFileName ='/data/{0}'.format(alosEpoch.split('/')[-1].replace('.tif','_Error.txt'))
    #     with open(errFileName,'w') as f:
    #         f.write('There has been an error processing this tile in the segmentation phase')
    #     shutil.rmtree(workspace)
    #     sys.exit()
        

    if waterValid == 1:

        alosEpochMrg = segmentedImageALOS.replace('.kea','Merge_Clumps.kea')

        rsgislib.segmentation.union_of_clumps([segmentedImageLowBS,segmentedImageALOS], alosEpochMrg, 'KEA', 0, True)

        rsgislib.rastergis.pop_rat_img_stats(alosEpochMrg, True, True,True, 1)
    else:
        alosEpochMrg = segmentedImageALOS

    #alosEpochRe = alosEpochMrg.replace('.kea', '_Re.kea')
    alosEpochRe = alosEpochMrg
    #rsgislib.segmentation.rm_small_clumps_stepwise(alosEpoch, alosEpochMrg, alosEpochRe, 'KEA', False,'', False, True, 10,100000000000000)

    rsgislib.rastergis.pop_rat_img_stats(alosEpochRe, True, True,True, 1)

    segmentedImage=alosEpochRe

    try:

        alosEpoch = extractTraining([segmentedImage, alosEpoch, slopeRe, hand])

        #### Classify Water ####

        listVars = ['HHMean','HVMean','NDPIMean','NDPIstdDev','IncstdDev','SlopeMean','SlopestdDev','HandstdDev']

        print(listVars)


        predictData = pd.DataFrame()

        for var in listVars:
            print(var)
            print(alosEpoch)
            predictData[var] = rios.rat.readColumn(alosEpoch, var, bandNumber=1)

        # scaler = preprocessing.StandardScaler().fit(predictData)

        print(predictData)

        scaler = load(open(scalerPKLWater, 'rb'))

        predictData_scaled = scaler.transform(predictData)

        trainedXGBoostModel = xgb.Booster()

        trainedXGBoostModel.load_model(modelCMW)

        predictDataDMatrix = xgb.DMatrix(predictData_scaled)

        classPrediction = trainedXGBoostModel.predict(predictDataDMatrix)

        classPredictionReclass = np.where(classPrediction==1,2,0)

        classifiedImageWater = 'Classified_Output_Water.tif'

        rios.rat.writeColumn(alosEpoch, 'ClassOutputWater', classPredictionReclass)

        rsgislib.rastergis.export_col_to_gdal_img(alosEpoch, classifiedImageWater, 'GTIFF', rsgislib.TYPE_8INT, 'ClassOutputWater',
                                                  rat_band=1)


        #### Classify Flooded Forest ####

        listVars = ['HHMean','HVMean','NDPIMean','IncMean']

        print(listVars)

        predictData = pd.DataFrame()

        for var in listVars:
            print(var)
            print(alosEpoch)
            predictData[var] = rios.rat.readColumn(alosEpoch, var, bandNumber=1)

        # scaler = preprocessing.StandardScaler().fit(predictData)

        scaler = load(open(scalerPKLFlood, 'rb'))

        predictData_scaled = scaler.transform(predictData)

        trainedXGBoostModel = xgb.Booster()

        trainedXGBoostModel.load_model(modelCMF)

        predictDataDMatrix = xgb.DMatrix(predictData_scaled)

        classPrediction = trainedXGBoostModel.predict(predictDataDMatrix)

    #    classPredictionReclass = np.where(classPrediction==0,4,classPrediction)

        classifiedImageFlood = 'Classified_Output_Flooded_Forest.tif'

        rios.rat.writeColumn(alosEpoch, 'ClassOutputFlood', classPrediction)

        rsgislib.rastergis.export_col_to_gdal_img(alosEpoch, classifiedImageFlood, 'GTIFF', rsgislib.TYPE_8INT, 'ClassOutputFlood',
                                                  rat_band=1)

        #### Combine Class Outputs ####

        ds = gdal.Open(classifiedImageWater)
        waterArr = np.array(ds.GetRasterBand(1).ReadAsArray())
        del ds

        ds = gdal.Open(classifiedImageFlood)
        floodArr = np.array(ds.GetRasterBand(1).ReadAsArray())
        del ds

        ds = gdal.Open(lccMaskRe)
        lccArr = np.array(ds.GetRasterBand(1).ReadAsArray())
        del ds


        combArr = waterArr+floodArr

        gdalSave(classifiedImageWater,[combArr],'CombinedArr.tif','GTIFF')

        classOutputArr = np.where(lccArr==0,np.where(waterArr+floodArr==0,4,combArr),0)

        classOutputArrCor = np.where(classOutputArr==3,4,classOutputArr)

        classCombinedOutput = 'CombinedClass.tif'

        gdalSave(classifiedImageWater,[classOutputArrCor],classCombinedOutput,'GTIFF')

        classifiedImageVM = '{0}_Classified.tif'.format(alosEpoch.replace('.kea',''))
        rsgislib.imageutils.mask_img(classCombinedOutput, alosEpochVM, classifiedImageVM, 'GTIFF', 1, 0, 0)

        classifiedImageFilt = classifiedImageVM.replace('.tif','_Filt.tif')
        imagefilter.apply_mode_filter(classifiedImageVM, classifiedImageFilt, 3, "GTIFF",1)

        classifiedImageSieve = classifiedImageFilt.replace('.tif','_Sieve.tif')
        cmd = 'gdal_sieve.py -st 50 -8 -of GTIFF {0} {1}'.format(classifiedImageFilt,classifiedImageSieve)

        subprocess.call(cmd,shell=True)

        classifiedImageSieveLCCM = classifiedImageSieve.replace('.tif', '_LCC.tif')
        rsgislib.imageutils.mask_img(classifiedImageSieve, lccMaskRe, classifiedImageSieveLCCM, 'GTIFF', 1, 5, 1)

        classifiedImageSieveSlopeM = classifiedImageSieveLCCM.replace('.tif', '_SLopeM.tif')
        rsgislib.imageutils.mask_img(classifiedImageSieveLCCM, slopeMask, classifiedImageSieveSlopeM, 'GTIFF', 1, 6, 1)

        clr_lut = dict()
        clr_lut[0] = '#000000'
        clr_lut[1] = '#6CABDD'
        clr_lut[2] = '#000080'
        clr_lut[4] = '#004225'
        clr_lut[5] = '#a7a7a7'
        clr_lut[6] = '#d21255'

        rsgislib.imageutils.define_colour_table(classifiedImageVM, clr_lut)
        rsgislib.imageutils.define_colour_table(classifiedImageFilt, clr_lut)
        rsgislib.imageutils.define_colour_table(classifiedImageSieveSlopeM, clr_lut)

        shutil.copy(classifiedImageVM, outputDir)
        shutil.copy(classifiedImageFilt,outputDir)
        shutil.copy(classifiedImageSieveSlopeM, outputDir)
        os.chdir(cwd)
        shutil.rmtree(workspace)
    except:
        print('There Has Been An Error')
        os.chdir(cwd)
        shutil.rmtree(workspace)

    sys.exit()


