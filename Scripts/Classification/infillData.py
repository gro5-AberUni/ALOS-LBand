import glob
import numpy as np
from osgeo import gdal
from osgeo import osr
import gc
import datetime
import rsgislib
from numba import jit
from rsgislib import imageutils

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
    dataset = driver.Create(outputfile, x_pixels, y_pixels, numBands, gdal.GDT_Byte)
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

@jit(nopython=True)
def findEqual(curr,prev,next,change):
	equal = np.equal(prev,next)
	resultMeta = np.where(curr != 0, 255, np.where(equal==1,4,change))
	return resultMeta
@jit(nopython=True)
def fill(corrMetaData,currArr,prevArr,change,outArr):
	filledArray = np.where(corrMetaData == 255, currArr,
						   np.where(corrMetaData == 4, prevArr, np.where(corrMetaData == change, outArr, 0)))
	return filledArray

listFiles = glob.glob('Classified*Sub.tif')
sortListFiles = sorted(listFiles)


print(sortListFiles)

#### Sort Out Draw up Draw Down Range ####

minInunMnth = 11
minInunDay = 6

maxInunMnth = 4
maxInunDay = 25

counter = 0 
print()



for classImg in sortListFiles:

	if counter != 0 and counter <len(sortListFiles)-1:


		filledImage = classImg.replace('.tif','_Data-Fill.kea')

		orbitCycle = classImg.split('_')[3][:3]
		print(orbitCycle)

		prevImg = sortListFiles[counter-1]
		currentImg = classImg
		nextImg = sortListFiles[counter+1]

		#### Draw Up/Draw Down Chnage Char ####

		currDaySt = classImg.split('_')[3][10:][:2]
		currMnthSt = classImg.split('_')[3][10:][3:5]
		currYrSt = int('20'+classImg.split('_')[3][10:][6:8])

		currDayEd = classImg.split('_')[4][:2]
		currMnthEd = classImg.split('_')[4][3:5]
		currYrEd = '20'+classImg.split('_')[4][6:8]

		currStartDate = datetime.date(year=int(currYrSt), month=int(currMnthSt), day=int(currDaySt)).toordinal()
		currEndDate = datetime.date(year=int(currYrEd), month=int(currMnthEd), day=int(currDayEd)).toordinal()

		currMidDateOrd  = int(currEndDate-currStartDate)+currStartDate

		yearPeak = datetime.date(year=int(currYrSt), month=int(maxInunMnth), day=int(maxInunDay)).toordinal()

		yearMin = datetime.date(year=int(currYrSt), month=int(minInunMnth), day=int(minInunDay)).toordinal()

		if currMidDateOrd<yearPeak:
			changeVal = 3
		elif currMidDateOrd<yearMin:
			changeVal = 2
		elif currMidDateOrd>yearMin:
			changeVal = 3

		print('{0} Draw Up/Down Stage = {1}'.format(classImg,changeVal))
		print('Next Img: ', nextImg)
		print('Prev Img: ',prevImg)


		#### Read Imgs as Array ####
		ds = gdal.Open(currentImg)
		currArr = np.array(ds.GetRasterBand(1).ReadAsArray())

		print(currArr.shape)

		ds = gdal.Open(prevImg)
		prevArr = np.array(ds.GetRasterBand(1).ReadAsArray())

		print(prevArr.shape)

		ds = gdal.Open(nextImg)
		nextArr = np.array(ds.GetRasterBand(1).ReadAsArray())

		print(nextArr.shape)

		#np.where(nextArr == prevArr, 4, changeVal)

		corrMetaData = findEqual(currArr,prevArr,nextArr,changeVal)




		#equalArr = np.where(prevArr==nextArr,1,0)
		#print(equalArr.shape)
		#print(diff)
		#print(diff.shape)

		gdalSave(currentImg,[corrMetaData],'Meta_Data_Test_{0}.tif'.format(orbitCycle),'GTIFF')



		if changeVal ==2:
			gc.collect()
			#fill(corrMetaData, currArr, prevArr, change, outArr)
			filledArray = fill(corrMetaData,currArr,prevArr,changeVal,nextArr)
			#filledArray = np.where(corrMetaData==255,currArr,np.where(corrMetaData==4,prevArr,np.where(corrMetaData==2,nextArr,0)))
			del prevArr
			del nextArr
			del corrMetaData
			gc.collect()
		else:

			gc.collect()
			filledArray = fill(corrMetaData, currArr, prevArr, changeVal, prevArr)
			#filledArray = np.where(corrMetaData==255,currArr,np.where(corrMetaData==4,prevArr,np.where(corrMetaData==3,prevArr,0)))
			del nextArr
			del prevArr
			del corrMetaData
			gc.collect()
		
		gdalSave(currentImg, [filledArray], filledImage, 'KEA')
		del filledArray
		gc.collect() 
		clr_lut = dict()
		clr_lut[0] = '#000000'
		clr_lut[1] = '#6CABDD'
		clr_lut[2] = '#000080'
		clr_lut[4] = '#004225'
		clr_lut[5] = '#a7a7a7'
		clr_lut[6] = '#d21255'

		rsgislib.imageutils.define_colour_table(filledImage, clr_lut)

	else:
		print('Image 0')




	counter+=1
