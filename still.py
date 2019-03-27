import os
import cv2
import scipy.ndimage
import numpy as np
import cv2 as cv
import types

# Patched-based inpainting
# Rajesh Pandurang Borole and Sanjiv Vedu BondePatch-Based Inpainting for Object Removal and Region Filling in Images
# DOI 10.1515/jisys-2013-0031 Journal of Intelligent Systems 2013; 22(3): 335-350

#requires binary mask for edge detect
def mapBoundry(FillRegion,image):

    laplace = cv2.Laplacian(np.uint8(FillRegion),cv2.CV_8U)
    cv2.imwrite('laplace.png',laplace)




    return laplace

def prioritySort(item):
    return item['priority']

def computePatch(BoundryRegion,InterestRegion):

    priorityList = list()

    #7x7 patch
    for pixel in BoundryRegion:
        x0 = pixel[1] - 3
        x1 = pixel[1] + 2
        y0 = pixel[0] - 3
        y1 = pixel[0] + 2


        priority = 0.0

        for x in range(x0,x1-1):
            for y in range(y0,y1-1):

                if InterestRegion[y,x,0] != 0: #FIX
                    priority = priority + 1.0

        priority = priority / 49.0 #compute C(p)
        priorityList.append({'priority':priority,'x':x,'y':y})

    priorityList.sort(reverse=True,key=prioritySort)
    return priorityList

def removeSelectedObject():

    image = cv.imread('images/still/donkey.jpg')

    #target region
    x0 = 50
    y0 = 50
    x1 = 100
    y1 = 100

    #TODO: BOUNDS CHECKING

    #denote fill region (NEEDS TO CHANGE)
    image[y0:y1,x0:x1,:] = [0,0,0]
    image[y0+7:y1-7,x0+7:x1-7,:] = [255,255,255]

    #create FillRegion list, which comprises x,y coords to be filled
    newYRange = y1-y0
    newXRange = x1-x0

    #dictItem = {'priority':1,'BGR':[float,float,float]}

    FillRegion = np.empty([newYRange+7,newXRange+7])#empty([newYRange+1,newXRange+1],dtype=[('priority','Float8'),('')])

    #denote Source Space from Fill Space
    for y in range(0,newYRange+7):
        for x in range(0,newXRange+7):

            #Source Space (OUTSIDE BOUNDING BOX)
            if (((y+newYRange) >= y1) or ((y+newYRange-7) < y0) or ((x+newXRange) >= x1) or ((x+newXRange-7) < x0)):
                FillRegion[y,x] = 1

            #Fill Region
            else:
                FillRegion[y,x] = 0
    cv.imwrite('FillRegion.png',FillRegion)

    BoundryRegion = mapBoundry(FillRegion,image)

    for y in range(0,newYRange+7):
        for x in range(0,newXRange+7):

            #Source Space (OUTSIDE BOUNDING BOX)
            if BoundryRegion[y,x] > 0:
                image[y+newYRange,x+newXRange,:] = [255,255,255]

    cv.imwrite('Boundry.png',BoundryRegion)

    #print BoundryRegion
    #compute pixel patch priority
    #priorityList = computePatch(BoundryRegion,image)
    #print priorityList

    #pixel = priorityList.pop(1) #highest priority
    #print pixel
    #pixelX = pixel['x']
    #pixelY = pixel['y']
    #print pixelX
    #print pixelY

    #image[pixelY-3:pixelY+3,pixelX-3:pixelX+3,:] = [255,255,255]

    """
    fillList = list();

    #initial priority list. 1=Source, 0=Fill Region
    for x in range(x0-7,x1+7):
        for y in range(y0-7,y1+7):
            if ((x < x0 or x > x1) and (y < y0 or y > y1)):
                fillList.append({'coordinate':(x,y),'priority':1})
            else:
                fillList.append({'coordinate':(x,y),'priority':0})

    print fillList
    """
    cv.imwrite("img.jpg",image)

    return

removeSelectedObject()
