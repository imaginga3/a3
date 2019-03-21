import os
import cv2
import numpy as np
import cv2 as cv

<<<<<<< HEAD
def prioritySort(item):
    return item['priority']

def computePatch(BoundryRegion,InterestRegion):

    priorityList = list()

    #7x7 patch
    for pixel in BoundryRegion:
        x0 = pixel[1] - 3
        x1 = pixel[1] + 3
        y0 = pixel[0] - 3
        y1 = pixel[0] + 3

        priority = 0.0

        for x in range(x0,x1):
            for y in range(y0,y1):

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

    image[y0:y1,x0:x1,:] = [0,0,0]  #denote fill region
    #InterestRegion = image[x0-7:x1+7,y0-7:y1+7,:] #denote region of interest including fill region
    #BoundryRegion = np.array()
    BoundryRegion = np.vstack([image[y0:y1,x0,:],image[y0:y1,x1,:],image[y0,x0:x1,:],image[y1,x0:x1,:]])

    #compute pixel patch priority
    priorityList = computePatch(BoundryRegion,image)
    print priorityList

    pixel = priorityList.pop(1) #highest priority
    pixelX = pixel['x']
    pixelY = pixel['y']

    image[pixelY-3:pixelY+3,pixelX-3:pixelX+3,:] = [255,255,255]

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
=======
def removeSelectedObject(image,x0,y0,x1,y1):

    #remove selected region, assume square box
    for x in range(x0,x1):
        for y in range(y0,y1):
            image[y,x] = 0
>>>>>>> f38c22fef6acf844f11b0852532d58cda5d3db08

    cv.imwrite('Removed.png',image)

    # Patched-based inpainting
    # Rajesh Pandurang Borole and Sanjiv Vedu BondePatch-Based Inpainting for Object Removal and Region Filling in Images
    # DOI 10.1515/jisys-2013-0031  Journal of Intelligent Systems 2013; 22(3): 335–350

    """
    create list of target area pixels

    iterate until no target pixels remain
        assign pixel priorities
            compute pixel confidence within patch
            compute isophote priorty?
            compute pixel priority
        sort list by priority
        inpaint highest priorty pixel
    """
    return

removeSelectedObject()
