import os
import cv2
import numpy as np
import cv2 as cv

def removeSelectedObject(image,x0,y0,x1,y1):

    #remove selected region, assume square box
    for x in range(x0,x1):
        for y in range(y0,y1):
            image[y,x] = 0

    cv.imwrite('Removed.png',image)

    

    return
