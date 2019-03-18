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
