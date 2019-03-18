import os
import cv2
import numpy as np
def temporalIntensity(dir):
    images = os.listdir(dir)
    I = [0 for x in xrange(len(images))]
    for image in xrange(len(images)):
        img = cv2.imread(dir + images[image])
        I[image] = img
    return I
def temporalMean(I):
    M = np.mean([I[0],I[1],I[2],I[3],I[4]],axis=0)
    return M
def temporalMedian(I):
    M = np.median([I[0],I[1],I[2],I[3],I[4]],axis=0)
    return M
def lowPassFilter(img):
    kernel = np.ones((5,5),np.float32)/25
    mask = cv2.filter2D(img,-1,kernel)
    return mask
def removeMovingObjects():
    dir = "images/set2/"
    images = os.listdir(dir)
    I = [0 for x in xrange(len(images))]
    for i in xrange(len(images)):
        img = cv2.imread(dir + images[i],cv2.IMREAD_GRAYSCALE)
        I[i] = img
        I[i] = np.asarray(I[i], dtype=np.int16)
    T = np.array(I, copy=True)
    for i in xrange(len(images)):
        T[i] = lowPassFilter(T[i])
    diff = [0 for x in xrange(len(images))]
    kernel = np.ones((5,5),np.uint8)
    # Create a mask of 4/5 images using the difference from the next frame to the current frame
    for i in xrange(0,len(images)):
        diff[i] = T[(i+2)%5]-T[i]
        # diff[i] = np.clip(diff[i], a_min=0,a_max=255)
        diff[i] = np.absolute(diff[i])
        for j in xrange(10):
            diff[i] = cv2.morphologyEx(diff[i], cv2.MORPH_OPEN, kernel)
            diff[i] = cv2.morphologyEx(diff[i], cv2.MORPH_CLOSE, kernel)
        t = diff[i] > 5
        diff[i][t] = 255
        t = diff[i] <= 5
        diff[i][t] = 0
        # cv2.imwrite("diff"+str(i)+".jpg", diff[i])
        diff[i] = np.equal(diff[i],0)
        diff[i] = diff[i].astype(int)
    # Get the image that has the largest foreground
    numero = 0
    smallest = np.inf
    for i in xrange(len(diff)):
        T[i] = np.multiply(T[i], diff[i])
        # cv2.imwrite(str(i)+".jpg", I[i])
        temp = np.equal(T[i],0)
        temp = temp.astype(int)
        temp = np.sum(temp)
        if temp < smallest:
            smallest = temp
            numero = i
    # Using the largest foreground, apply the background of all other images
    mask = T[numero]
    for i in xrange(len(I)):
        if i == numero:
            pass
        else:
            a = np.greater(T[i], 0)
            a = a.astype(int)
            b = np.equal(mask, 0)
            b = b.astype(int)
            c = np.equal(a,b)
            c = c.astype(int)
            c = np.multiply(T[i], c)
            mask = np.add(mask,c)
    mask = np.array(mask, dtype=np.uint8)
    a = np.equal(mask,0)
    a = a.astype(int)
    median = np.median([I[0],I[1],I[2],I[3],I[4]],axis=0)
    median = np.multiply(median, a)
    mask = np.add(median,mask)
    cv2.imwrite("mask.jpg", mask)
