import os
import cv2
import imutils
import numpy as np
from sklearn.cluster import KMeans
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
def edgeDetection(img):
    # img = cv2.imread('messi5.jpg',0)
    edges = cv2.Canny(img,100,200)
    return edges
def oldthreshold(T):
    diff = [0 for x in xrange(len(T))]
    kernel = np.ones((5,5),np.float32)/25
    for i in xrange(0,len(T)):
        diff[i] = T[(i+2)%5]-T[i]
        diff[i] = np.absolute(diff[i])
        for j in xrange(10):
            diff[i] = cv2.morphologyEx(diff[i], cv2.MORPH_OPEN, kernel)
            diff[i] = cv2.morphologyEx(diff[i], cv2.MORPH_CLOSE, kernel)
        t = diff[i] > 5
        diff[i][t] = 255
        t = diff[i] <= 5
        diff[i][t] = 0
        # cv2.imwrite("old_diff"+str(i)+".jpg", diff[i])
        diff[i] = np.equal(diff[i],0)
        diff[i] = diff[i].astype(int)
    return diff
def threshold(T):
    diff = [0 for x in xrange(len(T))]
    kernel = np.ones((5,5),np.float32)/25
    for i in xrange(0,len(T)):
        d1 = cv2.absdiff(T[(i+4)%5],T[i])
        d2 = cv2.absdiff(T[(i+4)%5],T[i])
        dfinal = cv2.bitwise_and(d1,d2)
        t = dfinal > 10
        dfinal[t] = 255
        t = dfinal <= 10
        dfinal[t] = 0
        dfinal = cv2.erode(dfinal, kernel, iterations=1)
        for j in xrange(5):
            dfinal = cv2.morphologyEx(dfinal, cv2.MORPH_OPEN, kernel)
            dfinal = cv2.morphologyEx(dfinal, cv2.MORPH_CLOSE, kernel)
        dfinal = cv2.dilate(dfinal, kernel, iterations=10)
        # cv2.imwrite("diff"+str(i)+".jpg",dfinal)
        diff[i] = dfinal
        diff[i] = np.equal(diff[i],0)
        diff[i] = diff[i].astype(int)
    return diff
def KMeans(img):
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    return res2
def newThreshold(gray,color):
    copy = np.array(gray,copy=True,dtype=np.int16)
    thresh_imgs = [0 for x in xrange(len(gray))]
    kernel = np.ones((5,5),np.float32)/25
    for i in xrange(0,len(gray)):
        thresh = [0 for x in xrange(len(color)-1)]
        cnt = 0
        for j in xrange(0,len(color)):
            if i != j:
                d1 = copy[j]-copy[i]
                d1 = d1.clip(0)
                d2 = copy[i]-copy[j]
                d2 = d1.clip(0)
                dfinal = cv2.bitwise_and(d1,d2)
                thresh[cnt] = dfinal
                cnt += 1
        mask = np.median([thresh[0],thresh[1],thresh[2],thresh[3]],axis=0)
        t = mask >= 10
        mask[t] = 255
        t = mask < 10
        mask[t] = 0
        for j in xrange(5):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=10)
        thresh_imgs[i] = mask
    return thresh_imgs
def oldCombineImages(I,diff):
    # Get the image that has the largest foreground
    T = np.array(I, copy=True)
    numero = 0
    smallest = np.inf
    for i in xrange(len(diff)):
        T[i][:,:,0] = np.multiply(T[i][:,:,0], diff[i])
        T[i][:,:,1] = np.multiply(T[i][:,:,1], diff[i])
        T[i][:,:,2] = np.multiply(T[i][:,:,2], diff[i])
        # cv2.imwrite(str(i)+".jpg", T[i])
        # cv2.imwrite(str(i)+"full.jpg", I[i])
        temp = np.equal(I[i],0)
        temp = temp.astype(int)
        temp = np.sum(temp)
        if temp < smallest:
            smallest = temp
            numero = i
    mask = T[numero]
    D = [0 for x in xrange(len(T)-2)]
    cnt = 0
    for i in xrange(0,len(T)):
        if i == numero or i == ((numero+4)%5):
            pass
        else:
            D[cnt] = I[i]
            cnt += 1
    # mean = np.mean([D[0],D[1],D[2]],axis=0)
    mean = np.median([D[0],D[1],D[2]],axis=0)
    a = np.equal(T[numero], 0)
    a = a.astype(int)
    b = np.multiply(mean, a)
    mask = np.add(mask,b)
    return mask
    D = [0 for x in xrange(len(T)-1)]
    diff2 = [0 for x in xrange(len(T)-1)]
    cnt = 0
    for i in xrange(0,len(T)):
        if i == numero:
            pass
        else:
            D[cnt] = T[i]
            diff2[cnt] = diff[i]
            cnt += 1
    median = np.median([D[0],D[1],D[2],D[3]],axis=0)
    b = np.equal(T[numero],0)
    b = b.astype(int)
    b = np.multiply(median, b)
    # mask = np.add(mask,b)
    return mask
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
    return mask
def integerFunction(I,M):
    f = np.subtract(I,M)
    f = np.absolute(f)
    f = np.equal(f, 0)
    f = f.astype(int)
    return f
def alphaMetricallyTrimmedMean(I,f):
    lambda_k = np.sum([f[0],f[1],f[2],f[3],f[4]], axis=0)
    temp = np.multiply(lambda_k,0.3)
    lambda_k = np.subtract(lambda_k,temp)
    temp = np.multiply(I,f)
    temp = np.sum([temp[0],temp[1],temp[2],temp[3],temp[4]], axis=0)
    lambda_k = np.divide(temp, lambda_k)
    return lambda_k
def combineImages(T,I,diff):
    pass
def baseImages(dir):
    images = os.listdir(dir)
    color = [0 for x in xrange(len(images))]
    gray = [0 for x in xrange(len(images))]
    for i in xrange(len(images)):
        gray[i] = cv2.imread(dir + images[i],cv2.IMREAD_GRAYSCALE)
        color[i] = img = cv2.imread(dir + images[i])
    return gray,color
def newCombineImages(color,gray,thresh):
    # order the images such that 0 has least white, length has most
    temp = [0 for x in xrange(len(thresh))]
    for i in xrange(0,len(thresh)):
        total = np.equal(thresh[i], 255)
        total = total.astype(int)
        total = np.sum(total)
        temp[i] = total
    temp2 = np.array(temp, copy=True)
    temp = np.sort(temp,axis=0)
    thresh_order = [0 for x in xrange(len(thresh))]
    gray_order = [0 for x in xrange(len(thresh))]
    for i in xrange(0,len(temp2)):
        for j in xrange(0,len(temp)):
            if temp2[i] == temp[j]:
                thresh_order[j] = thresh[i]
                gray_order[j] = gray[i]
    for i in xrange(0,len(thresh_order)):
        total = np.equal(thresh_order[i], 255)
        total = total.astype(int)
        total = np.sum(total)
    # now begin
    mask = thresh_order[0]
    mask = np.equal(mask, 0)
    mask = mask.astype(int)
    mask = np.multiply(mask, gray_order[0])
    # cv2.imwrite("mask.jpg",mask)
    for i in xrange(1,len(thresh_order)):
        a = np.equal(thresh_order[i], 0)
        a = a.astype(int)
        a = np.multiply(a, gray_order[i])
        b = np.equal(mask, 0)
        b = b.astype(int)
        c = np.multiply(a, b)
        mask = np.add(mask,c)
    med = np.median([gray_order[1],gray_order[2],gray_order[3],gray_order[4]],axis=0)
    black = np.equal(mask,0)
    black = black.astype(int)
    med = np.multiply(med, black)
    mask = np.add(mask,med)
    return mask
def removeMovingObjects(dir,output):
    # dir = "images/set2/"
    # I[i] = np.asarray(I[i], dtype=np.int16)
    # Create a mask of images using the difference from the next frame to the current frame
    # diff = oldthreshold(T)
    # diff = threshold(T)
    diff = threshold(T)
    # Using the largest foreground, apply the background of all other images
    mask = oldCombineImages(I,diff)
    cv2.imwrite(output, mask)
def newRemoveMovingObjects(dir,output):
    gray,color = baseImages(dir)
    thresh = newThreshold(gray,color)
    mask = newCombineImages(color,gray,thresh)
    cv2.imwrite(output,mask)
