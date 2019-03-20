import math
import cv2
import os
import numpy as np
from scipy import stats
# Unsupervised background reconstruction based on iterative median blending and spatial segmentation
def coarseObjects(I,B):
    I = np.array(I, copy=True, dtype=np.int16)
    B = np.array(B, copy=True, dtype=np.int16)
    diff = cv2.absdiff(I,B)
    diff = np.max(diff,axis=2)
    th = 30
    tl = 10
    M = np.array(diff,copy=True)
    t = diff >= tl
    M[t] = 130
    t = diff > th
    M[t] = 1
    t = diff < tl
    M[t] = 0
    t = M > 0
    M[t] = 1
    kernel = np.array([[0,1,0],[1,0,1],[0,1,0]])
    M = cv2.filter2D(M,-1,kernel)
    t = M > 0
    M[t] = 1
    return M
def temporalMedian(I):
    M = np.median([I[0],I[1],I[2],I[3],I[4]],axis=0)
    return M
def frameIntensities(dir):
    images = os.listdir(dir)
    color = [0 for x in xrange(len(images))]
    for i in xrange(len(images)):
        color[i] = cv2.imread(dir + images[i])
    return color
def removeForeground(M,B):
    B2 = np.array(B,copy=True)
    B3 = np.array(B,copy=True)
    n = np.equal(M,0)
    n = n.astype(int)
    B2[:,:,0] = np.multiply(B2[:,:,0],n)
    B2[:,:,1] = np.multiply(B2[:,:,1],n)
    B2[:,:,2] = np.multiply(B2[:,:,2],n)
    n = np.equal(M,1)
    n = n.astype(int)
    B3[:,:,0] = np.multiply(B3[:,:,0],n)
    B3[:,:,1] = np.multiply(B3[:,:,1],n)
    B3[:,:,2] = np.multiply(B3[:,:,2],n)
    return B2,B3
def fill(frames,t,bg):
    # 1. take the average across all of the images with background that is not black
    return bg
    a = np.equal(bg,0)
    a = a.astype(int)
    b = np.multiply(frames,a)
    c = np.sum(a,axis=2)
    d = np.sum(b,axis=0)
    t = c == 0
    c[t] = 1
    d[:,:,0] = np.divide(d[:,:,0],c)
    d[:,:,1] = np.divide(d[:,:,1],c)
    d[:,:,2] = np.divide(d[:,:,2],c)
    final = np.add(bg,d)
    print np.max(final)
    return final
def alphaMetricallyTrimmedMean(I,f):
    lambda_k = np.sum([f[0],f[1],f[2],f[3],f[4]], axis=0)
    temp = np.multiply(lambda_k,0.2)
    lambda_k = np.subtract(lambda_k,temp)
    temp = np.multiply(I,f)
    temp = np.sum([temp[0],temp[1],temp[2],temp[3],temp[4]], axis=0)
    t = lambda_k == 0
    lambda_k[t] = 1
    lambda_k = np.divide(temp, lambda_k)
    return lambda_k
def integerFunction(I,M):
    i = np.array(I,copy=True,dtype=np.int16)
    m = np.array(M,copy=True,dtype=np.int16)
    f = np.subtract(i,m)
    f = np.absolute(f)
    f = np.equal(f, 0)
    f = f.astype(int)
    return f
def a(I,B,F):
    # I = frameIntensities('images/set2/')
    # M = temporalMedian(I)
    f = integerFunction(I,F)
    lambda_k = alphaMetricallyTrimmedMean(I,f)
    full = B + lambda_k
    return full
def iterativeMedianBlending(dir,output):
    frames = frameIntensities(dir)
    B = temporalMedian(frames)
    M = [0 for x in xrange(len(frames))]
    t = [0 for x in xrange(len(frames))]
    for i in xrange(len(frames)):
        M[i] = coarseObjects(frames[i],B)
        b,f = removeForeground(M[i],B)
        t[i] = b
    m = temporalMedian(M)
    bg,fg = removeForeground(m,B)
    out = a(frames,bg,fg)
    cv2.imwrite(output,bg)
