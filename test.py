import cv2
import os
import numpy as np
def temporalMedianBlending(I):
    return np.nanmedian([I],axis=1)[0]
def frameIntensities(dir):
    images = os.listdir(dir)
    imgs = [0 for x in xrange(len(images))]
    for i in xrange(len(images)):
        imgs[i] = cv2.imread(dir + images[i])
    return imgs
def resizeImage(img,scale,interp=None):
    if interp == None:
        width = int((img.shape[1])/scale)
        height = int((img.shape[0])/scale)
        dim = (width,height)
        return cv2.resize(img,dim)
    else:
        width = int((img.shape[1])*scale)
        height = int((img.shape[0])*scale)
        dim = (width,height)
        return cv2.resize(img,dim,interpolation=interp)
def lowpassFilter(img):
    kernel = np.ones((5,5),np.float32)/25
    return cv2.filter2D(img,-1,kernel)
def hysterisesThresholding(diff,t_h,t_l):
    thresh = np.array(diff,copy=True)
    t = diff >= t_l
    thresh[t] = 1
    t = diff >= t_h
    thresh[t] = 10
    t = diff < t_l
    thresh[t] = 0
    kernel = np.array([[0,1,0],[1,0,1],[0,1,0]])
    thresh = cv2.filter2D(thresh,-1,kernel)
    thresh = np.greater_equal(thresh,10)
    thresh = thresh.astype(int)
    new_thresh = np.zeros((thresh.shape[0],thresh.shape[1]))
    t = thresh[:,:,0] == 1
    new_thresh[t] = 1
    t = thresh[:,:,1] == 1
    new_thresh[t] = 1
    t = thresh[:,:,2] == 1
    new_thresh[t] = 1
    return new_thresh
def coarseObjectDetection(frames,B):
    scale = 8
    small_frames = [0 for x in xrange(len(frames))]
    for i in xrange(len(small_frames)):
        small_frames[i] = resizeImage(frames[i],scale)
    small_B = resizeImage(B,scale)
    # small_B = lowpassFilter(small_B)
    diff = [0 for x in xrange(len(frames))]
    small_frames = np.array(small_frames,copy=True,dtype=np.int16)
    small_B = np.array(small_B,copy=True,dtype=np.int16)
    for i in xrange(len(small_frames)):
        # diff[i] = cv2.absdiff(small_frames[i],small_B)
        diff[i] = np.subtract(small_B,small_frames[i])
        diff[i] = resizeImage(diff[i],8,cv2.INTER_CUBIC)
    # thresholding
    mask = [0 for x in xrange(len(frames))]
    for i in xrange(len(diff)):
        threshold = hysterisesThresholding(diff[i],20,10)
        temp = np.array(frames[i],copy=True)
        temp[:,:,0] = np.multiply(temp[:,:,0],threshold)
        temp[:,:,1] = np.multiply(temp[:,:,1],threshold)
        temp[:,:,2] = np.multiply(temp[:,:,2],threshold)
        mask[i] = temp
        cv2.imwrite(str(i)+'.jpg',mask[i])
    return mask
def newBackground(frames,mask):
    m = np.equal(mask,0)
    m = m.astype(int)
    bg = np.multiply(frames,m)
    bg = np.array(bg,copy=True,dtype=np.float32)
    t = bg == 0
    bg[t] = np.nan
    bg = temporalMedianBlending(bg)
    return bg
def reconstruct(dir,output):
    frames = frameIntensities(dir)
    median = temporalMedianBlending(frames)
    coarse = coarseObjectDetection(frames,median)
    out = newBackground(frames,coarse)
    cv2.imwrite(output,out)
