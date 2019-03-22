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
def removeMostLike(frames,median):
    # convert to gray
    m = np.array(median,copy=True,dtype=np.uint8)
    m = cv2.cvtColor(m,cv2.COLOR_BGR2GRAY)
    m = np.array(median,copy=True,dtype=np.int16)
    f = [0 for x in xrange(len(frames))]
    for i in xrange(len(frames)):
        f[i] = np.array(frames[i],copy=True,dtype=np.uint8)
        f[i] = cv2.cvtColor(f[i],cv2.COLOR_BGR2GRAY)
        f[i] = np.array(f[i],copy=True,dtype=np.int16)
    # get the difference between the total counts between the median and the current frame
    m_total = np.sum(m)
    lg_diff = 0
    fr_n = 0
    for i in xrange(len(f)):
        curr = np.sum(f[i])
        diff = abs(curr-m_total)
        if diff < lg_diff:
            fr_n = i
            lg_diff = diff
    # remove the frame from frames and return both the frames and number
    bg = np.array(frames[fr_n],copy=True)
    frames = np.delete(frames,fr_n,0)
    return frames,bg
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
def hysterisesThresholding(diff,t_h,t_l):
    thresh = np.array(diff,copy=True)
    t = diff >= t_l
    thresh[t] = 1
    t = diff >= t_h
    thresh[t] = 10
    t = diff < t_l
    thresh[t] = 0
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
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
def objectDetection(frames,bg):
    scale = 8
    small_frames = [0 for x in xrange(len(frames))]
    for i in xrange(len(small_frames)):
        small_frames[i] = resizeImage(frames[i],scale)
    small_B = resizeImage(bg,scale)
    diff = [0 for x in xrange(len(frames))]
    small_frames = np.array(small_frames,copy=True,dtype=np.int16)
    small_B = np.array(small_B,copy=True,dtype=np.int16)
    for i in xrange(len(small_frames)):
        diff[i] = cv2.absdiff(small_frames[i],small_B)
        diff[i] = resizeImage(diff[i],8,cv2.INTER_CUBIC)
    mask = [0 for x in xrange(len(frames))]
    kernel = np.ones((5,5),np.uint8)
    for i in xrange(len(diff)):
        threshold = hysterisesThresholding(diff[i],10,5)
        temp = np.array(frames[i],copy=True)
        temp[:,:,0] = np.multiply(temp[:,:,0],threshold)
        temp[:,:,1] = np.multiply(temp[:,:,1],threshold)
        temp[:,:,2] = np.multiply(temp[:,:,2],threshold)
        temp = cv2.erode(temp,kernel,iterations = 5)
        temp = cv2.dilate(temp,kernel,iterations = 5)
        mask[i] = temp
    bg = np.array(bg,copy=True,dtype=np.int16)
    fg = [0 for x in xrange(len(frames))]
    for i in xrange(len(mask)):
        fg[i] = np.array(mask[i],copy=True,dtype=np.int16)
        bg_mask = np.greater(fg[i],0)
        bg_mask = bg_mask.astype(int)
        fg[i] = np.multiply(bg,bg_mask)
    foreground = np.mean([fg[0],fg[1],fg[2],fg[3]],axis=0)
    fgg = np.isclose(foreground,bg,atol=1)
    fgg = fgg.astype(int)
    foreground = np.multiply(foreground,fgg)
    foreground = cv2.dilate(foreground,kernel,iterations = 3)
    t = foreground > 0
    foreground[t] = 255
    r = np.equal(foreground[:,:,0],255)
    g = np.equal(foreground[:,:,1],255)
    b = np.equal(foreground[:,:,2],255)
    nd = np.logical_and(r,g)
    nd = nd.astype(int)
    nd = np.logical_and(nd,b)
    nd = nd.astype(int)
    foreground[:,:,0] = np.multiply(foreground[:,:,0],nd)
    foreground[:,:,1] = np.multiply(foreground[:,:,1],nd)
    foreground[:,:,2] = np.multiply(foreground[:,:,2],nd)
    foreground = np.multiply(foreground,255)
    foreground = np.greater(foreground,0)
    foreground = foreground.astype(int)
    object = np.multiply(foreground,bg)
    object = np.array(object,copy=True,dtype=np.int16)
    return object
def generateBackground(objects,frames):
    bg = [0 for x in xrange(len(frames))]
    for i in xrange(len(bg)):
        fg_mask = np.equal(objects[i],0)
        fg_mask = fg_mask.astype(int)
        bg[i] = np.multiply(frames[i],fg_mask)
    background = temporalMedianBlending(bg)
    mask = np.equal(background,0)
    mask = mask.astype(int)
    mask = np.multiply(mask,255)
    mask = mask[:,:,0]
    mask = np.array(mask,copy=True,dtype=np.uint8)
    background = np.array(background,copy=True,dtype=np.uint8)
    background = cv2.inpaint(background,mask,3,cv2.INPAINT_TELEA)
    return background
def reconstruct(dir,output):
    frames = frameIntensities(dir)
    median = temporalMedianBlending(frames)
    objects = [0 for x in xrange(len(frames))]
    for i in xrange(len(frames)):
        # frames,bg = removeMostLike(frames,median)
        bg = frames[i]
        fr = np.delete(frames,i,0)
        objects[i] = objectDetection(fr,bg)
    out = generateBackground(objects,frames)
    cv2.imwrite(output,out)
