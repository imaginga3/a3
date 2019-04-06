import cv2
import os
import numpy as np
def temporalMedianBlending(I):
    # Calculate temporal median of I, disregaring NaN values
    return np.nanmedian([I],axis=1)[0]
def frameIntensities(dir):
    # Read an entire directory of images in
    images = os.listdir(dir)
    imgs = [0 for x in xrange(len(images))]
    for i in xrange(len(images)):
        imgs[i] = cv2.imread(dir + images[i])
    return imgs
def resizeImage(img,scale,interp=None):
    if interp == None: # Option 1: Make image smaller
        width = int((img.shape[1])/scale)
        height = int((img.shape[0])/scale)
        dim = (width,height)
        return cv2.resize(img,dim)
    else: # Option 2: Make image bigger
        width = int((img.shape[1])*scale)
        height = int((img.shape[0])*scale)
        dim = (width,height)
        return cv2.resize(img,dim,interpolation=interp)
def hysterisesThresholding(diff,t_h,t_l):
    thresh = np.array(diff,copy=True)
    # If the threshold is greater than T_l set to weak foreground pixel
    t = diff >= t_l
    thresh[t] = 1
    # If the threshold is greater than T_h set to strong foreground pixel
    t = diff >= t_h
    thresh[t] = 10
    # Set rest to background
    t = diff < t_l
    thresh[t] = 0
    # Go through neighbours of 8 and keep only weak pixels if they have a strong
    # foreground pixel near
    kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
    thresh = cv2.filter2D(thresh,-1,kernel)
    thresh = np.greater_equal(thresh,10)
    thresh = thresh.astype(int)
    new_thresh = np.zeros((thresh.shape[0],thresh.shape[1]))
    # Threshold to background and foreground
    t = thresh[:,:,0] == 1
    new_thresh[t] = 1
    t = thresh[:,:,1] == 1
    new_thresh[t] = 1
    t = thresh[:,:,2] == 1
    new_thresh[t] = 1
    return new_thresh
def removeForeground(frames,background,time):
    bg = np.array(background, copy=True, dtype=np.int16)
    mask = [0 for x in xrange(len(frames))]
    kernel = np.ones((3,3),np.uint8)
    for i in xrange(len(frames)):
        # Determine the foreground
        fr = np.array(frames[i],copy=True,dtype=np.int16)
        t = np.isclose(bg,fr,atol=15)
        t = t.astype(int)
        t = np.all(t != [0,0,0], axis=-1)
        # Using neighbours of 8, remove all those that shouldn't be foreground
        ########################################################
        t = np.array(t, copy=True, dtype=np.uint8)
        # kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
        # thresh = cv2.filter2D(t,-1,kernel)
        # thresh = np.greater_equal(thresh,1)
        # thresh = thresh.astype(int)
        # mask[i] = t
        ########################################################
        # mask[i] = cv2.erode(t,kernel,iterations = 1)
        # temp = cv2.dilate(temp,kernel,iterations = 5)
        # mask[i] = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel)
        mask[i] = cv2.morphologyEx(t, cv2.MORPH_OPEN, kernel)
    # Get the temporal median of all backgrounds computed
    bg_mask = temporalMedianBlending(mask)
    bg_mask = np.multiply(bg_mask, 255)
    # Further separate the foregound from the background, keeping only those pixels
    # that are black across all iterations
    bg_mask = np.greater(bg_mask, 0)
    bg_mask = bg_mask.astype(int)
    # Grab the background from the original image
    new_bg = np.zeros((background.shape[0],background.shape[1],background.shape[2]))
    new_bg[:,:,0] = np.multiply(bg[:,:,0], bg_mask)
    new_bg[:,:,1] = np.multiply(bg[:,:,1], bg_mask)
    new_bg[:,:,2] = np.multiply(bg[:,:,2], bg_mask)
    return new_bg
def generateBackground(objects,frames):
    # Grab the temporal median of all frames of background
    new_bg = temporalMedianBlending(objects)
    # Sometimes there will be nothing remaining in some pixels, find these pixels,
    # use the image which resembles the current background most
    median = temporalMedianBlending(frames)
    mask = np.equal(new_bg,0)
    mask = mask.astype(int)
    median = np.multiply(median, mask)
    # Add the good background and the remainder together
    new_bg = np.add(new_bg, median)
    # new_bg = np.add(new_bg, closest)
    return new_bg
def reconstruct(dir,output):
    # Get all frame intensities to be used
    frames = frameIntensities(dir)
    median = temporalMedianBlending(frames)
    backgrounds = [0 for x in xrange(len(frames))]
    # Iteratively remove the current frame to compare against the remainder of the frames
    # in order to threshold out the foreground from background
    for i in xrange(len(frames)):
        bg = frames[i]
        fr = np.delete(frames,i,0)
        # Separate the foregound and background
        backgrounds[i] = removeForeground(fr,bg,i)
        cv2.imwrite(str(i)+".jpg",backgrounds[i])
    # Using the generated backrounds, get the overall background
    out = generateBackground(backgrounds,frames)
    cv2.imwrite(output,out)
