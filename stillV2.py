import cv2
import math
import numpy as np
from scipy import ndimage
def checkSource(source):
    count = np.sum(source)
    total = source.shape[0]*source.shape[1]*source.shape[2]
    print str(count) + " of " + str(total)
    if count == total:
        return True
    else:
        return False
def fillSource(source,sub_reg,range_i,range_j,pixel):
    small_region = sub_reg[range_i[0]:range_i[1], range_j[0]:range_j[1],:]
    mean = np.mean(small_region,axis=tuple(range(small_region.ndim-1)))
    kernel = 3
    sse_min = float('inf')
    win = 0
    win_colour = 0
    i_range = [0,0]
    j_range = [0,0]
    for i in xrange(small_region.shape[0]):
        for j in xrange(small_region.shape[1]):
            i_1 = np.max([0,i-kernel])
            i_2 = np.min([sub_reg.shape[0],i+kernel])
            j_1 = np.max([0,j-kernel])
            j_2 = np.min([sub_reg.shape[1],j+kernel])
            window = small_region[i_1:i_2,j_1:j_2,:]
            # mean = np.mean(window,axis=tuple(range(window.ndim-1)))
            sse = window - mean
            sse = np.multiply(sse,sse)
            sse = np.sum(sse)
            if sse < sse_min:
                win = window
                sse_min = sse
                i_range = [i_1-i_1,i_2-i_1]
                j_range = [j_1-j_1,j_2-j_1]
    i_range = np.add(i_range,pixel[0])
    j_range = np.add(j_range,pixel[1])
    # Update the sub region
    neg_source = np.equal(source,0)
    neg_source = neg_source.astype(int)
    neg_source = neg_source[i_range[0]:i_range[1],j_range[0]:j_range[1],:]
    if win.shape == neg_source.shape:
        win = np.multiply(neg_source,win)
        sub_reg[i_range[0]:i_range[1],j_range[0]:j_range[1],:] = np.add(sub_reg[i_range[0]:i_range[1],j_range[0]:j_range[1],:],win)
        print win
        # sub_reg[i_range[0]:i_range[1],j_range[0]:j_range[1],:] += win
    # Update the source image
    source[i_range[0]:i_range[1],j_range[0]:j_range[1],:] = 1
    return source,sub_reg
def regionToCheck(pixel,img,boundary):
    i_min = img.shape[0]-pixel[0]
    i_max = img.shape[0]+pixel[0]
    j_min = img.shape[1]-pixel[1]
    j_max = img.shape[1]+pixel[1]
    range_i = [i_min,i_max]
    range_j = [j_min,j_max]
    return range_i,range_j
def confidence(source,kernel,boundary):
    k = kernel/2
    max = 0.0
    max_i = 0
    max_j = 0
    for i in xrange((boundary-1),(source.shape[0]-boundary+1)):
        for j in xrange(boundary,(source.shape[1]-boundary+1)):
            x = source[(i-k):(i+k),(j-k):(j+k),0]
            c_p = (np.sum(x) / (kernel*kernel*3))
            if source[i][j][0] == 0 and c_p > max and c_p != 1.0:
                max = c_p
                max_i = i
                max_j = j
    max = [max_i,max_j]
    return max
def subregion(img,region,boundary):
    # Extend region to have 7 extra bordering pixels
    region[0] -= boundary
    region[1] -= boundary
    region[2] += boundary
    region[3] += boundary
    # region[0] = np.clip(region[0],0)
    # region[2] = np.clip(region[2],1600)
    # region[1] = np.clip(region[1],0)
    # region[3] = np.clip(region[3],2400)
    subregion = np.array(img[region[1]:region[3],region[0]:region[2],:], copy=True)
    # subregion = img[region[1]:region[3],region[0]:region[2],:]
    # Grab the region for the source region
    source = np.array(subregion, copy=True)
    t = source >= 0
    source[t] = 1
    source[boundary:(source.shape[0]-boundary),boundary:(source.shape[1]-boundary),:] = 0
    subregion = np.multiply(subregion,source)
    return source,subregion
def removeStillImage(rect,a_min,a_max):
    img = cv2.imread("images/still/5-ball_small.jpg")
    # temporary target region
    # [x1, y1, x2, y2]
    region = np.array([156,162,222,269])
    boundary = 7
    source,sub_reg = subregion(img,region,boundary)
    # cv2.imwrite("source.jpg",source)
    # cv2.imwrite("subregion.jpg",sub_reg)
    finished = False
    i = 0
    while finished == False:
        pixel = confidence(source,boundary,boundary)
        range_i,range_j = regionToCheck(pixel,sub_reg,boundary)
        source,sub_reg = fillSource(source,sub_reg,range_i,range_j,pixel)
        finished = checkSource(source)
        i+=1
        cv2.imwrite(str(i)+".jpg",sub_reg)
    cv2.imwrite("final.jpg",sub_reg)
removeStillImage([1,2,3,4],500,800)
