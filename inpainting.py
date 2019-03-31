import cv2
import numpy as np
def fillPatch(target_region,source_region,conf_level,region,kernel):
    target = np.array(target_region, copy=True)
    max = np.max(conf_level)
    location = np.where(conf_level == max)
    pixel = [location[1][0],location[0][0]]
    Xmin = np.max([0,pixel[1]-kernel])
    Xmax = np.min([target.shape[0],pixel[1]+kernel])
    Ymin = np.max([0,pixel[0]-kernel])
    Ymax = np.min([target.shape[1],pixel[0]+kernel])
    range1 = np.array([Xmax-Xmin,Ymax-Ymin])
    patch1 = target[Ymin:Ymax,Xmin:Xmax,:]
    min_ssd = float('inf')
    fill = 0
    x = [0,0]
    y = [0,0]
    for i in xrange(conf_level.shape[0]):
        for j in xrange(conf_level.shape[1]):
            Xmin = np.max([0,i-kernel])
            Xmax = np.min([target.shape[0],i+kernel])
            Ymin = np.max([0,j-kernel])
            Ymax = np.min([target.shape[1],j+kernel])
            range2 = np.array([Xmax-Xmin,Ymax-Ymin])
            if range1[0] == range2[0] and range1[1] == range2[1]:
                check = source_region[Ymin:Ymax,Xmin:Xmax]
                check = np.where(check == 0)
                if check[0].shape[0] == 0:
                    patch2 = target[Ymin:Ymax,Xmin:Xmax,:]
                    ssd = cv2.absdiff(patch1,patch2)
                    ssd = np.multiply(ssd,ssd)
                    ssd = np.sum(ssd)
                    if ssd < min_ssd:
                        x = [i-Xmin,Xmax-i]
                        y = [j-Ymin,Ymax-j]
                        min_ssd = ssd
                        fill = target   [Ymin:Ymax,Xmin:Xmax,:]
    # Add the fill to the correct location
    Xmin = np.max([0,pixel[0]-x[0]])
    Xmax = np.min([source_region.shape[0],pixel[0]+x[1]])
    Ymin = np.max([0,pixel[1]-y[0]])
    Ymax = np.min([source_region.shape[1],pixel[1]+y[1]])
    mask = np.equal(source_region,0)
    mask = mask.astype(int)
    mask = mask[Ymin:Ymax,Xmin:Xmax]
    fill[:,:,0] = np.multiply(fill[:,:,0],mask)
    fill[:,:,1] = np.multiply(fill[:,:,1],mask)
    fill[:,:,2] = np.multiply(fill[:,:,2],mask)
    print fill[:,:,0]
    print fill[:,:,1]
    print fill[:,:,2]
    target_region[Ymin:Ymax,Xmin:Xmax,:] = np.add(target_region[Ymin:Ymax,Xmin:Xmax,:], fill)
    source_region[Ymin:Ymax,Xmin:Xmax] = 1
    return target_region, source_region
def confidence(target_region,source_region,kernel):
    c_p = np.zeros((source_region.shape[0],source_region.shape[1]), dtype=np.float32)
    k = kernel / 2
    for i in xrange(source_region.shape[0]):
        for j in xrange(source_region.shape[1]):
            if source_region[i][j] == 0:
                window = source_region[np.max([0,i-k]):np.min([source_region.shape[0],i+k]), np.max([0,j-k]):np.min([source_region.shape[1],j+k])]
                sum = np.sum(window)
                total = window.shape[0] * window.shape[1]
                c_p[i][j] = sum / float(total)
    return c_p
def patch(img,region):
    img[region[1]:region[3],region[0]:region[2],:] = 0
    source_region = np.full((img.shape[0],img.shape[1]), 1)
    source_region[region[1]:region[3],region[0]:region[2]] = 0
    return img,source_region
def removeStillObject():
    I = cv2.imread("images/still/5-ball_small.jpg")
    kernel = 9
    # temporary target region
    # [x1, y1, x2, y2]
    region = np.array([150,160,225,270])
    I,source_region = patch(I,region)
    temp = np.multiply(source_region,255)
    for i in xrange(0,10):
        conf_level = confidence(I,source_region,kernel)
        I,source_region = fillPatch(I,source_region,conf_level,region,kernel)
        cv2.imwrite("fill_"+str(i)+".jpg",I)
        temp = np.multiply(source_region,255)
        cv2.imwrite("source_"+str(i)+".jpg",temp)
removeStillObject()
