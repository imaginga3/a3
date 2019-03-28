import cv2
import numpy as np
def border(img):
    return cv2.Laplacian(img,cv2.CV_64F)
def subregion(img,region,boundary):
    # Extend region to have 7 extra bordering pixels
    # Grab the subregion to be changed
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
    # Grab the region for the confidence level
    confidence = np.array(subregion, copy=True)
    t = confidence >= 0
    confidence[t] = 255
    confidence[boundary:(confidence.shape[0]-boundary),boundary:(confidence.shape[1]-boundary),:] = 0
    return confidence,subregion
def removeStillImage(rect,a_min,a_max):
    img = cv2.imread("images/still/5-ball.jpg")
    # temporary target region
    region = np.array([630,630,900,1070])
    boundary = 7
    conf_level,sub_reg = subregion(img,region,boundary)
    cv2.imwrite("confidence.jpg",conf_level)
    cv2.imwrite("subregion.jpg",sub_reg)
    outline = border(conf_level)
    cv2.imwrite("border.jpg",outline)
    # BELOW IS TO BE REMOVED
    # add something to confidence and try border again
    conf_level[boundary:(boundary+15),boundary:(boundary+15),:] = 255
    cv2.imwrite("confidence2.jpg",conf_level)
    outline = border(conf_level)
    cv2.imwrite("border2.jpg",outline)
    conf_level[boundary:(boundary+15),boundary:(boundary+15),:] = 255
    # second time
    conf_level[(conf_level.shape[0]-boundary-15):(conf_level.shape[0]-boundary),(conf_level.shape[1]-boundary-15):(conf_level.shape[1]-boundary),:] = 255
    cv2.imwrite("confidence3.jpg",conf_level)
    outline = border(conf_level)
    cv2.imwrite("border3.jpg",outline)
removeStillImage([1,2,3,4],500,800)
