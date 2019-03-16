import os
import math
import imageio
import cv2
import numpy as np
from skimage.exposure import rescale_intensity
# Moving object detection with background subtraction and shadow removal
# http://machinelearninguru.com/computer_vision/basics/convolution/image_convolution_1.html
def foreground(I,lambda_k,mad):
    w = np.array([[1,2,1],[2,4,2],[1,2,1]])
    h = 3
    # 1. window * abs(calc I - lambda_k)
    # 2. h * window * lambda_k
    padded_one_R = [0 for x in xrange(len(I))]
    padded_one_G = [0 for x in xrange(len(I))]
    padded_one_B = [0 for x in xrange(len(I))]
    one = np.subtract(I,lambda_k)
    one = np.absolute(one)
    for i in xrange(len(one)):
        b,g,r = cv2.split(one[i])
        # For the first calculation
        padded_one_R[i] = np.zeros((one[i].shape[0]+2, one[i].shape[1]+2))
        padded_one_R[i][1:-1, 1:-1] = r
        padded_one_G[i] = np.zeros((one[i].shape[0]+2, one[i].shape[1]+2))
        padded_one_G[i][1:-1, 1:-1] = g
        padded_one_B[i] = np.zeros((one[i].shape[0]+2, one[i].shape[1]+2))
        padded_one_B[i][1:-1, 1:-1] = b
    # For the comparitive calculation
    b,g,r = cv2.split(mad)
    padded_mad_R = np.zeros((mad.shape[0]+2, mad.shape[1]+2))
    padded_mad_R[1:-1, 1:-1] = r
    padded_mad_G = np.zeros((mad.shape[0]+2, mad.shape[1]+2))
    padded_mad_G[1:-1, 1:-1] = g
    padded_mad_B = np.zeros((mad.shape[0]+2, mad.shape[1]+2))
    padded_mad_B[1:-1, 1:-1] = b
    mask = np.zeros((mad.shape[0], mad.shape[1], 3))
    for x in xrange(0, I[0].shape[0]-1):
        for y in xrange(0, I[0].shape[1]-1):
            sum_one = 0
            for i in xrange(len(one)):
                sum_one += (w*padded_one_R[i][x:x+3,y:y+3]).sum()
                sum_one += (w*padded_one_G[i][x:x+3,y:y+3]).sum()
                sum_one += (w*padded_one_B[i][x:x+3,y:y+3]).sum()
            sum_two = (w*padded_mad_R[x:x+3,y:y+3]).sum()
            sum_two += (w*padded_mad_G[x:x+3,y:y+3]).sum()
            sum_two += (w*padded_mad_B[x:x+3,y:y+3]).sum()
            sum_two = h * sum_two
            if sum_one > sum_two:
                mask[x+1,y+1,0] = 255
                mask[x+1,y+1,1] = 255
                mask[x+1,y+1,2] = 255
            else:
                mask[x+1,y+1,0] = 0
                mask[x+1,y+1,1] = 0
                mask[x+1,y+1,2] = 0
    return mask
def MAD(I,M):
    mad = np.subtract(I,M)
    mad = np.absolute(mad)
    mad = np.median([mad[0],mad[1],mad[2],mad[3],mad[4]],axis=0)
    mad *= 1.4826
    return mad
def alphaMetricallyTrimmedMean(I,f):
    lambda_k = np.sum([f[0],f[1],f[2],f[3],f[4]], axis=0)
    temp = np.multiply(lambda_k,0.3)
    lambda_k = np.subtract(lambda_k,temp)
    temp = np.multiply(I,f)
    temp = np.sum([temp[0],temp[1],temp[2],temp[3],temp[4]], axis=0)
    lambda_k = np.divide(temp, lambda_k)
    return lambda_k
def integerFunction(I,M):
    f = np.subtract(I,M)
    f = np.absolute(f)
    f = np.equal(f, 0)
    f = f.astype(int)
    return f
def temporalMedian(T):
    M = np.median([T[0],T[1],T[2],T[3],T[4]],axis=0)
    return M
def temporalIntensity(dir):
    images = os.listdir(dir)
    I = [0 for x in xrange(len(images))]
    for image in xrange(len(images)):
        img = imageio.imread(dir + images[image])
        I[image] = img
    return I
def backgroundSubtraction():
    I = temporalIntensity('images/home/')
    M = temporalMedian(I)
    f = integerFunction(I,M)
    lambda_k = alphaMetricallyTrimmedMean(I,f)
    imageio.imwrite("test.jpg", lambda_k)
    I = temporalIntensity('images/Photos/')
    M = temporalMedian(I)
    f = integerFunction(I,M)
    lambda_k = alphaMetricallyTrimmedMean(I,f)
    imageio.imwrite("test2.jpg", lambda_k)
    # mad = MAD(I,M)
    # mask = foreground(I,lambda_k,mad)
def shadowSubtraction():
    pass
def main():
    backgroundSubtraction()
    # shadowSubtraction()
if __name__ == "__main__":
    main()
