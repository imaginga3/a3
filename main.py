import os
import math
import imageio
import numpy as np
# Moving object detection with background subtraction and shadow removal
def foreground(I,lambda_k,mad):
    w = np.array([[1,2,1],[2,4,2],[1,2,1]])
    h = 3
    # 1. window * abs(calc I - lambda_k)
    # 2. h * window * lambda_k
    print w
def MAD(I,M):
    mad = np.subtract(I,M)
    mad = np.absolute(mad)
    mad = np.median([mad[0],mad[1],mad[2],mad[3],mad[4]],axis=0)
    mad *= 1.4826
    return mad
def alphaMetricallyTrimmedMean(I,f):
    lambda_k = np.multiply(I,0.3)
    lambda_k = np.subtract(I,lambda_k)
    f = np.multiply(I,f)
    f = np.sum([f[0],f[1],f[2],f[3],f[4]], axis=0)
    lambda_k = np.divide(f, lambda_k)
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
def main():
    I = temporalIntensity('images/home/')
    M = temporalMedian(I)
    f = integerFunction(I,M)
    lambda_k = alphaMetricallyTrimmedMean(I,f)
    mad = MAD(I,M)
    foreground(I,lambda_k,mad)
if __name__ == "__main__":
    main()
