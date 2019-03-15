import os
import imageio
import numpy as np
# Moving object detection with background subtraction and shadow removal
def MAD(I,M):
    mad = np.subtract(I,M)
    mad = np.absolute(mad)
    mad *= 1.4826
    return mad
def alphaMetricallyTrimmedMean(I,f):
    lambda_k = np.multiply(I,0.3)
    lambda_k = np.subtract(I,lambda_k)
    f = np.multiply(I,f)
    f = np.sum([f[0],f[1],f[2],f[3],f[4]], axis=0)
    lambda_k = np.divide(f, lambda_k)
    print lambda_k
    # lambda_k = np.ceil(lambda_k)
    # lambda_k = np.divide(I,lambda_k)
    # return lambda_k
    pass
def integerFunction(I,M):
    f = np.subtract(I,M)
    f = np.absolute(f)
    f = np.equal(f, 0)
    f = f.astype(int)
    # f[1] = np.multiply(f[1], 2)
    # f[2] = np.multiply(f[2], 3)
    # f[3] = np.multiply(f[3], 4)
    # f[4] = np.multiply(f[4], 5)
    # f = np.amax([f[0],f[1],f[2],f[3],f[4]],axis=0)
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
    # imageio.imwrite('median.jpg', M)
    f = integerFunction(I,M)
    # imageio.imwrite('integerFunction.jpg', f)
    lambda_k = alphaMetricallyTrimmedMean(I,f)
    # imageio.imwrite('lambda_k.jpg', lambda_k)
    # mad = MAD(I,M)
    # imageio.imwrite('mad.jpg', mad )
if __name__ == "__main__":
    main()
