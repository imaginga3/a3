import cv2
import numpy as np
from skimage import filters
def fillPatch(I,object_region,region,k):
    # Bound the window to be grabbed so it does not go out of bounds
    Xmin = np.max([0,region[0]-k])
    Xmax = np.min([I.shape[1],region[2]+k])
    Ymin = np.max([0,region[1]-k])
    Ymax = np.min([I.shape[0],region[3]+k])
    # Fill in the inpainted region to the overall image
    I[Ymin:Ymax,Xmin:Xmax,:] = object_region
    return I
def fill(conf_level,source_region,object_region,kernel):
    k = kernel / 2
    # Get the pixel to be updated around
    pixel = np.unravel_index(conf_level.argmax(), conf_level.shape)
    # Create a smaller subwindow to make search faster
    win = 10
    # Bound the window to be grabbed so it does not go out of bounds
    I_sub_min = np.max([0,pixel[0]-win])
    I_sub_max = np.min([object_region.shape[0],pixel[0]+win])
    J_sub_min = np.max([0,pixel[1]-win])
    J_sub_max = np.min([object_region.shape[1],pixel[1]+win])
    # Grab the small window from the source and object images
    pixel_region = np.array(object_region[I_sub_min:I_sub_max,J_sub_min:J_sub_max,:])
    pixel_source = np.array(source_region[I_sub_min:I_sub_max,J_sub_min:J_sub_max])
    pixel_conf = np.array(conf_level[I_sub_min:I_sub_max,J_sub_min:J_sub_max])
    sub_pixel = np.unravel_index(pixel_conf.argmax(), pixel_conf.shape)
    # Bound the window to be grabbed so it does not go out of bounds
    I_min = np.max([0,sub_pixel[0]-k])
    I_max = np.min([pixel_region.shape[0],sub_pixel[0]+k])
    J_min = np.max([0,sub_pixel[1]-k])
    J_max = np.min([pixel_region.shape[1],sub_pixel[1]+k])
    # Get the pixels intensity values
    object_patch = np.array(pixel_region[I_min:I_max,J_min:J_max,:],copy=True)
    source_patch = np.array(pixel_source[I_min:I_max,J_min:J_max],copy=True)
    # Grab the constant window to be checked against
    object_patch[:,:,0] = np.multiply(object_patch[:,:,0],source_patch)
    object_patch[:,:,1] = np.multiply(object_patch[:,:,1],source_patch)
    object_patch[:,:,2] = np.multiply(object_patch[:,:,2],source_patch)
    ssd = float('inf')
    I_fill = [0,0]
    J_fill = [0,0]
    for i in xrange(pixel_region.shape[0]):
        for j in xrange(pixel_region.shape[1]):
            if i == pixel[0] and j == pixel[1]:
                pass
            else:
                # Bound the window to be grabbed so it does not go out of bounds
                Imin = np.max([0,i-k])
                Imax = np.min([pixel_region.shape[0],i+k])
                Jmin = np.max([0,j-k])
                Jmax = np.min([pixel_region.shape[1],j+k])
                # Grab the window to be checked against the original window
                check = pixel_source[Imin:Imax,Jmin:Jmax]
                check = np.where(check == 0)
                if check[0].shape[0] == 0:
                    check_patch = np.array(pixel_region[Imin:Imax,Jmin:Jmax,:])
                    source_check = np.array(pixel_source[Imin:Imax,Jmin:Jmax])
                    check_patch[:,:,0] = np.multiply(check_patch[:,:,0],source_check)
                    check_patch[:,:,1] = np.multiply(check_patch[:,:,1],source_check)
                    check_patch[:,:,2] = np.multiply(check_patch[:,:,2],source_check)
                    if check_patch.shape == object_patch.shape:
                        # Calculate the SSD
                        temp_ssd = cv2.absdiff(check_patch,object_patch)
                        temp_ssd = np.multiply(temp_ssd,temp_ssd)
                        temp_ssd = np.sum(temp_ssd)
                        if temp_ssd < ssd:
                            # Update the window most similar only if SSD < current smallest SSD
                            ssd = temp_ssd
                            I_fill = [Imin,Imax]
                            J_fill = [Jmin,Jmax]
    # Create the fill to be added to the subregion
    fill = np.array(pixel_region[I_fill[0]:I_fill[1],J_fill[0]:J_fill[1],:])
    mask = np.array(pixel_source[I_min:I_max,J_min:J_max])
    mask = np.equal(mask,0)
    mask = mask.astype(int)
    fill[:,:,0] = np.multiply(fill[:,:,0],mask)
    fill[:,:,1] = np.multiply(fill[:,:,1],mask)
    fill[:,:,2] = np.multiply(fill[:,:,2],mask)
    # Add the fill to the subregion
    pixel_region[I_min:I_max,J_min:J_max,:] = np.add(pixel_region[I_min:I_max,J_min:J_max,:],fill)
    pixel_source[I_min:I_max,J_min:J_max] = 1
    object_region[I_sub_min:I_sub_max,J_sub_min:J_sub_max,:] = pixel_region
    source_region[I_sub_min:I_sub_max,J_sub_min:J_sub_max] = pixel_source
    return source_region,object_region
def confidence(target_region,source_region,kernel):
    # In order to make large inpainting faster, only the border is determined for priority
    # patch as the border is the only place the max priority pixel will be located
    temp = np.array(source_region,dtype=np.float32)
    border = cv2.Laplacian(temp,cv2.CV_32F)
    border = np.multiply(border,255)
    border_pixels = np.where(border > 0)
    c_p = np.zeros((source_region.shape[0],source_region.shape[1]), dtype=np.float32)
    rc_p = np.zeros((source_region.shape[0],source_region.shape[1]), dtype=np.float32)
    target_gray = cv2.cvtColor(target_region,cv2.COLOR_RGB2GRAY)
    k = kernel / 2
    omega = 0.5
    for i in xrange(border_pixels[0].shape[0]):
        # grab the current patch window to be checked
        window = source_region[np.max([0,border_pixels[0][i]-k]):np.min([source_region.shape[0],border_pixels[0][i]+k]), np.max([0,border_pixels[1][i]-k]):np.min([source_region.shape[1],border_pixels[1][i]+k])]
        # Calculate C(p) = sum(window) / total window pixels
        sum = np.sum(window)
        total = window.shape[0] * window.shape[1]
        c_p[border_pixels[0][i]][border_pixels[1][i]] = sum / float(total)
        # Calculate RC(p) = (1-omega)*C(p) + omega
        rc_p[border_pixels[0][i]][border_pixels[1][i]] = ((1-omega)*c_p[border_pixels[0][i]][border_pixels[1][i]]) + omega
    # Calcuate D(p) = gradient(p) / 255
    d_p = filters.sobel(target_gray)
    d_p = np.divide(d_p,255) # NOTE: Keep this?
    # Calculate P(p) = RC(p)*alpha + D(p)* beta
    alpha = 0.7
    beta = 0.3
    t1 = np.multiply(rc_p,alpha)
    t2 = np.multiply(d_p,beta)
    p_p = np.add(t1,t2)
    return p_p
def patch(img,region,k):
    # Grab a subregion of the image, around where the object to be removed is
    # to allow algorithm to run faster and more accurately
    object_region = np.array(img, copy=True)
    object_region[region[1]:region[3],region[0]:region[2],:] = 0
    # Create a source region as defined in the document
    source_region = np.full((img.shape[0],img.shape[1]), 1)
    source_region[region[1]:region[3],region[0]:region[2]] = 0
    # Bound the pixel region grabbed to ensure it does not go out of bounds
    Xmin = np.max([0,region[0]-k])
    Xmax = np.min([img.shape[1],region[2]+k])
    Ymin = np.max([0,region[1]-k])
    Ymax = np.min([img.shape[0],region[3]+k])
    source_region = source_region[Ymin:Ymax,Xmin:Xmax]
    object_region = object_region[Ymin:Ymax,Xmin:Xmax]
    return source_region,object_region
def isComplete(source_region):
    # Get the total number of filled in pixels, and the total number of pixels,
    # return true only if the sum equals the total pixel count
    sum = np.sum(source_region)
    total = source_region.shape[0] * source_region.shape[1]
    if sum == total:
        return True
    else:
        return False
def removeStillObject(img,region,output):
    # Load in the image to be used
    I = cv2.imread(img)
    kernel = 7
    k = 20
    # Create the source region, and the original image minus the portion to be replaced
    source_region,object_region = patch(I,region,k)
    # Loop through until the source region is full
    done = False
    while done == False:
        # Determing the confidence level of all of the pixels
        conf_level = confidence(object_region,source_region,kernel)
        # Fill in the window located at the priority patch
        source_region,object_region = fill(conf_level,source_region,object_region,kernel)
        # Check if the inpainting has completed
        done = isComplete(source_region)
    # Use the object region to fill in the remainder of the image
    I = fillPatch(I,object_region,region,k)
    cv2.imwrite(output,I)
