import cv2
import numpy as np
def fillPatch(I,object_region,region,k):
    Xmin = np.max([0,region[0]-k])
    Xmax = np.min([I.shape[1],region[2]+k])
    Ymin = np.max([0,region[1]-k])
    Ymax = np.min([I.shape[0],region[3]+k])
    I[Ymin:Ymax,Xmin:Xmax,:] = object_region
    return I
def fill(conf_level,source_region,object_region,kernel):
    k = kernel / 2
    # Get the pixel to be updated around
    pixel = np.unravel_index(conf_level.argmax(), conf_level.shape)
    # Create a smaller subwindow to make search faster
    win = 10
    I_sub_min = np.max([0,pixel[0]-win])
    I_sub_max = np.min([object_region.shape[0],pixel[0]+win])
    J_sub_min = np.max([0,pixel[1]-win])
    J_sub_max = np.min([object_region.shape[1],pixel[1]+win])
    pixel_region = np.array(object_region[I_sub_min:I_sub_max,J_sub_min:J_sub_max,:])
    pixel_source = np.array(source_region[I_sub_min:I_sub_max,J_sub_min:J_sub_max])
    pixel_conf = np.array(conf_level[I_sub_min:I_sub_max,J_sub_min:J_sub_max])
    sub_pixel = np.unravel_index(pixel_conf.argmax(), pixel_conf.shape)
    I_min = np.max([0,sub_pixel[0]-k])
    I_max = np.min([pixel_region.shape[0],sub_pixel[0]+k])
    J_min = np.max([0,sub_pixel[1]-k])
    J_max = np.min([pixel_region.shape[1],sub_pixel[1]+k])
    # Get the pixels intensity values
    # source_patch = np.array(source_region[I_min:I_max,J_min:J_max],copy=True)
    # object_patch = np.array(source_region[I_min:I_max,J_min:J_max],copy=True)
    object_patch = np.array(pixel_region[I_min:I_max,J_min:J_max,:],copy=True)
    source_patch = np.array(pixel_source[I_min:I_max,J_min:J_max],copy=True)
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
                Imin = np.max([0,i-k])
                Imax = np.min([pixel_region.shape[0],i+k])
                Jmin = np.max([0,j-k])
                Jmax = np.min([pixel_region.shape[1],j+k])
                check = pixel_source[Imin:Imax,Jmin:Jmax]
                check = np.where(check == 0)
                if check[0].shape[0] == 0:
                    check_patch = np.array(pixel_region[Imin:Imax,Jmin:Jmax,:])
                    source_check = np.array(pixel_source[Imin:Imax,Jmin:Jmax])
                    check_patch[:,:,0] = np.multiply(check_patch[:,:,0],source_check)
                    check_patch[:,:,1] = np.multiply(check_patch[:,:,1],source_check)
                    check_patch[:,:,2] = np.multiply(check_patch[:,:,2],source_check)
                    # print ''
                    # print check_patch.shape
                    # print object_patch.shape
                    if check_patch.shape == object_patch.shape:
                        temp_ssd = cv2.absdiff(check_patch,object_patch)
                        temp_ssd = np.multiply(temp_ssd,temp_ssd)
                        temp_ssd = np.sum(temp_ssd)
                        if temp_ssd < ssd:
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
    print [I_sub_min,I_sub_max,J_sub_min,J_sub_max]
    pixel_region[I_min:I_max,J_min:J_max,:] = np.add(pixel_region[I_min:I_max,J_min:J_max,:],fill)
    pixel_source[I_min:I_max,J_min:J_max] = 1
    # Add subregion to the overall region
    # object_region[I_sub_min:I_sub_max,J_sub_min:J_sub_max,:] = np.add(object_region[I_sub_min:I_sub_max,J_sub_min:J_sub_max,:],pixel_region)
    object_region[I_sub_min:I_sub_max,J_sub_min:J_sub_max,:] = pixel_region
    # source_region[I_sub_min:I_sub_max,J_sub_min:J_sub_max] = np.add(source_region[I_sub_min:I_sub_max,J_sub_min:J_sub_max],pixel_source)
    # t = source_region > 1
    # source_region[t] = 1
    source_region[I_sub_min:I_sub_max,J_sub_min:J_sub_max] = pixel_source
    # source_region[I_min:I_max,J_min:J_max] = 1
    return source_region,object_region
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
def patch(img,region,k):
    object_region = np.array(img, copy=True)
    object_region[region[1]:region[3],region[0]:region[2],:] = 0
    source_region = np.full((img.shape[0],img.shape[1]), 1)
    source_region[region[1]:region[3],region[0]:region[2]] = 0
    Xmin = np.max([0,region[0]-k])
    Xmax = np.min([img.shape[1],region[2]+k])
    Ymin = np.max([0,region[1]-k])
    Ymax = np.min([img.shape[0],region[3]+k])
    source_region = source_region[Ymin:Ymax,Xmin:Xmax]
    object_region = object_region[Ymin:Ymax,Xmin:Xmax]
    return source_region,object_region
def isComplete(source_region):
    sum = np.sum(source_region)
    total = source_region.shape[0] * source_region.shape[1]
    if sum == total:
        return True
    else:
        return False
def removeStillObject(img,region,output):
    I = cv2.imread(img)
    kernel = 7
    k = 50
    source_region,object_region = patch(I,region,k)
    done = False
    while done == False:
        conf_level = confidence(object_region,source_region,kernel)
        source_region,object_region = fill(conf_level,source_region,object_region,kernel)
        done = isComplete(source_region)
        cv2.imwrite("fill.jpg",object_region)
        temp = np.multiply(source_region,255)
        cv2.imwrite("source.jpg",temp)
    I = fillPatch(I,object_region,region,k)
    cv2.imwrite("images/stillResults/final.jpg",I)
# removeStillObject("images/still/5-ball_small.jpg",[150,160,225,270],"images/stillResults/5-ball-final.jpg")
removeStillObject("images/still/20190317153942_IMG_0234.jpg",[11,230,33,244],"images/stillResults/couch-final.jpg")
