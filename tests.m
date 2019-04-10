function [ssimVal, mse] = tests(ground,image)
    mse = immse(ground,image);
    ssimVal = ssim(image,ground);
end