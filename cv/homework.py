import sys
import cv2 as cv
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    def hu(img,kernel):
        h1=img.shape[0]
        w1=img.shape[1]
        h2=kernel.shape[0]
        w2=kernel.shape[1]
        s=0
        img1=np.zeros((h1,w1))
        for i1 in range(h1):
            for j1 in range(w1):
                sum=0
                for i2 in range(h2):
                    for j2 in range(w2):
                        if i1-int(h2/2)+i2 < 0 or j1-int(w2/2)+j2 < 0 or i1-int(h2/2)+i2 > h1-1 or j1-int(w2/2)+j2 > w1-1:
                            s=0
                        else:
                            s=kernel[i2][j2]*img[i1-int(h2/2)+i2][j1-int(w2/2)+j2]
                        sum=sum+s
                img1[i1][j1]=sum
        return img1
    if img.ndim == 3:
        b,g,r=cv.split(img)
        b1=hu(b,kernel)
        g1=hu(g,kernel)
        r1=hu(r,kernel)
        merge=cv.merge([b1,g1,r1])
    if img.ndim == 2:
        merge=hu(img,kernel)
    return merge
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    height=kernel.shape[0]
    width=kernel.shape[1]
    dkernel=np.zeros((height,width))
    for i1 in range(height):
        for j1 in range(width):
            dkernel[i1][j1]=kernel[-i1-1][-j1-1]
    return cross_correlation_2d(img,dkernel)
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    k=np.zeros((height,width))
    s=2*(sigma**2)
    hcenter=height//2
    wcenter=width//2
    sum=0
    for i in range(height):
        for j in range(width):
            x=j-wcenter
            y=i-hcenter
            k[i][j]=np.exp(- (x**2 + y**2)/s)
            sum+=k[i][j]
    return k*(1/sum)
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kernel=gaussian_blur_kernel_2d(sigma,size,size)
    return convolve_2d(img,kernel)
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    limg=low_pass(img, sigma, size)
    return img-limg
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

if __name__ == '__main__':
    f = [1, 2, 1]
    l = [0, 1, 2, 3, 3, 3, 1, 3, 6]
    # without padding
    l_f = []
    for i_l in range(2, len(l)):
        re = 0
        for i_f in range(3):
            re = re + f[i_f] * l[i_l - i_f]
        l_f.append(re)
    print("without padding: ", l_f)
    # with padding
    l_f = []
    for i_l in range(len(l) + 2):
        re = 0
        for i_f in range(3):
            if i_l - i_f >= 0 and i_l - i_f < len(l):
                add = f[i_f] * l[i_l - i_f]
            else:
                add = 0
            re = re + add
        l_f.append(re)
    print("with padding 0: ", l_f)
