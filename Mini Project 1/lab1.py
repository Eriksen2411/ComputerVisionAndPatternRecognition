import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import math

##### Part 1: image preprossessing #####

def rgb2gray(img):
    """
    5 points
    Convert a colour image greyscale
    Use (R,G,B)=(0.299, 0.587, 0.114) as the weights for red, green and blue channels respectively
    :param img: numpy.ndarray (dtype: np.uint8)
    :return gray_image: numpy.ndarray (dtype:np.uint8)
    """
    if len(img.shape) != 3:
        print('RGB Image should have 3 channels')
        return
    
    ###Your code here###
    img_gray = img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114
    ###
    return img_gray

def convolute(img, kernal):
    y, x = img.shape
    img_pad_zeros = pad_zeros(img,1,1,1,1)

    kernal_flip = np.flipud(np.fliplr(kernal))    
    m, n = kernal_flip.shape
    
    new_img = np.zeros((y,x))

    for i in range(y):
            for j in range(x):
                new_img[i][j] = np.sum(img_pad_zeros[i:i+m, j:j+m]*kernal_flip)
    return new_img

def gray2grad(img):
    """
    5 points
    Estimate the gradient map from the grayscale images by convolving with Sobel filters (horizontal and vertical gradients) and Sobel-like filters (gradients oriented at 45 and 135 degrees)
    The coefficients of Sobel filters are provided in the code below.
    :param img: numpy.ndarray
    :return img_grad_h: horizontal gradient map. numpy.ndarray
    :return img_grad_v: vertical gradient map. numpy.ndarray
    :return img_grad_d1: diagonal gradient map 1. numpy.ndarray
    :return img_grad_d2: diagonal gradient map 2. numpy.ndarray
    """
    sobelh = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]], dtype = float)
    sobelv = np.array([[-1, -2, -1], 
                       [0, 0, 0], 
                       [1, 2, 1]], dtype = float)
    sobeld1 = np.array([[-2, -1, 0],
                        [-1, 0, 1],
                        [0,  1, 2]], dtype = float)
    sobeld2 = np.array([[0, -1, -2],
                        [1, 0, -1],
                        [2, 1, 0]], dtype = float)
    

    ###Your code here####
    img_grad_h = convolute(img, sobelh)
    img_grad_v = convolute(img, sobelv)
    img_grad_d1 = convolute(img, sobeld1)
    img_grad_d2 = convolute(img, sobeld2)
    ###
    return img_grad_h, img_grad_v, img_grad_d1, img_grad_d2

def pad_zeros(img, pad_height_bef, pad_height_aft, pad_width_bef, pad_width_aft):
    """
    5 points
    Add a border of zeros around the input images so that the output size will match the input size after a convolution or cross-correlation operation.
    e.g., given matrix [[1]] with pad_height_bef=1, pad_height_aft=2, pad_width_bef=3 and pad_width_aft=4, obtains:
    [[0 0 0 0 0 0 0 0]
    [0 0 0 1 0 0 0 0]
    [0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0]]
    :param img: numpy.ndarray
    :param pad_height_bef: int
    :param pad_height_aft: int
    :param pad_width_bef: int
    :param pad_width_aft: int
    :return img_pad: numpy.ndarray. dtype is the same as the input img. 
    """
    height, width = img.shape[:2]
    new_height, new_width = (height + pad_height_bef + pad_height_aft), (width + pad_width_bef + pad_width_aft)
    img_pad = np.zeros((new_height, new_width)) if len(img.shape) == 2 else np.zeros((new_height, new_width, img.shape[2]))

    ###Your code here###
    if len(img.shape) == 2:
        img_pad[pad_height_bef:pad_height_bef+height, pad_width_bef:pad_width_bef+width] = img
    else:
        img_pad[pad_height_bef:pad_height_bef+height, pad_width_bef:pad_width_bef+width,:] = img
    img_pad = img_pad.astype(img.dtype)
    ###
    return img_pad




##### Part 2: Normalized Cross Correlation #####
def normalized_cross_correlation(img, template):
    """
    10 points.
    Implement the cross-correlation operation in a naive 6 nested for-loops. 
    The 6 loops include the height, width, channel of the output and height, width and channel of the template.
    :param img: numpy.ndarray.
    :param template: numpy.ndarray.
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    template = template.astype(float)
    sum_template = np.sum(template)
    
    if len(template.shape) == 3:
        for k in range(template.shape[2]):
            template[:,:,k] /= sum_template
    else:
        template /= sum_template
    
    template_mag = np.sqrt(np.sum(template*template))

    img = img.astype(float)

    response = np.zeros((Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            window = img[i:i+Hk, j:j+Wk, :] 
            window_mag = np.sqrt(np.sum(window*window)) 
            for k in range(Hk):
                for l in range(Wk):
                    if len(template.shape) >= 3:
                        for m in range(template.shape[2]):
                            response[i,j] += template[k,l,m]*window[k,l,m]
                    else:
                        response[i,j] += template[k,l]*window[k,l]
            response[i,j] /= (window_mag*template_mag)
    ###
    return response


#Normalize a RGB kernel so that the sum of each color channel is 1
def normalized_a_kernel(kernel):
    kernel = kernel.astype(float)
    
    sum_kernel = np.sum(kernel)
    
    kernel /= np.abs(sum_kernel)
    return kernel


def normalized_cross_correlation_fast(img, template):
    """
    10 points.
    Implement the cross correlation with 3 nested for-loops. 
    The for-loop over the template is replaced with the element-wise multiplication between the kernel and the image regions.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    img = img.astype(float)
    template = normalized_a_kernel(template)
    template_mag = np.sqrt(np.sum(template*template))
    
    response = np.zeros((Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            # three other loops are replaced with element-wise matrix multiplication
            window = img[i:i+Hk, j:j+Wk]
            window_mag = np.sqrt(np.sum(window*window))
            response[i, j] = np.sum(template*window) / (window_mag*template_mag)  
    ###
    return response


def im2col(x, kernel):
    kernel_shape = kernel.shape[0]
    rows = []
    
    for row in range(x.shape[0] - kernel_shape + 1):
        for col in range(x.shape[1] - kernel_shape + 1):
            window = x[row: row + kernel_shape, col: col + kernel_shape]
            rows.append(window.flatten())
            
    return np.array(rows)

def color_im2col(img, kernel):
    if (len(img.shape) == 2 and len(kernel.shape) == 2):
        return im2col(img, kernel)

    if (len(img.shape) > 2):
        matrix_list = []
        for i in range(img.shape[2]):
            matrix_list.append(im2col(img[:,:,i], kernel))    
        return np.hstack(matrix_list)

def reshape_template(kernel):
    if (len(kernel.shape) == 2):
        return kernel.flatten()
    else:
        row_list = []
        for i in range(kernel.shape[2]):
            row_list.append(kernel[:,:,i].flatten())
        return np.hstack(row_list)


def normalized_cross_correlation_matrix(img, template):
    """
    10 points.
    Converts cross-correlation into a matrix multiplication operation to leverage optimized matrix operations.
    Please check the detailed instructions in the pdf file.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    img = img.astype(float)
    img_matrix = color_im2col(img, template)

    # Normalize the magnitude of the template
    template = normalized_a_kernel(template)

    #Reshape the template into a column     
    template_matrix = reshape_template(template)

    #Multiple the image matrix and the template column
    reshaped_output = np.matmul(img_matrix,template_matrix)

    #Compute the magnitude of each window from the image matrix
    img_window_mag = np.sum(img_matrix**2, axis=1)
    img_window_mag = np.sqrt(img_window_mag)
    
    #Compute the magnitude of template
    template_mag = np.sqrt(np.sum(template*template))

    #Scale the window magnitude with the maginitude of the template
    img_window_mag *= template_mag
    
    #Compute the final response and reshape it
    reshaped_output_normalized = reshaped_output / img_window_mag
    response = reshaped_output_normalized.reshape((Ho, Wo)) 
    ###
    return response


##### Part 3: Non-maximum Suppression #####

def non_max_suppression(response, suppress_range, threshold=None):
    """
    10 points
    Implement the non-maximum suppression for translation symmetry detection
    The general approach for non-maximum suppression is as follows:
	1. Set a threshold τ; values in X<τ will not be considered.  Set X<τ to 0.  
    2. While there are non-zero values in X
        a. Find the global maximum in X and record the coordinates as a local maximum.
        b. Set a small window of size w×w points centered on the found maximum to 0.
	3. Return all recorded coordinates as the local maximum.
    :param response: numpy.ndarray, output from the normalized cross correlation
    :param suppress_range: a tuple of two ints (H_range, W_range). 
                           the points around the local maximum point within this range are set as 0. In this case, there are 2*H_range*2*W_range points including the local maxima are set to 0
    :param threshold: int, points with value less than the threshold are set to 0
    :return res: a sparse response map which has the same shape as response
    """
    ###Your code here###
    if threshold == None:
        threshold = 0
        
    res = np.where(response > threshold, response, 0)
    
    while np.max(res) > 0:
        rows, columns = np.where(res == np.max(res))
        x = rows[0]
        y = columns[0]
        for j in range(-suppress_range[0], suppress_range[0]):
            for k in range(-suppress_range[1], suppress_range[1]):
                if ((x + j) in range(res.shape[0])) and ((y+k) in range(res.shape[1])):
                    res[x + j, y + k] = 0
        res[x,y] = -1

    res = np.where(res == -1, 1, 0)
    ###
    return res

##### Part 4: Question And Answer #####


def mean_subtracted_kernel(kernel):
    kernel = kernel.astype(float)
    mean_kernel = np.zeros(kernel.shape[2])
    for i in range(kernel.shape[2]):
        mean_kernel[i] = np.sum(kernel[:,:,i])
    mean_kernel = mean_kernel / (kernel.shape[0] * kernel.shape[1])
    
    new_kernel = kernel
    for i in range(kernel.shape[2]):
        new_kernel[:,:,i] = new_kernel[:,:,i] - mean_kernel[i]
    return new_kernel

def mean_img(img, Hk, Ho, Wk, Wo):
    mean_windows = np.zeros((Ho, Wo, img.shape[2]))
    for i in range(Ho):
        for j in range(Wo):
            window = img[i:i+Hk, j:j+Wk, :]
            mean_window = np.zeros(img.shape[2])
            for k in range(img.shape[2]):
                mean_window[k] = np.sum(window[:,:,k])
            mean_window /= (Hk * Wk)
            mean_windows[i,j] = mean_window
    return mean_windows

def normalized_cross_correlation_ms(img, template):
    """
    10 points
    Please implement mean-subtracted cross correlation which corresponds to OpenCV TM_CCOEFF_NORMED.
    For simplicty, use the "fast" version.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    mean_template = normalized_a_kernel(template)
    mean_template = mean_subtracted_kernel(mean_template)
    template_mag = np.sqrt(np.sum(mean_template*mean_template))
    
    pad_height_bef, pad_height_aft = template.shape[0] // 2 - (1 if template.shape[0] % 2 == 0 else 0), template.shape[0] // 2
    pad_width_bef, pad_width_aft = template.shape[1] // 2 - (1 if template.shape[1] % 2 == 0 else 0), template.shape[1] // 2

    img = img.astype(float)
    mean_windows = mean_img(img, Hk, Ho, Wk, Wo)

    response = np.zeros((Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            window = img[i:i+Hk, j:j+Wk, :]
            window = window - mean_windows[i,j]
            window_mag = np.sqrt(np.sum(window*window))
            
            response[i, j] = np.sum(mean_template*window) / (window_mag*template_mag) 
    ###
    return response






###############################################
"""Helper functions: You should not have to touch the following functions.
"""
def read_img(filename):
    '''
    Read HxWxC image from the given filename
    :return img: numpy.ndarray, size (H, W, C) for RGB. The value is between [0, 255].
    '''
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_imgs(imgs, titles=None):
    '''
    Display a list of images in the notebook cell.
    :param imgs: a list of images or a single image
    '''
    if isinstance(imgs, list) and len(imgs) != 1:
        n = len(imgs)
        fig, axs = plt.subplots(1, n, figsize=(15,15))
        for i in range(n):
            axs[i].imshow(imgs[i], cmap='gray' if len(imgs[i].shape) == 2 else None)
            if titles is not None:
                axs[i].set_title(titles[i])
    else:
        img = imgs[0] if (isinstance(imgs, list) and len(imgs) == 1) else imgs
        plt.figure()
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)

def show_img_with_squares(response, img_ori=None, rec_shape=None):
    '''
    Draw small red rectangles of size defined by rec_shape around the non-zero points in the image.
    Display the rectangles and the image with rectangles in the notebook cell.
    :param response: numpy.ndarray. The input response should be a very sparse image with most of points as 0.
                     The response map is from the non-maximum suppression.
    :param img_ori: numpy.ndarray. The original image where response is computed from
    :param rec_shape: a tuple of 2 ints. The size of the red rectangles.
    '''
    response = response.copy()
    if img_ori is not None:
        img_ori = img_ori.copy()
    H, W = response.shape[:2]
    if rec_shape is None:
        h_rec, w_rec = 25, 25
    else:
        h_rec, w_rec = rec_shape

    xs, ys = response.nonzero()
    for x, y in zip(xs, ys):
        response = cv2.rectangle(response, (y - h_rec//2, x - w_rec//2), (y + h_rec//2, x + w_rec//2), (255, 0, 0), 2)
        if img_ori is not None:
            img_ori = cv2.rectangle(img_ori, (y - h_rec//2, x - w_rec//2), (y + h_rec//2, x + w_rec//2), (0, 255, 0), 2)
        
    if img_ori is not None:
        show_imgs([response, img_ori])
    else:
        show_imgs(response)

