import math
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from matplotlib import cm


##################### TASK 1 ###################

# 1.1 IMPLEMENT
def make_gaussian_kernel(ksize, sigma):
    '''
    Implement the simplified Gaussian kernel below:
    k(x,y)=exp(((x-x_mean)^2+(y-y_mean)^2)/(-2sigma^2))
    Make Gaussian kernel be central symmentry by moving the 
    origin point of the coordinate system from the top-left
    to the center. Please round down the mean value. In this assignment,
    we define the center point (cp) of even-size kernel to be the same as that of the nearest
    (larger) odd size kernel, e.g., cp(4) to be same with cp(5).
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    '''
    
    # YOUR CODE HERE
    kernel = np.zeros((ksize, ksize))

    i_mean =  ksize // 2
    j_mean = ksize // 2

    for i in range(kernel.shape[0]):    
        for j in range(kernel.shape[1]):
            kernel[i,j]=math.exp(((i-i_mean)*(i-i_mean)+(j-j_mean)*(j-j_mean))/(-2*sigma*sigma))

    # END

    return kernel / kernel.sum()

# GIVEN
def cs4243_filter(image, kernel):
    """
    Fast version of filtering algorithm.
    Pre-extract all the regions of kernel size,
    and obtain a matrix of shape (Hi*Wi, Hk*Wk), also reshape the flipped
    kernel to be of shape (Hk*Wk, 1), then do matrix multiplication, and reshape back
    to get the final output image. 
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    """
    def cs4243_rotate180(kernel):
        kernel = np.flip(np.flip(kernel, 0),1)
        return kernel
    
    def img2col(input, h_out, w_out, h_k, w_k, stride):
        h, w = input.shape
        out = np.zeros((h_out*w_out, h_k*w_k))
        
        convwIdx = 0
        convhIdx = 0
        for k in range(h_out*w_out):
            if convwIdx + w_k > w:
                convwIdx = 0
                convhIdx += stride
            out[k] = input[convhIdx:convhIdx+h_k, convwIdx:convwIdx+w_k].flatten()
            convwIdx += stride
        return out
    
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    if Hk % 2 == 0 or Wk % 2 == 0:
        raise ValueError
        
    hkmid = Hk//2
    wkmid = Wk//2

    image = cv2.copyMakeBorder(image, hkmid, hkmid, wkmid, wkmid, cv2.BORDER_REFLECT)
    filtered_image = np.zeros((Hi, Wi))
    kernel = cs4243_rotate180(kernel)
    col = img2col(image, Hi, Wi, Hk, Wk, 1)
    kernel_flatten = kernel.reshape(Hk*Wk, 1)
    output = col @ kernel_flatten 
    filtered_image = output.reshape(Hi, Wi)
    
    return filtered_image

# GIVEN
def cs4243_blur(img, gaussian_kernel, display=True):
    '''
    Performing Gaussian blurring on an image using a Gaussian kernel.
    :param img: input image
    :param gaussian_kernel: gaussian kernel
    :return blurred_img: blurred image
    '''

    blurred_img = cs4243_filter(img, gaussian_kernel)

    if display:

        fig1, axes_array = plt.subplots(1, 2)
        fig1.set_size_inches(8,4)
        image_plot = axes_array[0].imshow(img,cmap=plt.cm.gray) 
        axes_array[0].axis('off')
        axes_array[0].set(title='Original Image')
        image_plot = axes_array[1].imshow(blurred_img,cmap=plt.cm.gray)
        axes_array[1].axis('off')
        axes_array[1].set(title='Filtered Image')
        plt.show()  
    return blurred_img

# 2 IMPLEMENT
def estimate_gradients(original_img, display=True):
    '''
    Compute gradient orientation and magnitude for the input image.
    Perform the following steps:
    
    1. Compute dx and dy, responses to the vertical and horizontal Sobel kernel. Make use of the cs4243_filter function.
    
    2. Compute the gradient magnitude which is equal to sqrt(dx^2 + dy^2) 
    
    3. Compute the gradient orientation using the following formula:
        gradient = atan2(dy/dx)
        
    You may want to divide the original image pixel value by 255 to prevent overflow.
    
    Note that our axis choice is as follows:
            --> y
            |   
            ??? x    
    Where img[x,y] denotes point on the image at coordinate (x,y)
    
    :param original_img: original grayscale image
    :return d_mag: gradient magnitudes matrix
    :return d_angle: gradient orientation matrix (in radian)
    '''
    
    dx = None
    dy = None
    d_mag = None
    d_angle = None
    
    # YOUR CODE HERE 
    '''
    HINT:
    In the lecture, 
    
    Sx =  1  0 -1
          2  0 -2
          1  0 -1
          
    Sy =  1  2  1
          0  0  0
         -1 -2 -1
         
    Here:
    
    Kx = [[ 1,  2,  1],
          [ 0,  0,  0],
          [-1, -2, -1]]
    
    Ky = [[ 1,  0, -1],
          [ 2,  0, -2],
          [ 1,  0, -1]]
 
    This is because x direction is the downward line.
    '''

    Kx = np.array([[-1, -2, -1],
          [0, 0, 0],
          [1, 2, 1]])
    
    Ky = np.array([[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]])

    original_img /= 255
    dx = cs4243_filter(original_img, Kx)
    dy = cs4243_filter(original_img, Ky)

    d_mag = (dx*dx + dy*dy)
    d_mag = np.sqrt(d_mag)

    d_angle = np.arctan2(dy, dx)

    # END
    if display:
    
        fig2, axes_array = plt.subplots(1, 4)
        fig2.set_size_inches(16,4)
        image_plot = axes_array[0].imshow(d_mag, cmap='gray')  
        axes_array[0].axis('off')
        axes_array[0].set(title='Gradient Magnitude')

        image_plot = axes_array[1].imshow(dx, cmap='gray')  
        axes_array[1].axis('off')
        axes_array[1].set(title='dX')

        image_plot = axes_array[2].imshow(dy, cmap='gray')  
        axes_array[2].axis('off')
        axes_array[2].set(title='dY')

        image_plot = axes_array[3].imshow(d_angle, cmap='gray')  
        axes_array[3].axis('off')
        axes_array[3].set(title='Gradient Direction')
        plt.show()
    
    return d_mag, d_angle

# 3a IMPLEMENT

def is_local_max(d_mag, d_angle_180, i, j):
    angle = d_angle_180[i, j]
    pixel = d_mag[i, j]
    if (-22.5 <= angle and angle < 22.5) or (157.5 <= angle) or (angle < -157.5):
        index_1 = (i + 1, j)
        index_2 = (i - 1, j)
    elif(22.5 <= angle and angle < 67.5) or (-157.5 <= angle and angle < -112.5):
        index_1 = (i + 1, j + 1)
        index_2 = (i - 1, j - 1)
    elif(67.5 <= angle and angle < 112.5) or (-112.5 <= angle and angle < -67.5):
        index_1 = (i, j + 1)
        index_2 = (i, j - 1)
    elif(112.5 <= angle and angle < 157.5) or (-67.5 <= angle and angle < -22.5):
        index_1 = (i + 1, j - 1)
        index_2 = (i - 1, j + 1)

    #if index_1 is out of range 
    if (index_1[0] <= 0) or (index_1[0] >= d_mag.shape[0]) or (index_1[1] <= 0) or (index_1[1] >= d_mag.shape[1]):
        index_1 = (i, j)
    
    #if index_2 is out of range 
    if (index_2[0] <= 0) or (index_2[0] >= d_mag.shape[0]) or (index_2[1] <= 0) or (index_2[1] >= d_mag.shape[1]):
        index_2 = (i, j)
    
    pixel_1 = d_mag[index_1]
    pixel_2 = d_mag[index_2]

    if (np.max([pixel, pixel_1, pixel_2]) == pixel):
        return True
    else:
        return False
    

def non_maximum_suppression(d_mag, d_angle, display=True):
    '''
    Perform non-maximum suppression on the gradient magnitude matrix without interpolation.
    Split the range -180?? ~ 180?? into 8 even ranges. For each pixel, determine which range the gradient
    orientation belongs to and pick the corresponding two pixels from the adjacent eight pixels surrounding 
    that pixel. Keep the pixel if its value is larger than the other two.
    Do note that the coordinate system is as below and angular measurements are clockwise.
    ----------??? y  
    |
    |
    |
    |        x X x
    ??? x       \|/   
             x-o-x  
              /|\    
             x X x 
         -22.5 0 22.5
         
    For instance, in the example above if the orientation at the coordinate of interest (x,y) is 20??, it belongs to the -22.5??~22.5?? range, and the two pixels to be compared with are at (x+1,y) and (x-1,y) (aka the two big X's). If the angle was instead 40??, it belongs to the 22.5??-67.5?? and the two pixels we need to consider will be (x+1, y+1) and (x-1,y-1)
    
    There are only 4 sets of offsets: (0,1), (1,0), (1,1), and (1,-1), since to find the second pixel offset you just need 
    to multiply the first tuple by -1.
    
    :param d_mag: gradient magnitudes matrix
    :param d_angle: gradient orientation matrix (in radian)
    :return out: non-maximum suppressed image
    '''

    out = np.zeros(d_mag.shape, d_mag.dtype)
    # Change angles to degrees to improve quality of life
    d_angle_180 = d_angle * 180/np.pi
    

 
    # YOUR CODE HERE
    
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if is_local_max(d_mag, d_angle_180, i, j):
                out[i,j] = d_mag[i,j]
                
    # END
    if display:
        _ = plt.figure(figsize=(10,10))
        plt.imshow(out)
        plt.title("Suppressed image (without interpolation)")
    
    return out

# 3b IMPLEMENT

def is_local_max_interpol(d_mag, d_angle_180, i, j):
    angle = d_angle_180[i, j]
    pixel = d_mag[i, j]

    if (0 <= angle and angle < 45) or (-180 <= angle and angle < -135) or (angle == 180):
        index_11 = (i + 1, j)
        index_12 = (i + 1, j + 1)
        index_21 = (i - 1, j)
        index_22 = (i - 1, j - 1)
        angle_fix = angle
    elif(45 <= angle and angle < 90) or (-135 <= angle and angle < -90):
        index_11 = (i, j + 1)
        index_12 = (i + 1, j + 1)
        index_21 = (i, j - 1)
        index_22 = (i - 1, j - 1)
        angle_fix = 90 - angle
    elif(90 <= angle and angle < 135) or (-90 <= angle and angle < -45):
        index_11 = (i, j + 1)
        index_12 = (i - 1, j + 1)
        index_21 = (i, j - 1)
        index_22 = (i + 1, j - 1)
        angle_fix = angle - 90
    elif(135 <= angle and angle < 180) or (-45 <= angle and angle < 0):
        index_11 = (i - 1, j)
        index_12 = (i - 1, j + 1)
        index_21 = (i + 1, j)
        index_22 = (i + 1, j - 1)
        angle_fix = 180 - angle

    #if index_11 is out of range 
    if (index_11[0] <= 0) or (index_11[0] >= d_mag.shape[0]) or (index_11[1] <= 0) or (index_11[1] >= d_mag.shape[1]):
        index_11 = (i, j)

    #if index_12 is out of range 
    if (index_12[0] <= 0) or (index_12[0] >= d_mag.shape[0]) or (index_12[1] <= 0) or (index_12[1] >= d_mag.shape[1]):
        index_12 = (i, j)

    #if index_21 is out of range 
    if (index_21[0] <= 0) or (index_21[0] >= d_mag.shape[0]) or (index_21[1] <= 0) or (index_21[1] >= d_mag.shape[1]):
        index_21 = (i, j)

    #if index_22 is out of range 
    if (index_22[0] <= 0) or (index_22[0] >= d_mag.shape[0]) or (index_22[1] <= 0) or (index_22[1] >= d_mag.shape[1]):
        index_22 = (i, j)

    pixel_1 = d_mag[index_11] + math.tan(angle_fix  / 180.0 * np.pi) * (d_mag[index_12]  - d_mag[index_11])
    pixel_2 = d_mag[index_21] + math.tan(angle_fix  / 180.0 * np.pi) * (d_mag[index_22]  - d_mag[index_21])

    if (np.max([pixel, pixel_1, pixel_2]) == pixel):
        return True
    else:
        return False

def non_maximum_suppression_interpol(d_mag, d_angle, display=True):
    '''
    Perform non-maximum suppression on the gradient magnitude matrix with interpolation.
    :param d_mag: gradient magnitudes matrix
    :param d_angle: gradient orientation matrix (in radian)
    :return out: non-maximum suppressed image
    '''

    out = np.zeros(d_mag.shape, d_mag.dtype)
    d_angle_180 = d_angle * 180/np.pi
    
    # YOUR CODE HERE

    for i in range(out.shape[0]):    
        for j in range(out.shape[1]):
            if (is_local_max_interpol(d_mag, d_angle_180, i, j)):
                out[i, j] = d_mag[i, j]

    # END
    if display:
        _ = plt.figure(figsize=(10,10))
        plt.imshow(out, cmap='gray')
        plt.title("Suppressed image (with interpolation)")
    
    return out

# 4 IMPLEMENT
def double_thresholding(inp, perc_weak=0.1, perc_strong=0.3, display=True):
    '''
    Perform double thresholding. Use on the output of NMS. The high and low thresholds are computed as follow:
    
    delta = max_val - min_val
    high_threshold = min_val + perc_strong * delta 
    low_threshold = min_val + perc_weak * delta
    
    perc_weak being 0 is possible
    Do note that the return edge images should be binary (0-1 or True-False)
    :param inp: numpy.ndarray
    :param perc_weak: value to determine low threshold
    :param perc_strong: value to determine high threshold
    :return weak_edges, strong_edges: binary edge images
    '''
    weak_edges = strong_edges = None
    
    # YOUR CODE HERE
    max_val = np.max(inp)
    min_val = np.min(inp)

    delta = max_val - min_val
    high_threshold = min_val + perc_strong * delta 
    low_threshold = min_val + perc_weak * delta

    strong_edges = np.where(inp > high_threshold, 1, 0)
    weak_edges = np.where((inp > low_threshold) & (inp < high_threshold), 1, 0)
    
    max_mag = np.max(inp)
    min_mag = np.min(inp)
    
    high_thres = min_mag + perc_strong*(max_mag - min_mag)
    low_thres = min_mag + perc_weak*(max_mag - min_mag)
    
    strong_edges = np.where(inp > high_thres, 1, 0)
    weak_edges = np.where((inp > low_thres) & (inp < high_thres), 1, 0)
    
    # END
    
    if display:

        fig2, axes_array = plt.subplots(1, 2)
        fig2.set_size_inches(10,5)
        image_plot = axes_array[0].imshow(strong_edges, cmap='gray')  
        axes_array[0].axis('off')
        axes_array[0].set(title='Strong ')

        image_plot = axes_array[1].imshow(weak_edges, cmap='gray')  
        axes_array[1].axis('off')
        axes_array[1].set(title='Weak')
        
    return weak_edges, strong_edges

# 5 IMPLEMENT
def edge_linking(weak, strong, n=200, display=True):
    '''
    Perform edge-linking on two binary weak and strong edge images. 
    A weak edge pixel is linked if any of its eight surrounding pixels is a strong edge pixel.
    You may want to avoid using loops directly due to the high computational cost. One possible trick is to generate
    8 2D arrays from the strong edge image by offseting and sum them together; entries larger than 0 mean that at least one surrounding
    pixel is a strong edge pixel (otherwise the sum would be 0).
    
    You may also want to limit the number of iterations (test with 10-20 iterations first to check your implementation speed), and use a stopping condition (stop if no more pixel is added to the strong edge image).
    Also, when a weak edge pixel is added to the strong set, remember to remove it.


    :param weak: weak edge image (binary)
    :param strong: strong edge image (binary)
    :param n: maximum number of iterations
    :return out: final edge image
    '''
    assert weak.shape == strong.shape, \
        "Weak and strong edge image have to have the same dimension"
    out = None
    
    # YOUR CODE HERE
    
    count = 0
    for i in range(n):
        for j in range(strong.shape[0]):
            for k in range(strong.shape[1]):
                if strong[j, k] == 1:
                    j1 = j - 1 if j - 1 >= 0 else j
                    j2 = j + 2 if j + 2 <= weak.shape[0] else j + 1
                    k1 = k - 1 if k - 1 >= 0 else k
                    k2 = k + 2 if k + 2 <= weak.shape[1] else k + 1
                    weak_surrounding = weak[j1:j2, k1:k2]
                    count += weak_surrounding.sum()
                    strong_surrounding = strong[j1:j2,k1:k2]
                    strong_surrounding = np.where(weak_surrounding == 1, 1, strong_surrounding)
                    strong[j1:j2,k1:k2] = strong_surrounding
                    weak[j1:j2, k1:k2] = np.zeros((j2-j1, k2-k1))
        if count == 0:
            break
            
    out = strong
    
    # END
    if display:
        _ = plt.figure(figsize=(10,10))
        plt.imshow(out)
        plt.title("Edge image")
    return out

##################### TASK 2 ######################

# 1/2/3 IMPLEMENT
def hough_vote_lines(img):
    '''
    Use the edge image to vote for 2 parameters: distance and theta
    Beware of our coordinate convention.

    You may find the np.linspace function useful.

    :param img: edge image
    :return A: accumulator array
    :return distances: distance values array
    :return thetas: theta values array
    '''
    # YOUR CODE HERE
    
    max_distance = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    
    distances = np.arange(-max_distance, max_distance + 1, 1)
    
    angle_interval = np.pi/180
    thetas = np.arange(0, np.pi, angle_interval)
        
    A = np.zeros((len(distances), len(thetas)))
   
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] == 1:
                for t in range(len(thetas)):
                    theta = thetas[t]
                    distance = i*np.cos(theta) + j*np.sin(theta)
                    r = int(np.floor(distance - distances[0]))
                    A[r,t]+=1
                    
    # END
            
    return A, distances, thetas

# 4 GIVEN
from skimage.feature import peak_local_max
def find_peak_params(hspace, params_list,  window_size=1, threshold=0.5):
    '''
    Given a Hough space and a list of parameters range, compute the local peaks
    aka bins whose count is larger max_bin * threshold. The local peaks are computed
    over a space of size (2*window_size+1)^(number of parameters).

    Also include the array of values corresponding to the bins, in descending order.
    
    e.g.
    Suppose for a line detection case, you get the following output:
    [
    [122, 101, 93],
    [3,   40,  21],
    [0,   1.603, 1.605]
    ]
    This means that the local maxima with the highest vote gets a vote score of 122, and the corresponding parameter value is distance=3, 
    theta = 0.
    '''
    assert len(hspace.shape) == len(params_list), \
        "The Hough space dimension does not match the number of parameters"
    for i in range(len(params_list)):
        assert hspace.shape[i] == len(params_list[i]), \
            f"Parameter length does not match size of the corresponding dimension:{len(params_list[i])} vs {hspace.shape[i]}"
    peaks_indices = peak_local_max(hspace.copy(), exclude_border=False, threshold_rel=threshold, min_distance=window_size)
    peak_values = np.array([hspace[tuple(peaks_indices[j])] for j in range(len(peaks_indices))])
    res = []
    res.append(peak_values)
    for i in range(len(params_list)):
        res.append(params_list[i][peaks_indices.T[i]])
    return res


##################### TASK 3 ######################

# 1/2/3 IMPLEMENT
from skimage.draw import circle_perimeter
def hough_vote_circles(img, radius = None):
    '''
    Use the edge image to vote for 3 parameters: circle radius and circle center coordinates.
    We also accept a range of radii to save computation costs. If the radius range is not given, it is default to
    [3, diagonal of the circle]. This parameter is very useful for your experimentation later on (e.g. if there are only large circles then you don't have to keep R_min very small).
    
    Hint: You can use the function circle_perimeter to make a circular mask. Center the mask over the accumulator array and increment the array. In this case, you will have to pad the accumulator array first, and clip it afterwards. 
    Remember that the return accumulator array should have matching dimension with the lengths of the parameter ranges. 
    
    The dimensions of the accumulator array should be in this order: radius, x-coordinate, y-coordinate.

    :param img: edge image
    :param radius: min radius, max radius
    :return A: accumulator array
    :return R: radius values array
    :return X: x-coordinate values array
    :return Y: y-coordinate values array
    '''
    
    # Check the radius range
    h, w = img.shape[:2]    
    if radius == None:
        R_max = np.hypot(h,w)
        R_min = 3
    else:
        [R_min,R_max] = radius
    
    # YOUR CODE HERE


    R_max = int(R_max)
    accumulator = np.zeros((R_max - R_min + 1, h, w))
    for radius_i in range(R_min, R_max, 1):
        accumulator_pad = np.zeros((h + 2*radius_i, w + 2*radius_i))
        rr_0, cc_0 = circle_perimeter(0, 0, radius_i)
        for i in range(h):
            for j in range(w):
                if (img[i, j] == 1):
                    rr = rr_0 + i + radius_i
                    cc = cc_0 + j + radius_i
                    accumulator_pad[rr, cc] += 1/radius_i

        accumulator[radius_i - R_min, :, :] = accumulator_pad[radius_i:radius_i + h, radius_i: radius_i + w]

    A = accumulator
    R = np.arange(R_min, R_max + 1, 1)
    X = np.arange(0, h, 1)
    Y = np.arange(0, w, 1)
    # END
   
    return A, R, X, Y

##################### TASK 4 ######################

# IMPLEMENT
def hough_vote_circles_grad(img, d_angle, radius = None):
    '''
    Use the edge image to vote for 3 parameters: circle radius and circle center coordinates.
    We also accept a range of radii to save computation costs. If the radius range is not given, it is default to
    [3, diagonal of the circle].
    This time, gradient information is used to avoid casting too many unnecessary votes.
    
    Remember that for a given pixel, you need to cast two votes along the orientation line. One in the positive direction, the other in
    negative direction.
    
    :param img: edge image
    :param d_angle: corresponding gradient orientation matrix
    :param radius: min radius, max radius
    :return A: accumulator array
    :return R: radius values array
    :return X: x-coordinate values array
    :return Y: y-coordinate values array
    '''
    # Check the radius range
    h, w = img.shape[:2]    
    if radius == None:
        R_max = np.hypot(h,w)
        R_min = 3
    else:
        [R_min,R_max] = radius
    
    # YOUR CODE HERE
    
    R_max = int(R_max)
    
    interval = 1 #use this to change the interval/bin_size
    
    R = np.arange(R_min, R_max + 1, interval)
    X = np.arange(0, h, interval)
    Y = np.arange(0, w, interval)
    
    accumulator = np.zeros((R.shape[0], X.shape[0], Y.shape[0]))
    for radius_i in R:
        for i in range(0, h, 1):
            for j in  range(0, w, 1):
                if (img[i, j] == 1):
                    angle = d_angle[i,j]
                    cos_angle = np.cos(angle)
                    sin_angle = np.sin(angle)
                    a1 = i + radius_i*cos_angle
                    b1 = j + radius_i*sin_angle
                    a2 = i - radius_i*cos_angle
                    b2 = j - radius_i*sin_angle
                    if (a1 >= 0 and a1 < h and b1 >= 0 and b1 < w):
                        accumulator[(int) ((radius_i - R_min) // interval), (int) (a1 // interval), (int) (b1 // interval)] += 1
                    if (a2 >= 0 and a2 < h and b2 >= 0 and b2 < w):   
                        accumulator[(int) ((radius_i - R_min) // interval), (int) (a2 // interval), (int) (b2 // interval)] += 1
    A = accumulator

    # END
    return A, R, X, Y

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

def draw_lines(hspace, dists, thetas, hs_maxima, file_path):
    im_c = read_img(file_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(im_c, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    angle_step = 0.5 * np.diff(thetas).mean()
    d_step = 0.5 * np.diff(dists).mean()
    bounds = [np.rad2deg(thetas[0] - angle_step),
              np.rad2deg(thetas[-1] + angle_step),
              dists[-1] + d_step, dists[0] - d_step]

    ax[1].imshow(np.log(1 + hspace), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(im_c, cmap=cm.gray)
    ax[2].set_ylim((im_c.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    # You may want to change the codes below if you use a different axis choice.
    for _, dist, angle in zip(*hs_maxima):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((y0, x0), slope=np.tan(np.pi-angle))

    plt.tight_layout()
    plt.show()

def draw_circles(local_maxima, file_path, title):
    img = cv2.imread(file_path)
    fig = plt.figure(figsize=(7,7))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    circle = []
    for _,r,x,y in zip(*local_maxima):
        circle.append(plt.Circle((y,x),r,color=(1,0,0),fill=False))
        fig.add_subplot(111).add_artist(circle[-1])
    plt.title(title)    
    plt.show()