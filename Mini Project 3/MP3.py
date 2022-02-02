import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_filter
import math

### REMOVE THIS

from utils import pad, unpad

import cv2
_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

##################### PART 1 ###################

# 1.1 IMPLEMENT
def harris_corners(img, window_size=3, k=0.04):
    '''
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the functions filters.sobel_v filters.sobel_h & scipy.ndimage.filters.convolve, 
        which are already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    '''

    H, W= img.shape
    window = np.ones((window_size, window_size))
    response = np.zeros((H, W))

    # YOUR CODE HERE
    
    grad_h_map = filters.sobel_h(img)
    grad_v_map = filters.sobel_v(img)
    
    Iy_2 = grad_h_map*grad_h_map
    Ix_Iy = grad_h_map*grad_v_map
    Ix_2 = grad_v_map*grad_v_map
    
    uniform_window = np.ones((window_size, window_size))
    
    A = convolve(Ix_2, uniform_window, mode='constant', cval=0.0)
    B = convolve(Ix_Iy, uniform_window, mode='constant', cval=0.0)
    C = convolve(Iy_2, uniform_window, mode='constant', cval=0.0)
    
    det_H = A*C - B*B
    trace_H = A + C
    
    R = det_H - (trace_H*trace_H)*k
    
    response = R
    
    # END        
    return response

# 1.2 IMPLEMENT
def naive_descriptor(patch):
    '''
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.

    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    '''
    feature = []
    ### YOUR CODE HERE
    
    stab_const = 0.0001
    feature = patch.flatten().astype('float')
    mean_val = np.mean(feature)
    std_val = np.std(feature)
    feature -= mean_val
    feature /= (std_val + stab_const)

    ### END YOUR CODE

    return feature

# GIVEN
def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    '''
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (x, y) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    '''

    image.astype(np.float32)
    desc = []
    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[np.max([0,y-(patch_size//2)]):y+((patch_size+1)//2),
                      np.max([0,x-(patch_size//2)]):x+((patch_size+1)//2)]
      
        desc.append(desc_func(patch))
   
    return np.array(desc)

# GIVEN
def make_gaussian_kernel(ksize, sigma):
    '''
    Good old Gaussian kernel.
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    '''

    ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
    yy, xx = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(yy) + np.square(xx)) / np.square(sigma))

    return kernel / kernel.sum()


# 1.2 IMPLEMENT
def simple_sift(patch):
    '''
    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each length of 16/4=4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    Use the gradient orientation to determine the bin, and the gradient magnitude * weight from
    the Gaussian kernel as vote weight.

    Args:
        patch: grayscale image patch of shape (h, w)

    Returns:
        feature: 1D array of shape (128, )
    '''
    
    # You can change the parameter sigma, which has been default to 3
    weights = np.flipud(np.fliplr(make_gaussian_kernel(patch.shape[0],3)))
    
    histogram = np.zeros((4,4,8))
    
    # YOUR CODE HERE
    
    grad_y_map = filters.sobel_h(patch)
    grad_x_map = filters.sobel_v(patch)
    
    grad_mag_map = np.sqrt(grad_y_map**2 + grad_x_map**2)
    grad_ori_map = np.arctan2(grad_y_map, grad_x_map) / np.pi * 180
    grad_ori_map = np.where(grad_ori_map >= 0, grad_ori_map, 360 + grad_ori_map)
    grad_ori_map = np.where(grad_ori_map < 360, grad_ori_map, 0)
    
    grad_weight_map = grad_mag_map*weights
    
    for i in range(4):
        for j in range(4):
            cell_weight = grad_weight_map[i*4: i*4 + 4, j*4: j*4 + 4]
            cell_ori = grad_ori_map[i*4: i*4 + 4, j*4: j*4 + 4]
            for idx, ori in np.ndenumerate(cell_ori):
                k, l = idx
                histogram[i,j,int(ori//45)] += cell_weight[k,l]
    
    feature = histogram.reshape((128,))
    feature = feature / np.sqrt(np.sum(feature**2))
  
    # END
    return feature

# 1.3 IMPLEMENT
def top_k_matches(desc1, desc2, k=2):
    '''
    Compute the Euclidean distance between each descriptor in desc1 versus all descriptors in desc2 (Hint: use cdist).
    For each descriptor Di in desc1, pick out k nearest descriptors from desc2, as well as the distances themselves.
    Example of an output of this function:
    
        [(0, [(18, 0.11414082134194799), (28, 0.139670625444803)]),
         (1, [(2, 0.14780585099287238), (9, 0.15420019834435536)]),
         (2, [(64, 0.12429203239414029), (267, 0.1395765079352806)]),
         ...<truncated>
    '''
    match_pairs = []
    
    # YOUR CODE HERE
    
    distances = cdist(desc1, desc2, 'euclidean')
    for i in range(desc1.shape[0]):
        sorted_index = np.argsort(distances[i])
        two_closest_index = sorted_index[:k]
        two_closest_list = []
        for j in range(k):
            two_closest_list.append((two_closest_index[j], distances[i, two_closest_index[j]]))
        match_pairs.append((i, two_closest_list))
  
    # END
    return match_pairs

# 1.3 IMPLEMENT
def ratio_test_match(desc1, desc2, match_threshold):
    '''
    Match two set of descriptors using the ratio test.
    Output should be a numpy array of shape (k,2), where k is the number of matches found. 
    In the following sample output:
        array([[  3,   0],
               [  5,  30],
               [ 11,   9],
               [ 18,   7],
               [ 24,   5],
               [ 30,  17],
               [ 32,  24],
               [ 46,  23], ... <truncated>
              )
              
        desc1[3] is matched with desc2[0], desc1[5] is matched with desc2[30], and so on.
    
    All other match functions will return in the same format as does this one.
    
    '''
    match_pairs = []
    top_2_matches = top_k_matches(desc1, desc2)
    # YOUR CODE HERE
    
    for pairs in top_2_matches:
        distance_1 = pairs[1][0][1]
        distance_2 = pairs[1][1][1]
        
        if distance_1 / distance_2 < match_threshold:
            match_pairs.append([pairs[0], pairs[1][0][0]])
            
    # END
    # Modify this line as you wish
    match_pairs = np.array(match_pairs)
    return match_pairs

# GIVEN
def compute_cv2_descriptor(im, method=cv2.xfeatures2d.SIFT_create()):
    '''
    Detects and computes keypoints using one of the implementations in OpenCV
    You can use:
        cv2.xfeatures2d.SIFT_create()

    Do note that the keypoints coordinate is (col, row)-(x,y) in OpenCV. We have changed it to (row,col)-(y,x) for you. (Consistent with out coordinate choice)
    '''
    kpts, descs = method.detectAndCompute(im, None)
    
    keypoints = np.array([(kp.pt[1],kp.pt[0]) for kp in kpts])
    angles = np.array([kp.angle for kp in kpts])
    sizes = np.array([kp.size for kp in kpts])
    
    return keypoints, descs, angles, sizes

##################### PART 2 ###################

# GIVEN
def transform_homography(src, h_matrix, getNormalized = True):
    '''
    Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    '''
    transformed = None

    input_pts = np.insert(src, 2, values=1, axis=1)
    transformed = np.zeros_like(input_pts)
    transformed = h_matrix.dot(input_pts.transpose())
    if getNormalized:
        transformed = transformed[:-1]/transformed[-1]
    transformed = transformed.transpose().astype(np.float32)
    
    return transformed

# 2.1 IMPLEMENT
def normalize_transform(points):
    mean, std = np.mean(points, 0), np.std(points, 0)
    std = std / np.sqrt(2)
    
    transform_matrix = np.array([[1/std[0], 0, -mean[0]/std[0]],
                                 [0, 1/std[1], -mean[1]/std[1]],
                                 [0, 0, 1]])
    points = np.dot(transform_matrix, np.concatenate((points.T, np.ones((1, points.shape[0])))))
    
    points = points[0:2].T
    
    return points, transform_matrix

def get_matrix_A(points1, points2, number_of_points):
    A = []
    for i in range(number_of_points):
        x1, y1 = points1[i, 0], points1[i, 1]
        x2, y2 = points2[i, 0], points2[i, 1]
        A.append([-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])
        
    return np.asarray(A)

def compute_homography(src, dst):
    '''
    Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    '''
    h_matrix = np.eye(3, dtype=np.float64)
  
    # YOUR CODE HERE
    
    src_norm, src_transform = normalize_transform(src)
    dst_norm, dst_transform = normalize_transform(dst)
    
    A = get_matrix_A(src_norm, dst_norm, src_norm.shape[0])
    
    U, S, V_t = np.linalg.svd(A)
    
    h = V_t[-1,:] / V_t[-1, -1]
    h = h.reshape(3,3)
    h_matrix = np.dot(np.dot(np.linalg.inv(dst_transform), h), src_transform)
    
    # END 

    return h_matrix

# 2.2 IMPLEMENT
def ransac_homography(keypoints1, keypoints2, matches, sampling_ratio=0.5, n_iters=500, delta=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        sampling_ratio: percentage of points selected at each iteration
        n_iters: the number of iterations RANSAC will run
        threshold: the threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * sampling_ratio)

    matched1_unpad = keypoints1[matches[:,0]]
    matched2_unpad = keypoints2[matches[:,1]]

    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    ### YOUR CODE HERE
    
    for i in range(n_iters):
        random_indices = np.random.choice(N, n_samples)
        matched1_sub = matched1_unpad[random_indices]
        matched2_sub = matched2_unpad[random_indices]
        
        current_H = compute_homography(matched1_sub, matched2_sub)
        
        dst = transform_homography(matched1_unpad, current_H)
        
        cur_n_inliers = 0
        cur_inliers = []
        for i in range(dst.shape[0]):
            original = matched2_unpad[i]
            estimated = dst[i]
            if np.sqrt((original[0] - estimated[0])**2 + (original[1] - estimated[1])**2) < delta:
                cur_n_inliers += 1
                cur_inliers.append(i)
        if cur_n_inliers > n_inliers:
            max_inliers = np.asarray(cur_inliers)
    
    all_inliers_matches = matches[max_inliers]
    matched1_final = keypoints1[all_inliers_matches[:,0]]
    matched2_final = keypoints2[all_inliers_matches[:,1]]
    
    H = compute_homography(matched1_final, matched2_final)
        
    ### END YOUR CODE
    return H, matches[max_inliers]

##################### PART 3 ###################
# GIVEN FROM PREV LAB
from skimage.feature import peak_local_max
def find_peak_params(hspace, params_list,  window_size=1, threshold=0.8):
    '''
    Given a Hough space and a list of parameters range, compute the local peaks
    aka bins whose count is larger max_bin * threshold. The local peaks are computed
    over a space of size (2*window_size+1)^(number of parameters)

    Also include the array of values corresponding to the bins, in descending order.
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

# GIVEN
def angle_with_x_axis(pi, pj):  
    '''
    Compute the angle that the line connecting two points I and J make with the x-axis (mind our coordinate convention)
    Do note that the line direction is from point I to point J.
    '''
    # get the difference between point p1 and p2
    y, x = pi[0]-pj[0], pi[1]-pj[1] 
    
    if x == 0:
        return np.pi/2  
    
    angle = np.arctan(y/x)
    if angle < 0:
        angle += np.pi
    return angle

# GIVEN
def midpoint(pi, pj):
    '''
    Get y and x coordinates of the midpoint of I and J
    '''
    return (pi[0]+pj[0])/2, (pi[1]+pj[1])/2

# GIVEN
def distance(pi, pj):
    '''
    Compute the Euclidean distance between two points I and J.
    '''
    y,x = pi[0]-pj[0], pi[1]-pj[1] 
    return np.sqrt(x**2+y**2)

# 3.1 IMPLEMENT
def shift_sift_descriptor(desc):
    '''
       Generate a virtual mirror descriptor for a given descriptor.
       Note that you have to shift the bins within a mini histogram, and the mini histograms themselves.
       e.g:
       Descriptor for a keypoint
       (the dimension is (128,), but here we reshape it to (16,8). Each length-8 array is a mini histogram.)
      [[  0.,   0.,   0.,   5.,  41.,   0.,   0.,   0.],
       [ 22.,   2.,   1.,  24., 167.,   0.,   0.,   1.],
       [167.,   3.,   1.,   4.,  29.,   0.,   0.,  12.],
       [ 50.,   0.,   0.,   0.,   0.,   0.,   0.,   4.],
       
       [  0.,   0.,   0.,   4.,  67.,   0.,   0.,   0.],
       [ 35.,   2.,   0.,  25., 167.,   1.,   0.,   1.],
       [167.,   4.,   0.,   4.,  32.,   0.,   0.,   5.],
       [ 65.,   0.,   0.,   0.,   0.,   0.,   0.,   1.],
       
       [  0.,   0.,   0.,   0.,  74.,   1.,   0.,   0.],
       [ 36.,   2.,   0.,   5., 167.,   7.,   0.,   4.],
       [167.,  10.,   0.,   1.,  30.,   1.,   0.,  13.],
       [ 60.,   2.,   0.,   0.,   0.,   0.,   0.,   1.],
       
       [  0.,   0.,   0.,   0.,  54.,   3.,   0.,   0.],
       [ 23.,   6.,   0.,   4., 167.,   9.,   0.,   0.],
       [167.,  40.,   0.,   2.,  30.,   1.,   0.,   0.],
       [ 51.,   8.,   0.,   0.,   0.,   0.,   0.,   0.]]
     ======================================================
       Descriptor for the same keypoint, flipped over the vertical axis
      [[  0.,   0.,   0.,   3.,  54.,   0.,   0.,   0.],
       [ 23.,   0.,   0.,   9., 167.,   4.,   0.,   6.],
       [167.,   0.,   0.,   1.,  30.,   2.,   0.,  40.],
       [ 51.,   0.,   0.,   0.,   0.,   0.,   0.,   8.],
       
       [  0.,   0.,   0.,   1.,  74.,   0.,   0.,   0.],
       [ 36.,   4.,   0.,   7., 167.,   5.,   0.,   2.],
       [167.,  13.,   0.,   1.,  30.,   1.,   0.,  10.],
       [ 60.,   1.,   0.,   0.,   0.,   0.,   0.,   2.],
       
       [  0.,   0.,   0.,   0.,  67.,   4.,   0.,   0.],
       [ 35.,   1.,   0.,   1., 167.,  25.,   0.,   2.],
       [167.,   5.,   0.,   0.,  32.,   4.,   0.,   4.],
       [ 65.,   1.,   0.,   0.,   0.,   0.,   0.,   0.],
       
       [  0.,   0.,   0.,   0.,  41.,   5.,   0.,   0.],
       [ 22.,   1.,   0.,   0., 167.,  24.,   1.,   2.],
       [167.,  12.,   0.,   0.,  29.,   4.,   1.,   3.],
       [ 50.,   4.,   0.,   0.,   0.,   0.,   0.,   0.]]
    '''
    # YOUR CODE HERE
    
    desc_1 = np.copy(desc)
    desc_1 = np.reshape(desc_1, (16, 8))

    for i in range(desc_1.shape[0]):
        desc_1[i] =  np.insert(desc_1[i][1: 8][::-1], 0, desc_1[i][0])
        
    desc_2 = np.reshape(desc_1, (4, 4, 8))
    res = np.flipud(desc_2)
    res = res.flatten()
    
    # END
    return res

# 3.1 IMPLEMENT
def create_mirror_descriptors(img):
    '''
    Return the output for (which you can find in utils.py)
    Also return the set of virtual mirror descriptors.
    Make sure the virtual descriptors correspond to the original set of descriptors.
    '''
    # YOUR CODE HERE
    
    (kps, descs, angles, sizes) = compute_cv2_descriptor(img)
    mir_descs = []
    
    for i in range(descs.shape[0]):
        mir_descs.append(shift_sift_descriptor(descs[i]))
   
    mir_descs = np.asarray(mir_descs)
    
    # END
    return kps, descs, sizes, angles, mir_descs

# 3.2 IMPLEMENT
def match_mirror_descriptors(descs, mirror_descs, threshold = 0.7):
    '''
    First use `top_k_matches` to find the nearest 3 matches for each keypoint. Then eliminate the mirror descriptor that comes 
    from the same keypoint. Perform ratio test on the two matches left. If no descriptor is eliminated, perform the ratio test 
    on the best 2. 
    '''
    three_matches = top_k_matches(descs, mirror_descs, k=3)

    match_result = []
    # YOUR CODE HERE
    
    three_matches_diff_kps = []
    for i in range(len(three_matches)):
        index = three_matches[i][0]
        matches_with_index = three_matches[i][1]
        for j in range(len(matches_with_index)):
            if matches_with_index[j][0] == index:
                matches_with_index.pop(j)
                break
        three_matches_diff_kps.append( (index, matches_with_index ))

    for pairs in three_matches_diff_kps:
        distance_1 = pairs[1][0][1]
        distance_2 = pairs[1][1][1]
        if (distance_1 / distance_2) < threshold:
            match_result.append([pairs[0], pairs[1][0][0]])

    match_result = np.array(match_result)
    
    # END
    return match_result

# 3.3 IMPLEMENT
def find_symmetry_lines(matches, kps):
    '''
    For each pair of matched keypoints, use the keypoint coordinates to compute a candidate symmetry line.
    Assume the points associated with the original descriptor set to be I's, and the points associated with the mirror descriptor set to be
    J's.
    '''
    rhos = []
    thetas = []
    # YOUR CODE HERE
    
    for match in matches:

        mid_point = midpoint(kps[match[0]], kps[match[1]])
        
        theta = angle_with_x_axis(kps[match[0]],kps[match[1]]) 
        thetas.append(theta)

        rho = mid_point[1]*np.cos(theta) + mid_point[0]*np.sin(theta)
        rhos.append(rho)

    # END
    
    return rhos, thetas

# 3.4 IMPLEMENT
def hough_vote_mirror(matches, kps, im_shape, window=1, threshold=0.5, num_lines=1):
    '''
    Hough Voting:
                 0<=thetas<= 2pi      , interval size = 1 degree
        -diagonal <= rhos <= diagonal , interval size = 1 pixel
    Feel free to vary the interval size.
    '''
    rhos, thetas = find_symmetry_lines(matches, kps)
    
    # YOUR CODE HERE
    
    rho_values = []
    theta_values = []

    max_distance = np.sqrt(im_shape[0]**2 + im_shape[1]**2)
    distances_range = np.arange(-max_distance, max_distance + 1, 1)

    angle_interval = np.pi/180
    thetas_range = np.arange(0, 2*np.pi, angle_interval)

    A = np.zeros((len(distances_range), len(thetas_range)))

    for i in range(len(matches)):
        i_rho = (int) (( rhos[i] - (-max_distance)) / 1.0)
        i_theta = (int) (( thetas[i] - 0) / (angle_interval))
        A[i_rho, i_theta] += 1

    res = find_peak_params(A, [distances_range, thetas_range], window, threshold)

    for i in range (num_lines):
         rho_values.append(res[1][i])
         theta_values.append(res[2][i])
  
    # END
    
    return rho_values, theta_values

##################### PART 4 ###################

# 4.1 IMPLEMENT
def match_with_self(descs, kps, threshold=0.8):
    '''
    Use `top_k_matches` to match a set of descriptors against itself and find the best 3 matches for each descriptor.
    Discard the trivial match for each trio (if exists), and perform the ratio test on the two matches left (or best two if no match is removed)
    '''
   
    matches = []
    
    # YOUR CODE HERE
    
    top_3_matches = top_k_matches(descs, descs, 3)
        
    for trios in top_3_matches:
        if trios[1][0][0] == trios[0]:
            distance_1 = trios[1][1][1]
            distance_2 = trios[1][2][1]
            if distance_1 / distance_2 < threshold:
                matches.append([trios[0], trios[1][1][0]])
        else:
            distance_1 = trios[1][0][1]
            distance_2 = trios[1][1][1]
            if distance_1 / distance_2 < threshold:
                matches.append([trios[0], trios[1][0][0]])
    matches = np.asarray(matches)
    
    # END
    return matches

# 4.2 IMPLEMENT
def find_rotation_centers(matches, kps, angles, sizes, im_shape):
    '''
    For each pair of matched keypoints (using `match_with_self`), compute the coordinates of the center of rotation and vote weight. 
    For each pair (kp1, kp2), use kp1 as point I, and kp2 as point J. The center of rotation is such that if we pivot point I about it,
    the orientation line at point I will end up coinciding with that at point J. 
    
    You may want to draw out a simple case to visualize first.
    
    If a candidate center lies out of bound, ignore it.
    '''
    # Y-coordinates, X-coordinates, and the vote weights 
    Y = []
    X = []
    W = []
    
    # YOUR CODE HERE

    matches_non_parallel = []

    for match in matches:
        if(abs(angles[match[0]] - angles[match[1]]) > 1.0):
            matches_non_parallel.append(match)

    angles = np.array(angles)
    angles *= (np.pi / 180.0)

    for match in matches_non_parallel:
        d = distance(kps[match[0]], kps[match[1]])
        gamma = angle_with_x_axis(kps[match[0]], kps[match[1]])
        beta = (angles[match[0]] - angles[match[1]] + np.pi) / 2.0
        radius = d / (2.0 * abs(np.cos(beta)))
        x_c = kps[match[0]][1] + radius*np.cos(beta + gamma)
        y_c = kps[match[0]][0] + radius*np.sin(beta + gamma)
        if(x_c >= im_shape[1] or y_c >= im_shape[0] or x_c < 0 or y_c < 0):
            continue

        q = - abs( sizes[match[0]] - sizes[match[1]]) / ( sizes[match[0]] + sizes[match[1]] )
        weight = np.exp(q) * np.exp(q)

        Y.append(y_c)
        X.append(x_c)
        W.append(weight)

    # END
    
    return Y,X,W

# 4.3 IMPLEMENT
def hough_vote_rotation(matches, kps, angles, sizes, im_shape, window=1, threshold=0.5, num_centers=1):
    '''
    Hough Voting:
        X: bound by width of image
        Y: bound by height of image
    Return the y-coordianate and x-coordinate values for the centers (limit by the num_centers)
    '''
    
    Y,X,W = find_rotation_centers(matches, kps, angles, sizes, im_shape)
    
    # YOUR CODE HERE
    
    y_values = []
    x_values = []

    y_range = np.arange(0, im_shape[0], 1)
    x_range = np.arange(0, im_shape[1], 1)
        
    A = np.zeros((len(y_range), len(x_range)))

    for i in range(len(Y)):
        i_y = (int) (Y[i])
        i_x = (int) (X[i])
        A[i_y, i_x] += W[i]

    res = find_peak_params(A, [y_range, x_range], window, threshold)

    for i in range(num_centers):
        y_values.append(res[1][i])
        x_values.append(res[2][i])
        
    # END
    
    return y_values, x_values