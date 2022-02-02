import cv2
import numpy as np
import math
from sklearn.cluster import KMeans
import sys
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed

### Part 1
def detect_points(img, min_distance, rou, pt_num, patch_size, tau_rou, gamma_rou):
    """
    Patchwise Shi-Tomasi point extraction.

    Hints:
    (1) You may find the function cv2.goodFeaturesToTrack helpful. The initial default parameter setting is given in the notebook.

    Args:
        img: Input RGB image. 
        min_distance: Minimum possible Euclidean distance between the returned corners. A parameter of cv2.goodFeaturesToTrack
        rou: Parameter characterizing the minimal accepted quality of image corners. A parameter of cv2.goodFeaturesToTrack
        pt_num: Maximum number of corners to return. A parameter of cv2.goodFeaturesToTrack
        patch_size: Size of each patch. The image is divided into several patches of shape (patch_size, patch_size). There are ((h / patch_size) * (w / patch_size)) patches in total given a image of (h x w)
        tau_rou: If rou falls below this threshold, stops keypoint detection for that patch
        gamma_rou: Decay rou by a factor of gamma_rou to detect more points.
    Returns:
        pts: Detected points of shape (N, 2), where N is the number of detected points. Each point is saved as the order of (height-corrdinate, width-corrdinate)
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w, c = img.shape

    Np = pt_num * 0.9 # The required number of keypoints for each patch. `pt_num` is used as a parameter, while `Np` is used as a stopping criterion.

    # YOUR CODE HERE
    pts = np.empty([0, 2])
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            if (i + patch_size <= h and j + patch_size <= w):
                patch = img_gray[i:i+patch_size, j:j+patch_size]
            else:
                patch = img_gray[i:w, j:w]

            count = 0
            rou_current = rou

            while(count <= Np and rou_current >= tau_rou):
                corner_patch = cv2.goodFeaturesToTrack(patch,maxCorners=pt_num, qualityLevel=rou_current, minDistance=min_distance)
                try:
                    corner_patch = np.reshape(corner_patch, (-1, 2))
                    count = corner_patch.shape[0]
                except:
                    print("Patch does not contain any key point")
                    count = 0
                rou_current *= gamma_rou
            
            try:
                for corner in corner_patch:
                    pts = np.vstack([pts, [[corner[0] + i, corner[1] + j]]])
            except:
                print("Empty patch")
    # END

    return pts

def extract_point_features(img, pts, window_patch):
    """
    Extract patch feature for each point.

    The patch feature for a point is defined as the patch extracted with this point as the center.

    Note that the returned pts is a subset of the input pts. 
    We discard some of the points as they are close to the boundary of the image and we cannot extract a full patch.

    Please normalize the patch features by subtracting the mean intensity and dividing by its standard deviation.

    Args:
        img: Input RGB image.
        pts: Detected point corners from detect_points().
        window_patch: The window size of patch cropped around the point. The final patch is of size (5 + 1 + 5, 5 + 1 + 5) = (11, 11). The center is the given point.
                      For example, suppose an image is of size (300, 400). The point is located at (50, 60). The window size is 5. 
                      Then, we use the cropped patch, i.e., img[50-5:50+5+1, 60-5:60+5+1], as the feature for that point. The patch size is (11, 11), so the dimension is 11x11=121.
    Returns:
        pts: A subset of the input points. We can extract a full patch for each of these points.
        features: Patch features of the points of the shape (N, (window_patch*2 + 1)^2), where N is the number of points
    """


    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.astype(float)
    h, w, c = img.shape
    
    # YOUR CODE HERE
    pts_ans = np.empty([0, 2])
    pts_copy = np.copy(pts)
    features = np.empty([0,(window_patch*2 + 1)*(window_patch*2 + 1)])
    for point in pts_copy:
        start_i = int(point[0]) - window_patch
        end_i = int(point[0]) + window_patch + 1
        start_j = int(point[1]) - window_patch
        end_j = int(point[1]) + window_patch + 1
        if (start_i < 0 or start_j < 0 or end_i > h or end_j > w):
            continue

        
        patch = img_gray[start_i:end_i, start_j:end_j]
        patch_flatten = patch.flatten()

        patch_mean = np.mean(patch_flatten)
        patch_std = np.std(patch_flatten)
        if (patch_std == 0):
            continue
        patch_flatten -= patch_mean
        patch_flatten /= patch_std
        features = np.vstack([features, [patch_flatten]])
        pts_ans = np.vstack([pts_ans, [point]])
        
    pts = pts_ans

    # End

    return pts, features


def distance(pi, pj):
    '''
    Compute the Euclidean distance between two points I and J.
    '''
    diff = np.array(pi) - np.array(pj)
    return np.sqrt(np.sum(diff*diff))

def find_neighbors(X, point, radius):
    i_nbrs = []
    for i in range(len(X)):
        if (distance(point, X[i])) <= radius:
            i_nbrs.append(i)

    return i_nbrs

def assign_points_to_clusters(X, cluster_centers):
    labels = np.empty(X.shape[0])
    for idx_point, point in enumerate(X):
        distance_to_centers = np.empty(cluster_centers.shape[0])
        for idx, center in enumerate(cluster_centers):
            distance_to_centers[idx] = distance(point, center)
        labels[idx_point] = np.argmin(distance_to_centers)
    
    return labels.astype('int')

count = 0
# separate function for each point's iterative loop
def shift_single_point(point, X, max_iter, bandwidth):
    global count
    count += 1
    print("Done shift " + str(count), end="\r")
    
    current_centroid = point
    # For each point, shift until the change is smaller than the threshold or reach max_iter
    threshold = 1e-3 * bandwidth  # when mean has converged
    no_iterations = 0
    while True:
        # Find mean of points within bandwidth
        idx_nbrs = find_neighbors(X, current_centroid, bandwidth)
        neighbor_points = X[idx_nbrs]
        if len(neighbor_points) == 0:
            break  
        prev_centroid = current_centroid  
        current_centroid = np.mean(neighbor_points, axis=0)
        # If converged or at max_iter, adds the cluster
        if (np.linalg.norm(current_centroid - prev_centroid) < threshold or no_iterations == max_iter):
            break
        no_iterations += 1
    return tuple(current_centroid), len(neighbor_points), no_iterations

def mean_shift_clustering(features, bandwidth):
    clustering = {}
    max_iter=300
    n_jobs=None
    center_intensity_dict = {}

    print("Total number of points: " + str(features.shape[0]))
    # final_points = []
    # for i, point in enumerate(features):
    #     res = shift_single_point(point, features, max_iter, bandwidth)
    #     final_points.append(res)
    #     print("Done shift " + str(i), end="\r")
    
    #Use Parallel and delay to speed up the process of shifting points, the iterative loop is 
    #commented above 
    final_points = Parallel(n_jobs=n_jobs)(
        delayed(shift_single_point)(point, features, max_iter, bandwidth)
        for point in features
    )
    
    print("Done shifting all points")

    # Copy results in a dictionary, to group points that end up at the same final destinations
    for i in range(len(features)):
        if final_points[i][1]:  
            center_intensity_dict[final_points[i][0]] = final_points[i][1]

    print("Done creating dictionary")
    center_sorted_by_intensity = sorted(
        center_intensity_dict.items(),
        key=lambda tup: (tup[1], tup[0]),
        reverse=True,
    )

    sorted_centers = np.array([tup[0] for tup in center_sorted_by_intensity])

    #Merge centers that are close to each other
    unique = np.ones(len(sorted_centers), dtype=bool)
    for idx, center in enumerate(sorted_centers):
        if unique[idx]:
            neighbor_idxs = find_neighbors(sorted_centers, center, bandwidth)
            unique[neighbor_idxs] = 0
            unique[idx] = 1  # leave the current point as unique
    cluster_centers = sorted_centers[unique]

    # ASSIGN LABELS: a point belongs to the cluster that it is closest to
    labels = assign_points_to_clusters(features, cluster_centers)

    clustering['labels_'] = np.array(labels)
    clustering['cluster_centers_'] = np.array(cluster_centers)
    clustering['bandwidth'] = bandwidth

    return clustering

#Helper function
def breakdown_cluster(n_clusters, pts, features, label_pts_array, idx_of_cluster):

    features_cluster = []
    pts_cluster = []

    for i in range(len(label_pts_array)):
        if label_pts_array[i] == idx_of_cluster:
            pts_cluster.append(pts[i])
            features_cluster.append(features[i])    
    clustering = KMeans(n_clusters=n_clusters).fit(features_cluster)

    new_clusters = []    
    no_of_clusters = clustering.cluster_centers_.shape[0]

    for i in range(no_of_clusters):
        new_clusters.append(np.empty([0, 2]))

    for i in range(len(clustering.labels_)):
        cluster_index = clustering.labels_[i]
        new_clusters[cluster_index] = np.vstack([new_clusters[cluster_index], [pts_cluster[i]]])

    return np.array(new_clusters)

def cluster(img, pts, features, bandwidth, tau1, tau2, gamma_h):
    """
    Group points with similar appearance, then refine the groups.

    "gamma_h" provides another way of fine-tuning bandwidth to avoid the number of clusters becoming too large.
    Alternatively, you can ignore "gamma_h" and fine-tune bandwidth by yourself.

    Args:
        img: Input RGB image.
        pts: Output from `extract_point_features`.
        features: Patch feature of points. Output from `extract_point_features`.
        bandwidth: Window size of the mean-shift clustering. In pdf, the bandwidth is represented as "h", but we use "bandwidth" to avoid the confusion with the image height
        tau1: Discard clusters with less than tau1 points
        tau2: Perform further clustering for clusters with more than tau2 points using K-means
        gamma_h: To avoid the number of clusters becoming too large, tune the bandwidth by gradually increasing the bandwidth by a factor gamma_h
    Returns:
        clusters: A list of clusters. Each cluster is a numpy ndarray of shape [N_cp, 2]. N_cp is the number of points of that cluster.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.astype(float)
    h, w, c = img.shape

    # YOUR CODE HERE
    clusters_coarse = []
    N = pts.shape[0]
    no_of_clusters = N
    clustering = None

    while (no_of_clusters > N / 3):
        print("bandwidth: "+ str(bandwidth) )
        clustering = mean_shift_clustering(features, bandwidth)
        no_of_clusters = clustering['cluster_centers_'].shape[0]
        bandwidth *= gamma_h
        print("no_of_clusters: " + str(no_of_clusters))
    
    for i in range(no_of_clusters):
        clusters_coarse.append(np.empty([0, 2]))

    for i in range(len(clustering['labels_'])):
        cluster_index = clustering['labels_'][i]
        clusters_coarse[cluster_index] = np.vstack([clusters_coarse[cluster_index], [pts[i]]])
    
    clusters = []
    cnt = 0
    for cluster_coarse in clusters_coarse:
        if (cluster_coarse.shape[0] < tau1):
            continue
        elif(cluster_coarse.shape[0] <= tau2):
            clusters.append(cluster_coarse)
        else:
            N_i = cluster_coarse.shape[0]
            breakdown_clusters = breakdown_cluster(int(N_i / tau2), pts, features, clustering['labels_'], cnt)
            for smaller_cluster in breakdown_clusters:
                clusters.append(smaller_cluster)
        cnt += 1

    # END
    return clusters

### Part 2

def find_top_X_nearest(point, pts_cluster, X):
    dists = []
    for point_2 in pts_cluster:
        dists.append(distance(point, point_2))
    
    top_X_indices = sorted(range(len(dists)), key=lambda k: dists[k])[1:X+1] 

    top_X_nearest = np.empty([0, 2])
    for index in top_X_indices:
        top_X_nearest = np.vstack([top_X_nearest, pts_cluster[index]])
    
    return top_X_nearest

def find_nearest_pt_int(point_x_transformed):
    base_point = (int(point_x_transformed[0]), int(point_x_transformed[1]))
    min_dist = sys.float_info.max
    for i in range(-1, 2, 1):
        for j in range(-1, 2, 1):
            point = (base_point[0] + i, base_point[1] + j)
            dist = distance(point_x_transformed, point)
            if (dist < min_dist):
                min_dist = dist
                nearest_point = point

    return nearest_point

def sort_three_key_points(point_1, point_2, point_3):
    distances = [distance(point_3, point_2), distance(point_1, point_3), distance(point_1, point_2)]
    max_index = np.argmax(np.array(distances))
    if (max_index == 0):
        point_a = point_1
        point_b = point_2
        point_c = point_3
    elif (max_index == 1):
        point_a = point_2
        point_b = point_1
        point_c = point_3
    elif (max_index == 2):
        point_a = point_3
        point_b = point_2
        point_c = point_1
    return (point_a, point_b, point_c)

def swapHW(point):
    point_temp = (point[1], point[0])
    return point_temp

def compute_angle(point_a, point_b, point_c):
    vector_1 = point_b-point_a
    vector_2 = point_c-point_a
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    if (dot_product >= 1 or dot_product <= -1):
        return 180 #to make it invalid

    angle = np.arccos(dot_product)
    return angle * 180 / np.pi

def get_proposal(pts_cluster, tau_a, X):
    """
    Get the lattice proposal

    Hints:
    (1) As stated in the lab4.pdf, we give priority to points close to each other when we sample a triplet.
        This statement means that we can start from the three closest points and iterate N_a times.
        There is no need to go through every triplet combination.
        For instance, you can iterate over each point. For each point, you choose 2 of the 10 nearest points. The 3 points form a triplet.
        In this case N_a = num_points * 45.

    (2) It is recommended that you reorder the 3 points. 
        Since {a, b, c} are transformed onto {(0, 0), (1, 0), (0, 1)} respectively, the point a is expected to be the vertex opposite the longest side of the triangle formed by these three points

    (3) Another way of refining the choice of triplet is to keep the triplet whose angle (between the edges <a, b> and <a, c>) is within a certain range.
        The range, for instance, is between 20 degrees and 120 degrees.

    (4) You may find `cv2.getAffineTransform` helpful. However, be careful about the HW and WH ordering when you use this function.

    (5) If two triplets yield the same number of inliers, keep the one with closest 3 points.

    Args:
        pts_cluster: Points within the same cluster.
        tau_a: The threshold of the difference between the transformed corrdinate and integer positions.
               For example, if a point is transformed into (1.1, -2.03), the closest integer position is (1, -2), then the distance is sqrt(0.1^2 + 0.03^2) (for Euclidean distance case).
               If it is smaller than "tau_a", we consider this point as inlier.
        X: When we compute the inliers, we only consider X nearest points to the point "a". 
    Returns:
        proposal: A list of inliers. The first 3 inliers are {a, b, c}. 
                  Each inlier is a dictionary, with key of "pt_int" and "pt" representing the integer positions after affine transformation and orignal coordinates.
    """
    # YOU CODE HERE
    max_no_of_inliers = 0
    point_a_final = None
    point_b_final = None
    point_c_final = None
    distance_min = sys.float_info.max
    for point_1 in pts_cluster:

        top_10_nearest = find_top_X_nearest(point_1, pts_cluster, 10)  
        for i in range(0, top_10_nearest.shape[0] - 1, 1):
            for j in range(i + 1, top_10_nearest.shape[0], 1):

                no_of_inliers = 0
                point_2 = top_10_nearest[i]
                point_3 = top_10_nearest[j]

                (point_a, point_b, point_c) = sort_three_key_points(point_1, point_2, point_3)                
                if (compute_angle(point_a, point_b, point_c) > 120 or compute_angle(point_a, point_b, point_c) < 20):
                    continue
                ratio = distance(point_a, point_b) / distance(point_a, point_c)
                if (ratio < 0.4 or ratio > 2):
                    continue

                # Define the 3 pairs of corresponding points 
                input_pts = np.float32([point_a.tolist(), point_b.tolist(), point_c.tolist()])
                output_pts = np.float32([[0,0], [1,0], [0,1]])
                
                # Calculate the transformation matrix using cv2.getAffineTransform()
                M = cv2.getAffineTransform(input_pts, output_pts)
                transform_matrix = np.vstack([M, [0, 0, 1]])

                top_X_nearest = find_top_X_nearest(point_a, pts_cluster, X)
                set_of_distinct_points = set()
                set_of_distinct_points.add((0,0))
                for point_x in top_X_nearest:
                    point_x_list = point_x.tolist()
                    point_x_list.append(1)
                    point_x_homo = np.array(point_x_list)
                    point_x_transformed = np.matmul(transform_matrix, point_x_homo)[:2].tolist()                 
                    nearest_pt_int = find_nearest_pt_int(point_x_transformed)
                    
                    if (distance(nearest_pt_int, point_x_transformed) < tau_a):
                        if (nearest_pt_int not in set_of_distinct_points):
                            set_of_distinct_points.add(nearest_pt_int)
                            no_of_inliers += 1
                #print(no_of_inliers)
                if (no_of_inliers > max_no_of_inliers):
                    max_no_of_inliers = no_of_inliers
                    point_a_final = point_a
                    point_b_final = point_b
                    point_c_final = point_c
                    distance_min = distance(point_b, point_c)
                elif ((no_of_inliers == max_no_of_inliers) and distance(point_b, point_c) < distance_min):
                    point_a_final = point_a
                    point_b_final = point_b
                    point_c_final = point_c         
                    distance_min = distance(point_b, point_c)

    proposal = []
    try:    
        proposal = [{"pt_int": np.array([0,0]).astype('int'), "pt": point_a_final.astype('int')},
                    {"pt_int": np.array([1,0]).astype('int'), "pt": point_b_final.astype('int')},
                    {"pt_int": np.array([0,1]).astype('int'), "pt": point_c_final.astype('int')}]

        # Define the 3 pairs of corresponding points 
        input_pts = np.float32([point_a_final.tolist(), point_b_final.tolist(), point_c_final.tolist()])
        output_pts = np.float32([[0,0], [1,0], [0,1]])

        M = cv2.getAffineTransform(input_pts, output_pts)
        transform_matrix = np.vstack([M, [0, 0, 1]])

        top_X_nearest = find_top_X_nearest(point_a_final, pts_cluster, X)
        final_transform_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        proposal_points = set()
        proposal_points.add((0,0))
        proposal_points.add((0,1))
        proposal_points.add((1,0))
        for point_x in top_X_nearest:
            point_x_list = point_x.tolist()
            point_x_list.append(1)
            point_x_homo = np.array(point_x_list)
            point_x_transformed = np.matmul(transform_matrix, point_x_homo)[:2].tolist()
            nearest_pt_int = find_nearest_pt_int(point_x_transformed)

            if nearest_pt_int not in proposal_points:
                proposal_points.add(nearest_pt_int)
                proposal.append({"pt_int": np.around(np.array(nearest_pt_int), 0).astype('int'), "pt": point_x.astype('int')})
    except:
        print("Proposal error")
    # END

    return proposal


def is_valid_triplet(pt_int_1, pt_int_2, pt_int_3):
    diff_1 = (abs(pt_int_1[0] - pt_int_2[0]), abs(pt_int_1[1] - pt_int_2[1]))
    diff_2 = (abs(pt_int_1[0] - pt_int_3[0]), abs(pt_int_1[1] - pt_int_3[1]))
    diff_3 = (abs(pt_int_3[0] - pt_int_2[0]), abs(pt_int_3[1] - pt_int_2[1]))
    if (diff_1 == (0,1)) and (diff_2 == (1,0)):
        return True
    elif (diff_1 == (1,0)) and (diff_2 == (0,1)):
        return True
    elif (diff_1 == (0,1)) and (diff_3 == (1,0)):
        return True
    elif (diff_1 == (1,0)) and (diff_3 == (0,1)):
        return True
    elif (diff_3 == (0,1)) and (diff_2 == (1,0)):
        return True
    elif (diff_3 == (1,0)) and (diff_2 == (0,1)):
        return True

    return False

def does_not_exist(list_4_points, list_of_list_of_points):
    for list_of_points in list_of_list_of_points:
        if (set(list_of_points) == set(list_4_points)):
            return False
    return True

def get_texel(img_demo, point_a, point_b, point_c, texel_size):
    """
    point_a is the base

    Returns:
    warped texel
    """
    window = texel_size
    corners_src = [np.float32(point_b),
                np.float32(point_a),
                np.float32(point_c)] # (h, w) ordering
    point_fourth = corners_src[1] + (corners_src[0] - corners_src[1]) + (corners_src[2] - corners_src[1])

    corners_src.append(point_fourth)
    corners_src = np.float32(corners_src)
    corners_dst = np.float32([[ 0,  0],
                            [window,  0],
                            [window, window],
                            [0, window]])
    matrix_projective = cv2.getPerspectiveTransform(corners_src[:, [1, 0]], corners_dst) # transpose (h, w), as the input argument of cv2.getPerspectiveTransform is (w, h) ordering
    img_demo_warped = cv2.warpPerspective(img_demo, matrix_projective, (window, window))
    return img_demo_warped

def find_texels(img, proposal, texel_size=50):
    """
    Find texels from the given image.

    Hints:
    (1) This function works on RGB image, unlike previous functions such as point detection and clustering that operate on grayscale image.

    (2) You may find `cv2.getPerspectiveTransform` and `cv2.warpPerspective` helpful.
        Please refer to the demo in the notebook for the usage of the 2 functions.
        Be careful about the HW and WH ordering when you use this function.
    
    (3) As stated in the pdf, each texel is defined by 3 or 4 inlier keypoints on the corners.
        If you find this sentence difficult to understand, you can go to check the demo.
        In the demo, a corresponding texel is obtained from 3 points. The 4th point is predicted from the 3 points.


    Args:
        img: Input RGB image
        proposal: Outputs from get_proposal(). Proposal is a list of inliers.
        texel_size: The patch size (U, V) of the patch transformed from the quadrilateral. 
                    In this implementation, U is equal to V. (U = V = texel_size = 50.) The texel is a square.
    Returns:
        texels: A numpy ndarray of the shape (#texels, texel_size, texel_size, #channels).
    """
    # YOUR CODE HERE
    list_of_list_of_points = []
    texels=[]

    for i in range(0, len(proposal) - 2, 1):
        for j in range(i+1, len(proposal) - 1, 1):
            for k in range(j+1, len(proposal), 1):
                pt_int_1 = tuple(proposal[i]['pt_int'])
                pt_int_2 = tuple(proposal[j]['pt_int'])
                pt_int_3 = tuple(proposal[k]['pt_int'])
                if (is_valid_triplet(pt_int_1, pt_int_2, pt_int_3)): #Check if the different is correct
                    pt_int_1_np = proposal[i]['pt_int']
                    pt_int_2_np = proposal[j]['pt_int']
                    pt_int_3_np = proposal[k]['pt_int']
                    
                    (point_a, point_b, point_c) = sort_three_key_points(pt_int_1_np, pt_int_2_np, pt_int_3_np)
                    pt_int_4_np = point_a + (point_b - point_a) + (point_c - point_a)
                    pt_int_4 = tuple(pt_int_4_np)
          
                    if does_not_exist([pt_int_1, pt_int_2, pt_int_3, pt_int_4], list_of_list_of_points):
                        list_of_list_of_points.append([pt_int_1, pt_int_2, pt_int_3, pt_int_4])
                        
                        point_1 = tuple(proposal[i]['pt'])
                        point_2 = tuple(proposal[j]['pt'])
                        point_3 = tuple(proposal[k]['pt'])

                        (point_a, point_b, point_c) = sort_three_key_points(point_1, point_2, point_3)
                        texel = get_texel(img, point_a, point_b, point_c, texel_size)
                        texels.append(texel)                   


    texels = np.array(texels)              

    # END
    return texels

def normalize_texel_by_channels(texel):
    texel = texel.astype('float')
    U, V, C = texel.shape
    for i in range(C):
        mean = np.mean(texel[:,:,i])
        std = np.std(texel[:,:,i])
        texel[:,:,i] -= mean
        texel[:,:,i] /= (std + 0.000001)
    return texel

def score_proposal(texels, a_score_count_min=3):
    """
    Calcualte A-Score.

    Hints:
    (1) Each channel is normalized separately.
        The A-score for a RGB texel is the average of 3 A-scores of each channel.

    (2) You can return 1000 (in our example) to denote a invalid A-score.
        An invalid A-score is usually results from clusters with less than "a_score_count_min" texels.

    Args:
        texels: A numpy ndarray of the shape (#texels, window, window, #channels).
        a_score_count_min: Minimal number of texels we need to calculate the A-score.
    Returns:
        a_score: A-score calculated from the texels. If there are no sufficient texels, return 1000.    
    """
    try:
        K, U, V, C = texels.shape
    except:
        print("Empty texels")
        return 1000.0

    # YOUR CODE HERE
    if (K < a_score_count_min):
        return 1000.0
    texels_copy = []
    for j in range(K):
        texels_copy.append(normalize_texel_by_channels(texels[j,:,:,:])) 
    
    A_sum = 0
    for i in range(C):
        A_i = 0
        for u in range(U):
            for v in range(V):
                temp_list = []
                for k in range(K):
                    temp_list.append(texels_copy[k][u, v, i])
                temp_list = np.array(temp_list)
                A_i += np.std(temp_list)

        A_i /= (U*V*np.sqrt(K))
        A_sum += A_i

    a_score = A_sum/C

    # END

    return a_score

### Part 3
# You are free to change the input argument of the functions in Part 3.
# GIVEN
def non_max_suppression(response, suppress_range, threshold=None):
    """
    Non-maximum Suppression for translation symmetry detection

    The general approach for non-maximum suppression is as follows:
        1. Perform thresholding on the input response map. Set the points whose values are less than the threshold as 0.
        2. Find the largest response value in the current response map
        3. Set all points in a certain range around this largest point to 0. 
        4. Save the current largest point
        5. Repeat the step from 2 to 4 until all points are set as 0. 
        6. Return the saved points are the local maximum.

    Args:
        response: numpy.ndarray, output from the normalized cross correlation
        suppress_range: a tuple of two ints (H_range, W_range). The points around the local maximum point within this range are set as 0. In this case, there are 2*H_range*2*W_range points including the local maxima are set to 0
    Returns:
        threshold: int, points with value less than the threshold are set to 0
    """
    H, W = response.shape[:2]
    H_range, W_range = suppress_range
    res = np.copy(response)

    if threshold is not None:
        res[res<threshold] = 0

    idx_max = res.reshape(-1).argmax()
    x, y = idx_max // W, idx_max % W
    point_set = set()
    while res[x, y] != 0:
        point_set.add((x, y))
        res[max(x - H_range, 0): min(x+H_range, H), max(y - W_range, 0):min(y+W_range, W)] = 0
        idx_max = res.reshape(-1).argmax()
        x, y = idx_max // W, idx_max % W
    for x, y in point_set:
        res[x, y] = response[x, y]
    return res


def template_match(img, proposal, threshold):
    """
    Perform template matching on the original input image.

    Hints:
    (1) You may find cv2.copyMakeBorder and cv2.matchTemplate helpful. The cv2.copyMakeBorder is used for padding.
        Alternatively, you can use your implementation in Lab 1 for template matching.

    (2) For non-maximum suppression, you can either use the one you implemented for lab 1 or the code given above.

    Returns:
        response: A sparse response map from non-maximum suppression. 
    """
    # YOUR CODE HERE
    
    a, b, c = np.array(proposal[:3])
    coor_a = a['pt']
    coor_b = b['pt']
    coor_c = c['pt']
    coor_d = coor_c + (coor_b - coor_a)
    
    hori_min = min(coor_a[1], coor_b[1], coor_c[1], coor_d[1])
    hori_max = max(coor_a[1], coor_b[1], coor_c[1], coor_d[1])
    vert_min = min(coor_a[0], coor_b[0], coor_c[0], coor_d[0])
    vert_max = max(coor_a[0], coor_b[0], coor_c[0], coor_d[0])
    
    if (len(img.shape) == 3):
        template = img[vert_min:vert_max, hori_min:hori_max,:]
    else:
        template = img[vert_min:vert_max, hori_min:hori_max]
        
    h, w = template.shape[:2]
    
    if (len(img.shape) == 3):
        img = cv2.copyMakeBorder(img, h//2, h//2, w//2, w//2, cv2.BORDER_CONSTANT, np.zeros((3,)))
    else:
        img = cv2.copyMakeBorder(img, h//2, h//2, w//2, w//2, cv2.BORDER_CONSTANT, 0)
                                                         
    response = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
                                 
    response = non_max_suppression(response, (h//2, w//2), threshold)                                
    # END
    return response

def maxima2grid(img, proposal, response):
    """
    Estimate 4 lattice points from each local maxima.

    Hints:
    (1) We can transfer the 4 offsets between the center of the original template and 4 lattice unit points to new detected centers.

    Args:
        response: The response map from `template_match()`.

    Returns:
        points_grid: an numpy ndarray of shape (N, 2), where N is the number of grid points.
    
    """
    # YOUR CODE HERE
    
    H, W = img.shape[:2]
    
    a, b, c = np.array(proposal[:3])
    coor_a = a['pt']
    coor_b = b['pt']
    coor_c = c['pt']
    coor_d = coor_c + (coor_b - coor_a)
    
    hori_min = min(coor_a[1], coor_b[1], coor_c[1], coor_d[1])
    hori_max = max(coor_a[1], coor_b[1], coor_c[1], coor_d[1])
    vert_min = min(coor_a[0], coor_b[0], coor_c[0], coor_d[0])
    vert_max = max(coor_a[0], coor_b[0], coor_c[0], coor_d[0])
    hori_center = (hori_min + hori_max) // 2
    vert_center = (vert_min + vert_max) // 2
    coor_center = np.array([vert_center, hori_center])
    
    points_grid = []
    points_grid_dict = {}
    
    for i in range(response.shape[0]):
        for j in range(response.shape[1]):
            if response[i,j] != 0:
                points_grid_dict[(i,j)] = {} 
                point_a = np.array([i, j]) + (coor_a - coor_center)
                if point_a[0] >= 0 and point_a[0] < H and point_a[1] >= 0 and point_a[1] < W:
                    points_grid.append(point_a)
                    points_grid_dict[(i,j)]['a'] = point_a
                point_b = np.array([i, j]) + (coor_b - coor_center)
                if point_b[0] >= 0 and point_b[0] < H and point_b[1] >= 0 and point_b[1] < W:
                    points_grid.append(point_b)
                    points_grid_dict[(i,j)]['b'] = point_b
                point_c = np.array([i, j]) + (coor_c - coor_center)
                if point_c[0] >= 0 and point_c[0] < H and point_c[1] >= 0 and point_c[1] < W:
                    points_grid.append(point_c)
                    points_grid_dict[(i,j)]['c'] = point_c
                point_d = np.array([i, j]) + (coor_d - coor_center)
                if point_d[0] >= 0 and point_d[0] < H and point_d[1] >= 0 and point_d[1] < W:
                    points_grid.append(point_d)
                    points_grid_dict[(i,j)]['d'] = point_d
    
    points_grid = np.array(points_grid)
    
    # END

    return points_grid, points_grid_dict


def refine_grid(img, proposal, points_grid_dict):
    """
    Refine the detected grid points.

    Args:
        points_grid_dict: The output from the `maxima2grid()`.

    Returns:
        points: A numpy ndarray of shape (N, 2), where N is the number of refined grid points.
    """
    # YOUR CODE HERE
    
    a, b, c = proposal[:3]
    coor_a = np.array(a['pt'])
    coor_b = np.array(b['pt'])
    coor_c = np.array(c['pt'])
    coor_d = coor_c + (coor_b - coor_a)
    template_width = max(coor_a[1], coor_b[1], coor_c[1], coor_d[1]) - min(coor_a[1], coor_b[1], coor_c[1], coor_d[1])
    template_height = max(coor_a[0], coor_b[0], coor_c[0], coor_d[0]) - min(coor_a[0], coor_b[0], coor_c[0], coor_d[0])
    consider_range = math.sqrt(template_width**2 + template_height**2) / 3 
    
    points = []
    
    null_value = np.array((-1, -1))
    
    for center in points_grid_dict:
        for point in points_grid_dict[center]:
            point_to_add = points_grid_dict[center][point]
            if (point_to_add == null_value).all():
                continue
            n = 1
            if point == 'a':
                neighbor_center_d = np.array(center) + (coor_a - coor_d)
                for centerp in points_grid_dict:
                    if np.linalg.norm(neighbor_center_d - np.array(centerp)) < consider_range:
                        if ('d' in points_grid_dict[centerp].keys()):
                            if (points_grid_dict[centerp]['d'] == null_value).all():
                                continue
                            point_to_add += points_grid_dict[centerp]['d']
                            points_grid_dict[centerp]['d'] = null_value
                            n += 1
                neighbor_center_c = np.array(center) + (coor_a - coor_c)
                for centerp in points_grid_dict:
                    if np.linalg.norm(neighbor_center_c - np.array(centerp)) < consider_range:
                        if ('c' in points_grid_dict[centerp].keys()):
                            if (points_grid_dict[centerp]['c'] == null_value).all():
                                continue
                            point_to_add += points_grid_dict[centerp]['c']
                            points_grid_dict[centerp]['c'] = null_value
                            n += 1
                neighbor_center_b = np.array(center) + (coor_a - coor_b)
                for centerp in points_grid_dict:
                    if np.linalg.norm(neighbor_center_b - np.array(centerp)) < consider_range:
                        if ('b' in points_grid_dict[centerp].keys()):
                            if (points_grid_dict[centerp]['b'] == null_value).all():
                                continue
                            point_to_add += points_grid_dict[centerp]['b']
                            points_grid_dict[centerp]['b'] = null_value
                            n += 1
            if point == 'b':
                neighbor_center_d = np.array(center) + (coor_b - coor_d)
                for centerp in points_grid_dict:
                    if np.linalg.norm(neighbor_center_d - np.array(centerp)) < consider_range:
                        if ('d' in points_grid_dict[centerp].keys()):
                            if (points_grid_dict[centerp]['d'] == null_value).all():
                                continue
                            point_to_add += points_grid_dict[centerp]['d']
                            points_grid_dict[centerp]['d'] = null_value
                            n += 1
                neighbor_center_c = np.array(center) + (coor_b - coor_c)
                for centerp in points_grid_dict:
                    if np.linalg.norm(neighbor_center_c - np.array(centerp)) < consider_range:
                        if ('c' in points_grid_dict[centerp].keys()):
                            if (points_grid_dict[centerp]['c'] == null_value).all():
                                continue
                            point_to_add += points_grid_dict[centerp]['c']
                            points_grid_dict[centerp]['c'] = null_value
                            n += 1
                neighbor_center_a = np.array(center) + (coor_b - coor_a)
                for centerp in points_grid_dict:
                    if np.linalg.norm(neighbor_center_a - np.array(centerp)) < consider_range:
                        if ('a' in points_grid_dict[centerp].keys()):
                            if (points_grid_dict[centerp]['a'] == null_value).all():
                                continue
                            point_to_add += points_grid_dict[centerp]['a']
                            points_grid_dict[centerp]['a'] = null_value
                            n += 1
            if point == 'c':
                neighbor_center_d = np.array(center) + (coor_c - coor_d)
                for centerp in points_grid_dict:
                    if np.linalg.norm(neighbor_center_d - np.array(centerp)) < consider_range:
                        if ('d' in points_grid_dict[centerp].keys()):
                            if (points_grid_dict[centerp]['d'] == null_value).all():
                                continue
                            point_to_add += points_grid_dict[centerp]['d']
                            points_grid_dict[centerp]['d'] = null_value
                            n += 1
                neighbor_center_b = np.array(center) + (coor_c - coor_b)
                for centerp in points_grid_dict:
                    if np.linalg.norm(neighbor_center_b - np.array(centerp)) < consider_range:
                        if ('b' in points_grid_dict[centerp].keys()):
                            if (points_grid_dict[centerp]['b'] == null_value).all():
                                continue
                            point_to_add += points_grid_dict[centerp]['b']
                            points_grid_dict[centerp]['b'] = null_value
                            n += 1
                neighbor_center_a = np.array(center) + (coor_c - coor_a)
                for centerp in points_grid_dict:
                    if np.linalg.norm(neighbor_center_a - np.array(centerp)) < consider_range:
                        if ('a' in points_grid_dict[centerp].keys()):
                            if (points_grid_dict[centerp]['a'] == null_value).all():
                                continue
                            point_to_add += points_grid_dict[centerp]['a']
                            points_grid_dict[centerp]['a'] = null_value
                            n += 1
            if point == 'd':
                neighbor_center_c = np.array(center) + (coor_d - coor_c)
                for centerp in points_grid_dict:
                    if np.linalg.norm(neighbor_center_c - np.array(centerp)) < consider_range:
                        if ('c' in points_grid_dict[centerp].keys()):
                            if (points_grid_dict[centerp]['c'] == null_value).all():
                                continue
                            point_to_add += points_grid_dict[centerp]['c']
                            points_grid_dict[centerp]['c'] = null_value
                            n += 1
                neighbor_center_b = np.array(center) + (coor_d - coor_b)
                for centerp in points_grid_dict:
                    if np.linalg.norm(neighbor_center_b - np.array(centerp)) < consider_range:
                        if ('b' in points_grid_dict[centerp].keys()):
                            if (points_grid_dict[centerp]['b'] == null_value).all():
                                continue
                            point_to_add += points_grid_dict[centerp]['b']
                            points_grid_dict[centerp]['b'] = null_value
                            n += 1
                neighbor_center_a = np.array(center) + (coor_d - coor_a)
                for centerp in points_grid_dict:
                    if np.linalg.norm(neighbor_center_a - np.array(centerp)) < consider_range:
                        if ('a' in points_grid_dict[centerp].keys()):
                            if (points_grid_dict[centerp]['a'] == null_value).all():
                                continue
                            point_to_add += points_grid_dict[centerp]['a']
                            points_grid_dict[centerp]['a'] = null_value
                            n += 1
            point_to_add //= n
            points.append(point_to_add)
    
    # The result contains all grid points already so no need to interpolate missing grid points.
    points = np.array(points)
    
    # END

    return points


def grid2latticeunit(img, proposal, points):
    """
    Convert each lattice grid point into integer lattice grid.

    Hints:
    (1) Since it is difficult to know whether two points should be connected, one way is to map each point into an integer position.
        The integer position should maintain the spatial relationship of these points.
        For instance, if we have three points x1=(50, 50), x2=(70, 50) and x3=(70, 70), we can map them (4, 5), (5, 5) and (5, 6).
        As the distances between (4, 5) and (5, 5), (5, 5) and (5, 6) are both 1, we know that (x1, x2) and (x2, x3) form two edges.
    
    (2) You can use affine transformation to build the mapping above, but do not perform global affine transformation.

    (3) The mapping in the hints above are merely to know whether two points should be connected. 
        If you have your own method for finding the relationship, feel free to implement your owns and ignore the hints above.


    Returns:
        edges: A list of edges in the lattice structure. Each edge is defined by two points. The point coordinate is in the image coordinate.
    """

    # YOUR CODE HERE
    H, W = img.shape[:2]
    
    a, b, c = proposal[:3]
    coor_a = np.array(a['pt'])
    coor_b = np.array(b['pt'])
    coor_c = np.array(c['pt'])
    
    # Found the unit x, y vector
    x_unit = coor_b - coor_a
    y_unit = coor_c - coor_a
    
    # consider_range defines the error distance to know if a neighbor could have an edge with with the current point or not
    # this could be changed in order to accept more error to connect all the grid points
    consider_range = math.sqrt(x_unit[0]**2 + x_unit[1]**2 + y_unit[0]**2 + y_unit[1]**2) / 5 
    
    edges = []
    
    connect_dict = {}
    for point in points:
        connect_dict[tuple(point)] = []
    
    # For every point, consider their 4 neighbors with unit x, y vector to find adjacent grid point and make an edges
    # Here, affine transformation is done implicitly by calculating unit x, y vector based on proposal
    # THe integer lattice grid is not assigned explicitly since it is not the output required,
    # But if we want to assign, just use integer coordinate of 1 point considered as (0,0)
    # Then its 4 neighbors will be (1, 0) (-1, 0) (0, 1) (0, -1) and continue considering with its neighbors
    
    for point in points:
        neighbor_1 = point + x_unit
        if neighbor_1[0] >=0 and neighbor_1[0] < H and neighbor_1[1] >= 0 and neighbor_1[1] < W:
            for other_point in points:
                if np.linalg.norm(other_point - neighbor_1) < consider_range:
                    if not (tuple(point) in connect_dict[tuple(other_point)]):
                        connect_dict[tuple(point)].append(tuple(other_point))
        neighbor_2 = point - x_unit
        if neighbor_2[0] >=0 and neighbor_2[0] < H and neighbor_2[1] >= 0 and neighbor_2[1] < W:
            for other_point in points:
                if np.linalg.norm(other_point - neighbor_2) < consider_range:
                    if not (tuple(point) in connect_dict[tuple(other_point)]):
                        connect_dict[tuple(point)].append(tuple(other_point))
        neighbor_3 = point + y_unit
        if neighbor_3[0] >=0 and neighbor_3[0] < H and neighbor_3[1] >= 0 and neighbor_3[1] < W:
            for other_point in points:
                if np.linalg.norm(other_point - neighbor_3) < consider_range:
                    if not (tuple(point) in connect_dict[tuple(other_point)]):
                        connect_dict[tuple(point)].append(tuple(other_point))
        neighbor_4 = point - y_unit
        if neighbor_4[0] >=0 and neighbor_4[0] < H and neighbor_4[1] >= 0 and neighbor_4[1] < W:
            for other_point in points:
                if np.linalg.norm(other_point - neighbor_4) < consider_range:
                    if not (tuple(point) in connect_dict[tuple(other_point)]):
                        connect_dict[tuple(point)].append(tuple(other_point))
    
    for point in connect_dict:
        for other_point in connect_dict[point]:
            edges.append((point, other_point))

    # END

    return edges


