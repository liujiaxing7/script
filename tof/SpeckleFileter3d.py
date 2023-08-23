import numpy as np
import cv2


def speckle_filter_3d(src, new_val, area_thresh, thresh_ptr, axis):
    h, w, _ = src.shape
    pixels = h * w

    has_labeled = np.zeros(pixels, dtype=bool)
    connect_x = np.zeros(pixels, dtype=int)
    connect_y = np.zeros(pixels, dtype=int)
    record_loc = np.zeros(pixels, dtype=int)

    src_ptr = src.reshape(-1, 3)
    hsize = 3 * w
    loci = 0
    h1 = h - 1
    w1 = w - 1

    for k in range(pixels):
        has_labeled[k] = False

    for i in range(h):
        for j in range(w):
            loc = loci + j
            loc3z = loc * 3 + axis

            if not has_labeled[loc]:
                connect_num = 0
                record_num = 0

                connect_x[connect_num] = j
                connect_y[connect_num] = i

                has_labeled[loc] = True

                while connect_num >= 0 and src_ptr[loc3z] > 0:
                    cur_x = connect_x[connect_num]
                    cur_y = connect_y[connect_num]

                    cur_loc = cur_x + cur_y * w
                    cur_loc_z = cur_loc * 3 + axis
                    cur_value = src_ptr[cur_loc_z]
                    depth_thresh = thresh_ptr * cur_value

                    record_loc[record_num] = cur_loc_z

                    connect_num -= 1
                    record_num += 1

                    if cur_x > 1 and not has_labeled[cur_loc - 1] and abs(src_ptr[cur_loc_z - 3] - cur_value) < depth_thresh:
                        connect_num += 1
                        connect_x[connect_num] = cur_x - 1
                        connect_y[connect_num] = cur_y
                        has_labeled[cur_loc - 1] = True

                    if cur_x < w1 and not has_labeled[cur_loc + 1] and abs(src_ptr[cur_loc_z + 3] - cur_value) < depth_thresh:
                        connect_num += 1
                        connect_x[connect_num] = cur_x + 1
                        connect_y[connect_num] = cur_y
                        has_labeled[cur_loc + 1] = True

                    if cur_y > 1 and not has_labeled[cur_loc - w] and abs(src_ptr[cur_loc_z - hsize] - cur_value) < depth_thresh:
                        connect_num += 1
                        connect_x[connect_num] = cur_x
                        connect_y[connect_num] = cur_y - 1
                        has_labeled[cur_loc - w] = True

                    if cur_y < h1 and not has_labeled[cur_loc + w] and abs(src_ptr[cur_loc_z + hsize] - cur_value) < depth_thresh:
                        connect_num += 1
                        connect_x[connect_num] = cur_x
                        connect_y[connect_num] = cur_y + 1
                        has_labeled[cur_loc + w] = True

                if record_num < area_thresh and src_ptr[loc3z] > 0:
                    for k in range(record_num):
                        cur_pos = record_loc[k] - axis
                        src_ptr[cur_pos] = new_val[0]
                        src_ptr[cur_pos + 1] = new_val[1]
                        src_ptr[cur_pos + 2] = new_val[2]

            loci += w

    return src

# Example usage
image = cv2.imread('03_1614044883510422.png').astype(np.float32) / 255.0  # Load and normalize input image
new_value = np.array([0, 0, 0])  # New value for replacing small speckle regions
area_threshold = 100  # Area threshold for determining the size of speckle regions
threshold_coefficient = 0.1  # Threshold coefficient for calculating depth thresholds
processing_axis = 2  # Axis for specifying the processing direction

filtered_image = speckle_filter_3d(image, new_value, area_threshold, threshold_coefficient, processing_axis)

cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
