# -*- coding: utf-8 -*-
import numpy as np


def depth_estimation(boxes):
    if boxes.shape[0] == 0:
        return np.zeros_like(boxes[:, 0])
    camera_height = 0.15
    K = np.asarray([4.9008025087775349e+02, 0., 6.5899436439430656e+02, 0.,
                    4.8978248934587106e+02, 4.1209311026514359e+02, 0., 0., 1.]).reshape(3, 3)

    # middle point in the bottom line
    points_2d = boxes[:, 2:] * 2
    points_2d_homo = np.concatenate(
        [points_2d, np.ones_like(points_2d[:, -1:])], axis=-1)
    points_3d = np.dot(np.linalg.inv(K), points_2d_homo.T).T
    scale = camera_height / points_3d[:, 1]
    points_3d = points_3d * scale[..., None]
    # z-coords
    return points_3d[:, -1]
