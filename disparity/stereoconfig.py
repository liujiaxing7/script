import numpy as np
import cv2

#双目相机参数
class stereoCameral(object):
    def __init__(self):

        #左相机内参数
        self.cam_matrix_left = np.array([[249.82379, 0., 156.38459], [0., 249.07678, 122.46872], [0., 0., 1.]])
        #右相机内参数
        self.cam_matrix_right = np.array([[242.77875, 0., 153.22330], [0., 242.27426, 117.63536], [0., 0., 1.]])

        #左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[-0.02712, -0.03795, -0.00409, 0.00526, 0.00000]])
        self.distortion_r = np.array([[-0.03348, 0.08901, -0.00327, 0.00330, 0.00000]])

        #旋转矩阵
        om = np.array([-0.00320, -0.00163, -0.00069])
        self.R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
        #平移矩阵
        self.T = np.array([-90.24602, 3.17981, -19.44558])
        