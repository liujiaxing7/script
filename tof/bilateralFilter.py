import cv2

# 读取深度图像
import numpy as np

depth_image = cv2.imread('48_1614047984636253.png')


# 双边滤波参数设置
d = 15  # 邻域直径
sigma_color = 40  # 颜色空间标准差
sigma_space = 40  # 坐标空间标准差

# 应用双边滤波
filtered_image = cv2.bilateralFilter(depth_image, d, sigma_color, sigma_space)

# 显示原始深度图像和滤波后的图像
cv2.imwrite("origin_depth_image.png", depth_image)
cv2.imwrite("filtered_Image.png", filtered_image)
cv2.imshow('Original Depth Image', depth_image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
