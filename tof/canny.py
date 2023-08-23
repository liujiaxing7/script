import cv2
import numpy as np

# 读取深度图像
depth_image = cv2.imread('filtered_image_ori.png', cv2.IMREAD_GRAYSCALE)

# 边界检测参数设置
threshold1 = 0  # Canny边界检测低阈值
threshold2 = 400  # Canny边界检测高阈值
kernel_size = 3  # 边界检测的Sobel算子核大小

# 应用边界检测
edges = cv2.Canny(depth_image, threshold1, threshold2, apertureSize=kernel_size)

# 填充边界内部区域
filled_edges = edges.copy()
cv2.floodFill(filled_edges, None, (0, 0), 255)

mask = np.zeros_like(depth_image)
mask[filled_edges != 0] = 255

# 将掩膜应用到深度图像上，剔除边界噪声
filtered_image = cv2.bitwise_and(depth_image, mask)

# 显示原始深度图像、边界图像和剔除边界噪声后的图像
cv2.imshow('Original Depth Image', depth_image)
cv2.imshow('Edges', edges)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
