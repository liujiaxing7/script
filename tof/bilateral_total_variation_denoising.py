import cv2
import numpy as np


def bilateral_total_variation_denoising(image, lambda_tv, lambda_spatial, num_iterations):
    denoised_image = np.copy(image).astype(np.float64)  # 将图像转换为浮点型

    for _ in range(num_iterations):
        # 计算空间梯度
        gradient_spatial = cv2.Sobel(denoised_image, cv2.CV_64F, 1, 1)

        # 计算灰度梯度
        gradient_gray = cv2.Sobel(denoised_image, cv2.CV_64F, 0, 1)

        # 计算全变差梯度
        gradient_tv = np.sqrt(gradient_spatial ** 2 + gradient_gray ** 2)

        # 计算双向全变差梯度
        gradient_btv = gradient_tv / (gradient_tv + lambda_spatial)

        # 更新图像
        denoised_image += lambda_tv * gradient_btv

    return denoised_image.astype(np.uint8)  # 将图像转换回无符号整型

# 读取三维图像
image_tof = cv2.imread('filtered_image_ori.png')

lambda_tv = 10.1  # 全变差正则化参数
lambda_spatial = 10.01  # 空间正则化参数
num_iterations = 10  # 迭代次数

filtered_image = bilateral_total_variation_denoising(image_tof, lambda_tv, lambda_spatial, num_iterations)
# 使用 SpeckleFilter3D 进行去噪
# filtered_image = cv2.filterSpeckles(image_tof, 0 , 1, maxDiff=100)
# filtered_image = cv2.fastNlMeansDenoising(image_tof, h=10 ,templateWindowSize=7,searchWindowSize=21)
# 保存去噪后的图像
cv2.imwrite('filtered_image.png', filtered_image)


