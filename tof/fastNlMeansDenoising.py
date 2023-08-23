import cv2
# 读取三维图像
image_tof = cv2.imread('48_1614047984636253.png')
image_tof = image_tof[:, :, 0] + (image_tof[:, :, 1] > 0) * 255 + image_tof[:, :, 1] + (image_tof[:, :, 2] > 0) * 511 + image_tof[:, :, 2]
image_tof = image_tof.astype("uint8")
cv2.imwrite('filtered_image_ori.png', image_tof)
# 使用 SpeckleFilter3D 进行去噪
# filtered_image = cv2.filterSpeckles(image_tof, 0 , 0, maxDiff=5)
filtered_image = cv2.fastNlMeansDenoising(image_tof, h=10 ,templateWindowSize=7,searchWindowSize=21)
# 保存去噪后的图像
cv2.imwrite('filtered_image.png', filtered_image)