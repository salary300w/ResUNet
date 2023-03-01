import cv2
import numpy as np

# 读取图像
img = cv2.imread('Dilation.png', 0)

# 定义腐蚀操作的卷积核
kernel = np.ones((5,5),np.uint8)

# 对图像进行腐蚀操作
erosion = cv2.erode(img, kernel, iterations = 1)

# 显示原图像和腐蚀后的图像
cv2.imwrite('Eroded.png', erosion)