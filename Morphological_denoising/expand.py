import cv2
import numpy as np

# 读取图像
img = cv2.imread('Morphological_denoising/Noise_Example.png', 0)

# 定义膨胀核大小
kernel = np.ones((5,5), np.uint8)

# 膨胀操作
dilation = cv2.dilate(img, kernel, iterations = 1)

# 显示图像
cv2.imwrite('Dilation.png', dilation)