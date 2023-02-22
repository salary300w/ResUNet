import cv2
import os


# 对图片进行闭运算
filename = 'opening_Noise_Example.png'

# 读入RGB图像
img = cv2.imread(os.path.join('Morphological_denoising', filename))

# 设置结构元素大小
kernel_size = (3, 3)

# 创建结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)

# 进行开运算
opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# 存储处理后的图像
cv2.imwrite(os.path.join('Morphological_denoising',
            'closing_'+filename), opening)