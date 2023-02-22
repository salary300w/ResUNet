import cv2
import numpy as np
import os
from tqdm import tqdm
import time


def two_pass(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 第一遍扫描
    current_label = 0    # 当前标签
    labels = np.zeros(gray.shape, dtype=np.int32)    # 标签矩阵
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i][j] != 0:    # 如果当前像素不是背景像素
                neighbors = []    # 邻居像素的标签
                if i > 0 and labels[i-1][j] != 0:
                    neighbors.append(labels[i-1][j])    # 上方邻居像素的标签
                if j > 0 and labels[i][j-1] != 0:
                    neighbors.append(labels[i][j-1])    # 左侧邻居像素的标签
                if not neighbors:
                    current_label += 1    # 如果没有邻居像素，则创建一个新标签
                    labels[i][j] = current_label
                else:
                    labels[i][j] = min(neighbors)    # 否则，将该像素的标签设置为邻居中的最小标签
                    for neighbor in neighbors:
                        if neighbor != labels[i][j]:
                            # 将与该像素不相同的邻居像素的标签替换为最小标签
                            labels = np.where(
                                labels == neighbor, labels[i][j], labels)
    # 第二遍扫描
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128),
              (128, 0, 128), (192, 192, 192), (128, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128), (128, 0, 255), (0, 128, 255)]    # 为每个标签生成一个固定颜色
    output = np.zeros(
        (gray.shape[0], gray.shape[1], 3), dtype=np.uint8)    # 输出图像矩阵
    unique_values = sorted(np.unique(labels))  # 获取label矩阵中的唯一值并排序
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if labels[i][j] != 0:    # 如果该像素有标签，则使用对应颜色填充输出图像
                index = np.where(unique_values == labels[i][j])[0][0]
                if i <= 130:
                    output[i][j] = colors[index]
                else:
                    output[i][j] = colors[20-len(unique_values)+index]
    return output


if __name__ == "__main__":
    # 定义文件夹路径和输出目录
    input_dir = 'predict_out'
    output_dir = 'two_pass'

    # 遍历每个图片文件，读取并处理图像，然后保存处理后的图像到输出目录
    start_time = time.time()
    for image_file in tqdm(os.listdir(input_dir)):
        # 读取图片
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)

        # 处理图像
        processed_image = two_pass(image)

        # 保存处理后的图像到输出目录
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, processed_image)
    end_time = time.time()
    run_time = end_time - start_time
    print(f"代码运行时间: {run_time:.6f} 秒")