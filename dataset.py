import torch
from PIL import Image
import os
import numpy as np


def get_data_and_label(root_dir, is_train=True):
    # 定义训练集、验证集路径、训练集输入列表、训练集标签列表、验证集输入列表、验证集标签列表
    images_path = []
    labels_path = []
    path = os.path.join(root_dir, 'val')
    if is_train:
        path = os.path.join(root_dir, 'train')
    for filename in os.listdir(os.path.join(path, 'images')):
        name = os.path.splitext(filename)[0]
        images_filename = name+'.jpg'
        labels_filename = name+'.png'
        images_path.append(os.path.join(path, 'images', images_filename))
        labels_path.append(os.path.join(path, 'labels', labels_filename))
    return images_path, labels_path


def preprocess_input(image):
    image /= 255.0
    return image


class Mydataset(torch.utils.data.Dataset):
    """自定义数据集"""

    def __init__(self, root_dir, is_train=True):
        self.images_path, self.images_labels = get_data_and_label(root_dir=root_dir, is_train=is_train)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        label = Image.open(self.images_labels[item])
        img = np.transpose(preprocess_input(np.array(img, np.float32)), [2, 0, 1])
        label = np.array(label)
        return img, label