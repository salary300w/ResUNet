import torch
from PIL import Image
import os


def get_data_and_label(root_dir, is_train=True):
    # 定义训练集、验证集路径、训练集输入列表、训练集标签列表、验证集输入列表、验证集标签列表
    images_path = []
    labels_path = []
    if is_train:
        path = os.path.join(root_dir, 'train')
    else:
        path = os.path.join(root_dir, 'val')
    for filename in os.listdir(os.path.join(path, 'images')):
        name = os.path.splitext(filename)[0]
        images_filename = name+'.jpg'
        labels_filename = name+'.png'
        images_path.append(os.path.join(path, 'images', images_filename))
        labels_path.append(os.path.join(path, 'labels', labels_filename))
    return images_path, labels_path


class Mydataset(torch.utils.data.Dataset):
    """自定义数据集"""

    def __init__(self, root_dir, is_train=True, transform=None):
        self.images_path, self.images_labels = get_data_and_label(root_dir=root_dir, is_train=is_train)
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        label = Image.open(self.images_labels[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)
        return img, label