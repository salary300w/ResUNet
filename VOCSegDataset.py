import torch
import torchvision
import os


def read_voc_images(data_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(data_dir, 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            data_dir, 'JPEGImages', f'{fname}.jpg')).to(dtype=torch.float32))
        labels.append(torchvision.io.read_image(os.path.join(
            data_dir, 'SegmentationClass', f'{fname}.png'), mode).to(dtype=torch.float32))
    return features, labels


class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""

    def __init__(self, is_train, voc_dir, settransform=None):
        self.transform = settransform
        self.features, self.labels = read_voc_images(
            voc_dir, is_train=is_train)

    def __getitem__(self, idx):
        if self.transform is not None:
            self.features[idx] = self.transform(self.features[idx])
            self.labels[idx] = self.transform(self.labels[idx])
        return (self.features[idx], self.labels[idx])

    def __len__(self):
        return len(self.features)
