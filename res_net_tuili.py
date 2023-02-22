from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from PIL import Image
import time
import os
import shutil
from VOCSegDataset import *
from res_net_module import *
from emailtool import *

def res_net_out(Module_dir,input_dir):
    '''
        Module_dir:需要加载的模型路径
        input_dir:需要输入的图片路径
    '''
    res_net_module=torch.load(Module_dir)
    out_images=res_net_module(torchvision.io.read_image(input_dir).to(dtype=torch.float32))
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    # 反归一化
    denormalize = torchvision.transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)],std=[1/s for s in std])
    image=Image.fromarray(denormalize(out_images).permute(1, 2, 0).numpy().astype('uint8'))
    return image

if __name__ == "__main__":
    module_dir='module_file/1677058224.9426198/module_loss=34930.07812'
    input='data/train/images/cc1.jpg'
    image=res_net_out(Module_dir=module_dir,input_dir=input)
    image.save('1.png')
