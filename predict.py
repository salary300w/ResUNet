import torch
import torchvision
from PIL import Image
from core.res_unet import *
from emailtool import *
from config import *

def res_net_out(Module_dir,input_dir):
    '''
        Module_dir:需要加载的模型路径
        input_dir:需要输入的图片路径
    '''
    config=NetConfig()
    module=torch.load(Module_dir)
    image=Image.open(input_dir)
    image=config.transform(image)
    image=module(image)
    transformer = torchvision.transforms.ToPILImage()
    image=transformer(image)
    return image

if __name__ == "__main__":
    module_dir='/home/cdk991014/workspace/ResUNet/module_file/1677164918.1164932/module_loss=74.92534'
    input='/home/cdk991014/workspace/ResUNet/data/val/images/cc7.jpg'
    image=res_net_out(Module_dir=module_dir,input_dir=input)
    image.save("output.png")