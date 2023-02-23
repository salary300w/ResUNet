import torch
import torchvision
from res_net_module import *
from mydataset import *
from torch.utils.data import DataLoader


def singleton(cls):
    instances = {}

    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return getinstance


@singleton
class NetConfig:
    def __init__(self):
        self.epoch = 1
        self.device = 'cuda'
        self.email = True
        self.email_address = 'Atm991014@163.com'
        self.tensorboard = True
        self.data_dir='/home/cdk991014/workspace/ResUNet/data'
        self.module_save_dir='module_file'
        self.learning_rate=1e-3
        self.batch_size=2
        self.num_workers=4
        self.loss_fn=nn.MSELoss()
        self.transform = torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor()])