import torchvision
import torch.nn as nn
from dataset import *
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
        self.epoch = 100
        self.device = 'cuda'
        self.email = True
        self.email_address = 'Atm991014@163.com'
        self.tensorboard = True
        self.data_dir='data'
        self.module_save_dir='module_file'
        self.log_save_dir='train_logs'
        self.learning_rate=1e-3
        self.batch_size=2
        self.num_workers=4
        self.loss_fn=nn.CrossEntropyLoss()