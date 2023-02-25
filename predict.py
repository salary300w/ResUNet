# ----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import numpy as np
from PIL import Image

from unet import Unet
import os


def predict(ImagePath, SavePath, NameClasses):
    # -------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    # -------------------------------------------------------------------------#
    unet = Unet()
    # -------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    # -------------------------------------------------------------------------#
    count = False
    try:
        image = Image.open(ImagePath)
    except:
        print('Open Error! Try again!')
    else:
        r_image = unet.detect_image(image, count=count, name_classes=NameClasses)
        r_image.save(SavePath)
        print('perdict done!')
        print('----------------------------------------------------------------------')

if __name__ == "__main__":
    name_classes = ["background", "feat", "ress", "bott"]
    imagepath='data/val/images/cc7.jpg'
    savepath='./predict-image.png'
    predict(ImagePath=imagepath,SavePath=savepath,NameClasses=name_classes)