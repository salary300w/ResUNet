from PIL import Image
from unet import Unet
import os
from tqdm import tqdm
import time


def predict(ImagePath, SavePath, NameClasses, ModelPath, MixType):
    # -------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    # -------------------------------------------------------------------------#
    unet = Unet(ModelPath, MixType)
    # -------------------------------------------------------------------------#
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    # -------------------------------------------------------------------------#
    if not os.path.exists(SavePath):
        os.makedirs(SavePath)
    if os.path.isdir(ImagePath):
        start_time = time.time()
        for filename in tqdm(os.listdir(ImagePath)):
            image = Image.open(os.path.join(ImagePath, filename))
            savepath = os.path.join(SavePath, os.path.splitext(filename)[0]+'.png')
            unet.detect_image(image,NameClasses).save(savepath)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"代码运行时间: {run_time:.6f} 秒")
    else:
        filename= os.path.splitext(os.path.basename(ImagePath))[0]
        image = Image.open(ImagePath)
        savepath = os.path.join(SavePath, filename+'.png')
        unet.detect_image(image,NameClasses).save(savepath)
    print('perdict done!')

if __name__ == "__main__":
    name_classes = ["background", "feat", "ress", "bott"]
    modelpath = 'logs/best_epoch_weights.pth'

    # 可以是文件也可以是文件夹
    imagepath = 'data/val/images'
    save_dir = './predict_out'
    MixType = 1
    predict(ImagePath=imagepath, SavePath=save_dir, 
    NameClasses=name_classes, ModelPath=modelpath, MixType=MixType)