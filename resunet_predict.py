from PIL import Image
from tqdm import tqdm
import os
import torch
import torchvision


def predict(ImagePath, SavePath, ModelPath):

    model = torch.load(ModelPath)
    img2tensor = torchvision.transforms.PILToTensor()
    tensor2img = torchvision.transforms.ToPILImage()

    if not os.path.exists(SavePath):
        os.makedirs(SavePath)
    if os.path.isdir(ImagePath):
        for filename in tqdm(os.listdir(ImagePath)):
            image = Image.open(os.path.join(ImagePath, filename))
            image = img2tensor(image)
            # 添加batch_size属性
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            outputs = model(image)
            outputs = tensor2img(outputs)
            savepath = os.path.join(SavePath, os.path.splitext(filename)[0]+'.png')
            Image.save(savepath, outputs)
    else:
        image = Image.open(ImagePath)
        image = img2tensor(image)
        # 添加batch_size属性
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        outputs = model(image)
        outputs = tensor2img(outputs)
        savepath = os.path.join(SavePath, '1.png')
        Image.save(savepath, outputs)
    print('perdict done!')


if __name__ == "__main__":
    modelpath = 'resunet'
    # 可以是文件也可以是文件夹
    imagepath = 'data/train/images/cc1.jpg'
    save_dir = './mypredict'
    predict(ImagePath=imagepath, SavePath=save_dir, ModelPath=modelpath)
