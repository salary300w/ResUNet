from PIL import Image
from tqdm import tqdm
import os
import torch
import torchvision


def LtoRGB(image):
    image=image.convert("RGB")
    pixel_access = image.load()
    for i in range(400):
        for j in range(240):
            if pixel_access[i, j][0] == 1:
                pixel_access[i, j] = (255, 0, 0)
            elif pixel_access[i, j][0] == 2:
                pixel_access[i, j] = (0, 255, 0)
            elif pixel_access[i, j][0] == 3:
                pixel_access[i, j] = (0, 0, 255)
    return image
def predict(ImagePath, SavePath, ModelPath):

    model = torch.load(ModelPath)
    img2tensor = torchvision.transforms.ToTensor()
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
            # 去除batch_size属性
            outputs = outputs.reshape(image.shape[1], image.shape[2], image.shape[3])
            outputs = tensor2img(outputs)
            outputs = LtoRGB(outputs)
            savepath = os.path.join(SavePath, os.path.splitext(filename)[0]+'.png')
            outputs.save(savepath)
    else:
        image = Image.open(ImagePath)
        image = img2tensor(image)
        # 添加batch_size属性
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        outputs = model(image)
        # 去除batch_size属性
        outputs = outputs.reshape(outputs.shape[1], outputs.shape[2], outputs.shape[3])
        outputs = tensor2img(outputs)
        outputs = LtoRGB(outputs)
        savepath = os.path.join(SavePath, os.path.splitext(os.path.basename(ImagePath))[0]+'.png')
        outputs.save(savepath)
    print('perdict done!')


if __name__ == "__main__":
    modelpath = 'resunet'
    # 可以是文件也可以是文件夹
    imagepath = 'data/train/images/cc1.jpg'
    save_dir = './mypredict'
    predict(ImagePath=imagepath, SavePath=save_dir, ModelPath=modelpath)
