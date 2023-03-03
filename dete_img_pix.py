import torchvision
from PIL import Image


image = Image.open("data/train/labels/cc1.png")
mode = image.mode
print('voc图片格式:', mode)
pixel_value = image.getpixel((0, 0))
print('voc背景像素值:', pixel_value)  # voc背景像素值
pixel_value = image.getpixel((60, 60))
print('voc特征线像素值:', pixel_value)  # voc特征线像素值
pixel_value = image.getpixel((80, 80))
print('voc电阻丝像素值:', pixel_value)  # voc电阻丝像素值
pixel_value = image.getpixel((140, 140))
print('voc回波像素值:', pixel_value)  # voc回波像素值

pixel = []
for i in range(400):
    for j in range(240):
        pixel.append(image.getpixel((i, j)))
pixel = set(pixel)
print(pixel)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
transformtoPIL = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage()
])
image = transform(image)
image = transformtoPIL(image)

pixel = []
for i in range(400):
    for j in range(240):
        pixel.append(image.getpixel((i, j)))
pixel = set(pixel)
print(pixel)

mode = image.mode
print('voc图片转torch张量再转回PIL后的格式:', mode)
pixel_value = image.getpixel((0, 0))
print('voc图片转torch张量再转回PIL后的背景像素值:', pixel_value)  # voc背景像素值
pixel_value = image.getpixel((60, 60))
print('voc图片转torch张量再转回PIL后的特征线像素值:', pixel_value)  # voc特征线像素值
pixel_value = image.getpixel((80, 80))
print('voc图片转torch张量再转回PIL后的电阻丝像素值:', pixel_value)  # voc电阻丝像素值
pixel_value = image.getpixel((140, 140))
print('voc图片转torch张量再转回PIL后的回波像素值:', pixel_value)  # voc回波像素值
p_image = image.convert('P')
mode = p_image.mode
print('voc图片格式:', mode)