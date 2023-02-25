from PIL import Image
from unet import Unet


def predict(ImagePath, SavePath, NameClasses, ModelPath, MixType):
    # -------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    # -------------------------------------------------------------------------#
    unet = Unet(ModelPath, MixType)
    # -------------------------------------------------------------------------#
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    # -------------------------------------------------------------------------#
    try:
        image = Image.open(ImagePath)
    except:
        print('Open Error! Try again!')
    else:
        r_image = unet.detect_image(image, name_classes=NameClasses)
        r_image.save(SavePath)
        print('perdict done!')
        print('----------------------------------------------------------------------')


if __name__ == "__main__":
    name_classes = ["background", "feat", "ress", "bott"]
    imagepath = 'data/val/images/cc7.jpg'
    savepath = './predict-image.png'
    predict(ImagePath=imagepath, SavePath=savepath, NameClasses=name_classes, ModelPath='logs/best_epoch_weights.pth', MixType=1)
