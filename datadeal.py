import os
import shutil


'''
设置：
    train_images
    train_labels
    val_images
    val_labels
运行后生成data2train文件夹，用于送入模型训练
'''


# 移动/复制source_dir所有文件至target_dir
def move_file(source_dir, target_dir, copy=True):
    files = os.listdir(source_dir)
    if copy:
        for file in files:
            source_file = os.path.join(source_dir, file)
            target_file = os.path.join(target_dir, file)
            shutil.copy(source_file, target_file)
    else:
        for file in files:
            source_file = os.path.join(source_dir, file)
            target_file = os.path.join(target_dir, file)
            shutil.move(source_file, target_file)


def get_filenames(folder_path, save_path, file_name):
    # 获取文件夹中所有的文件名
    file_names = os.listdir(folder_path)

    save_path = os.path.join(save_path, file_name)
    # 打开txt文件
    with open(save_path, 'w') as f:
        # 遍历每个文件名
        for file_name in file_names:
            # 去掉文件后缀名
            name = os.path.splitext(file_name)[0]
            # 将文件名写入txt文件
            f.write(name + '\n')

def make_dataset(train_path, train_label_path, val_path, val_label_path):
    # 创建data2train文件夹
    dir = 'data2train'
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    # 在data2train中创建子文件夹Segmentation、JPEGImages、SegmentationClass
    Seg_path = os.path.join(dir, 'Segmentation')
    JPEG_path = os.path.join(dir, 'JPEGImages')
    SegClass_path = os.path.join(dir, 'SegmentationClass')
    os.makedirs(Seg_path)
    os.makedirs(JPEG_path)
    os.makedirs(SegClass_path)

    # 获取训练集的图片名字存入train.txt中，txt文件存入Segmentation中
    get_filenames(train_path, Seg_path, "train.txt")

    # 获取验证集的图片名字存入val.txt中，txt文件存入Segmentation中
    get_filenames(val_path, Seg_path, "val.txt")

    # 复制训练集和验证集图片至JPEGImages文件夹内
    move_file(train_path, JPEG_path)
    move_file(val_path, JPEG_path)

    # 复制训练集标签和验证集标签至SegmentationClass文件夹内
    move_file(train_label_path, SegClass_path)
    move_file(val_label_path, SegClass_path)

if __name__ == "__main__":
    train_images = 'data/train/images'
    train_labels = 'data/train/labels'
    val_images = 'data/val/images'
    val_labels = 'data/val/labels'
    make_dataset(train_path=train_images, train_label_path=train_labels,
                 val_path=val_images, val_label_path=val_labels)
