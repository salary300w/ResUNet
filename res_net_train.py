from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torch
import torchvision
import time
import os
import shutil
from mydataset import *
from res_net_module import *
from emailtool import *


def train(epoch=200, dev="cuda", email=True, email_addr="Atm991014@163.com", tensorboard=True):

    # epoch:迭代次数
    # dev:训练设备
    # email:训练完成是否发送邮件通知
    # email_addr:接收通知的邮箱地址
    # accuracy_level:当训练集准确率大于accuracy_level,会进行测试。测试集准确率大于accuracy_level会进行模型保存并结束训练
    # tensorboard:是否使用tensorboard绘制训练曲线
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    # 归一化处理
    normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    learning_rate = 1e-3
    model_batch_size=2

    # 定义训练的设备
    dev = torch.device(device=dev if torch.cuda.is_available() else "cpu")

    # 数据集准备
    train_data = Mydataset(root_dir='/home/cdk991014/workspace/ResUNet/data',is_train=True,transform=transform)
    test_data = Mydataset(root_dir='/home/cdk991014/workspace/ResUNet/data',is_train=False,transform=transform)

    # 数据集大小
    print("-----训练集大小= {} -----".format(len(train_data)))
    print("-----测试集大小= {} -----".format(len(test_data)))

    # 数据集加载
    train_loader = DataLoader(dataset=train_data, batch_size=model_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_data, batch_size=model_batch_size, shuffle=True, num_workers=4)

    # 创建网络模型,转移至训练设备
    module = Res_U_Net().to(device=dev)

    # 损失函数,转移至训练设备
    loss_fn = nn.MSELoss().to(device=dev)

    # 优化器
    optimizer = torch.optim.Adam(params=module.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(params=module.parameters(), lr=learning_rate)

    # 设置训练网络的一些参数

    # 记录测试的次数
    test_step = 0

    if tensorboard:
        # 使用tensorboard画出训练曲线
        if os.path.exists("train_logs"):
            shutil.rmtree("train_logs")
        writer = SummaryWriter("train_logs")

    # 设置模型存储目录
    save_path = "module_file"
    save_path = os.path.join(save_path, str(time.time()))
    os.makedirs(save_path)

    # -----开始训练-----
    print("-----开始训练-----")
    start_time = time.time()
    for i in range(1, epoch+1):
        print("-----第 {} 轮训练开始-----".format(i))
        total_train_loss = 0  # 记录每次迭代的总误差值
        total_test_loss = 0  # 记录每次迭代的总误差值
        # 训练步骤
        module.train()  # 设定为训练模式，仅对某些特殊层生效，具体看说明文档
        for data in train_loader:
            imgs, labels = data
            # 转移至训练设备
            imgs = imgs.to(dev)
            labels = labels.to(dev)
            
            # 将数据输入模型
            outputs = module(imgs)

            # 累加训练集的损失值
            loss = loss_fn(outputs, labels)
            total_train_loss += loss

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 计算本轮训练集的正确率
        print("-----第 {} 轮训练Loss: {} -----".format(i, total_train_loss))
        if i % 10 == 0:
            print(f"-----总用时: {time.time()-start_time:.2f} 秒-----")

        # 绘制训练曲线图
        if tensorboard:
            writer.add_scalar(tag="total_train_loss", scalar_value=total_train_loss, global_step=i)

        # 训练集准确度达标则进行测试
        if i == epoch or i % 5 == 0:
            # 测试步骤开始
            print("-----开始测试-----")
            module.eval()  # 设定为验证模式，仅对某些特殊层生效，具体看说明文档
            with torch.no_grad():  # 去掉梯度，保证测试过程不会对网络模型的参数调优
                for data in test_loader:
                    imgs, labels = data
                    # 转移至训练设备
                    imgs = imgs.to(dev)
                    labels = labels.to(dev)

                    # 将数据输入模型
                    outputs = module(imgs)

                    # 累加测试集的损失值
                    loss = loss_fn(outputs, labels)
                    total_test_loss += loss
            test_step += 1
            # 绘制测试曲线图
            if tensorboard:
                writer.add_scalar(tag="test_loss", scalar_value=total_test_loss, global_step=test_step)
            print("-----第 {} 轮测试Loss: {} -----".format(test_step, total_test_loss))
            if i % 10 == 0:
                print(f"-----总用时: {time.time()-start_time:.2f} 秒-----")
    # -----迭代结束-----
    # 计算训练用时并输出
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("-----训练完成-----")
    print(f"-----总用时: {elapsed_time:.2f} 秒-----")
    print("-----总迭代次数: {} -----".format(i))

    # 如果没有模型保存，则进行模型保存
    if not os.listdir(save_path):
        savemodule(MODULE=module, PATH=save_path, LOSS=total_test_loss)
    writer.close()

    # 发送邮件
    if email:
        print("-----发送邮件通知-----")
        sendemail = Email(email_addr)
        sendemail.send(
            "训练完成<br/>测试集Loss: {}<br/>迭代次数: {}<br/>用时: {} 秒".format(
                total_test_loss, i, elapsed_time
            )
        )


def savemodule(MODULE, PATH, LOSS):
    print("-----保存模型参数-----")
    MODULE.to(torch.device(device="cpu"))  # 将模型转移至cpu保存
    torch.save(MODULE, os.path.join(PATH, "module_loss={}".format(round(LOSS.item(), 5))))


if __name__ == "__main__":
    train(epoch=30, email=True)
