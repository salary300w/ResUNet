from torch.utils.tensorboard import SummaryWriter
import torch
import time
import os
import shutil
from config import *
from dataset import *
from core.res_unet import *
from emailtool import *


def train():
    
    config=NetConfig()
    # 定义训练的设备
    dev = torch.device(device=config.device if torch.cuda.is_available() else 'cpu')
    # 创建网络模型、优化器
    module = ResUnet(3).to(device=dev)
    optimizer = torch.optim.Adam(params=module.parameters(), lr=config.learning_rate)


    # 数据集准备
    train_data = Mydataset(root_dir=config.data_dir,is_train=True)
    test_data = Mydataset(root_dir=config.data_dir,is_train=False)

    # 数据集大小
    print("-----训练集大小= {} -----".format(len(train_data)))
    print("-----测试集大小= {} -----".format(len(test_data)))

    # 数据集加载
    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    #损失函数,转移至训练设备
    loss_fn = config.loss_fn.to(device=dev)
    # optimizer = torch.optim.SGD(params=module.parameters(), lr=learning_rate)

    # 记录测试的次数
    test_step = 0

    if config.tensorboard:
        # 使用tensorboard画出训练曲线
        if os.path.exists("train_logs"):
            shutil.rmtree("train_logs")
        writer = SummaryWriter("train_logs")

    # 设置模型存储目录
    save_path = config.module_save_dir
    save_path = os.path.join(save_path, str(time.time()))
    os.makedirs(save_path)

    # -----开始训练-----
    print("-----开始训练-----")
    start_time = time.time()
    for i in range(1, config.epoch+1):
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
        print(f"-----总用时: {time.time()-start_time:.2f} 秒-----")

        # 绘制训练曲线图
        if config.tensorboard:
            writer.add_scalar(tag="total_train_loss", scalar_value=total_train_loss, global_step=i)

        # 训练集准确度达标则进行测试
        if i == config.epoch or i % 5 == 0:
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
            if config.tensorboard:
                writer.add_scalar(tag="test_loss", scalar_value=total_test_loss, global_step=test_step)
            print("-----第 {} 轮测试Loss: {} -----".format(test_step, total_test_loss))
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
    if config.email:
        print("-----发送邮件通知-----")
        sendemail = Email(config.email_address)
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
    train()
