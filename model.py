import torch.nn as nn


class Res_U_Net(nn.Module):
    def __init__(self):
        super(Res_U_Net, self).__init__()
        self.layer_pooling = nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU()
        )
        self.layer_res = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU()
        )
        self.layer_tran = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3)
        )
        self.layer_tran1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.Softmax2d()
        )

    def forward(self, input):
        # Conv[7*7，stride=2]
        input = self.layer1(input)

        # Max-Pooling
        input = self.layer_pooling(input)

        # Res-Blocks
        input = self.layer_res(input)+input

        # Res-Blocks
        output1 = self.layer_res(input)+input

        # Max-Pooling
        input = self.layer_pooling(output1)

        # Res-Blocks
        input = self.layer_res(input)+input

        # Res-Blocks
        output2 = self.layer_res(input)+input

        # Max-Pooling
        input = self.layer_pooling(output2)

        # Res-Blocks
        input = self.layer_res(input)+input

        # Res-Blocks
        output3 = self.layer_res(input)+input

        # Max-Pooling
        input = self.layer_pooling(output3)

        # 2*Conv[3*3]
        input = self.layer2(input)

        # Transposed-Conv[3*3]
        input = self.layer_tran(input)+output3

        # 2*Conv[3*3]
        input = self.layer2(input)

        # Transposed-Conv[3*3]
        input = self.layer_tran(input)+output2

        # 2*Conv[3*3]
        input = self.layer2(input)

        # Transposed-Conv[3*3]
        input = self.layer_tran(input)+output1

        # 2*Conv[3*3]
        input = self.layer2(input)

        # Transposed-Conv[3*3]
        input = self.layer_tran(input)

        # Transposed-Conv[4*4]
        input = self.layer_tran1(input)

        # Conv[1*1]
        input = self.layer3(input)
        return input