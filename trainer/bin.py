import torch
import torch.nn as nn
import torch.nn.functional as F


# 二值化激活函数
class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        return torch.sign(x)


# 二值化权重
class BinaryWeight(nn.Module):
    def __init__(self, weight):
        super(BinaryWeight, self).__init__()
        self.weight = weight

    def forward(self):
        return torch.sign(self.weight)

    def backward(self, grad):
        return grad


# 二值化卷积层
class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.weight_bin = BinaryWeight(self.weight)

    def forward(self, x):
        # 正向传播使用浮点参数
        weight = self.weight_bin()
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# 二值化残差块
class BinaryResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BinaryResidualBlock, self).__init__()
        self.conv1 = BinaryConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.binary_activation = BinaryActivation()
        self.conv2 = BinaryConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                BinaryConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.binary_activation(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        out += identity
        out = self.binary_activation(out)

        return out


# 18 层二值化残差网络
class BinaryResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(BinaryResNet18, self).__init__()
        self.in_channels = 64

        self.conv1 = BinaryConv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.binary_activation = BinaryActivation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BinaryResidualBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BinaryResidualBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BinaryResidualBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BinaryResidualBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.binary_activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# 测试模型
if __name__ == "__main__":
    model = BinaryResNet18(num_classes=37)
    print(model)
