import torch
from torch import nn
from efficientnet_pytorch import EfficientNet


class TestNet(nn.Module):
    def __init__(self, output_size=1):
        super(TestNet, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pool_1 = nn.MaxPool2d((2, 2))

        self.block_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.pool_2 = nn.MaxPool2d((2, 2))

        self.block_5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.block_6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.linear = nn.Linear(in_features=256, out_features=output_size)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.pool_1(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.pool_2(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.global_pool(x)
        x = x.view(b, -1)
        x = self.linear(x)

        return x


class ENet(nn.Module):
    def __init__(self, output_size=1, model_name='efficientnet-b0'):
        super(ENet, self).__init__()
        self.enet = EfficientNet.from_pretrained(model_name=model_name, num_classes=output_size)

    def forward(self, x):
        out = self.enet(x)

        return out


class ENet_2(nn.Module):
    def __init__(self, output_size=1, model_name='efficientnet-b0'):
        super(ENet_2, self).__init__()
        self.enet = EfficientNet.from_pretrained(model_name=model_name, num_classes=output_size)
        self.bn = nn.BatchNorm2d(1408, eps=0.001)
        self.pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(1408, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        out = self.enet.extract_features(x)
        out = self.bn(out)
        out = self.pool(out)
        out = out.squeeze()
        out = self.fc(out)

        return out


if __name__ == '__main__':
    z = torch.randn(8, 3, 384, 384)
    model = ENet_2()

    print(model)

    out = model(z)
    print(out.size())
