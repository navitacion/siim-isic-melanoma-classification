import torch
from torch import nn
from efficientnet_pytorch import EfficientNet


class ENet(nn.Module):
    def __init__(self, output_size=1, model_name='efficientnet-b0', meta_features_num=3):
        super(ENet, self).__init__()
        self.enet = EfficientNet.from_pretrained(model_name=model_name)
        self.fc = nn.Sequential(
            nn.Linear(in_features=meta_features_num, out_features=500),
            nn.BatchNorm1d(500),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.classification = nn.Linear(1500, out_features=output_size)

    def forward(self, x, d):
        out1 = self.enet(x)
        out2 = self.fc(d)
        out = torch.cat((out1, out2), dim=1)

        out = self.classification(out)

        return out
