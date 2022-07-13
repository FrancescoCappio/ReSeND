from torch import nn
from torchvision import models



class ResNetFc(nn.Module):
    def __init__(self,device,network):
        super(ResNetFc, self).__init__()
        if network=='resnet18':
            self.model_resnet = models.resnet18(pretrained=True)
        elif network=='resnet50':
            self.model_resnet = models.resnet50(pretrained=True)

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features
        self.device = device


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features

class CLS_binary(nn.Module):
    def __init__(self, in_dim, n_classes,bottle_neck_dim=256):
        super(CLS_binary, self).__init__()
        self.main = nn.Sequential(
                        nn.Linear(in_dim, 4096),
                        nn.BatchNorm1d(4096),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Linear(4096, 4096),
                        nn.BatchNorm1d(4096),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Linear(4096, 2048),
                        nn.BatchNorm1d(2048),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Linear(2048, 128),
                        nn.BatchNorm1d(128),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Linear(128, n_classes)
                )

    def forward(self, x):

        x = self.main(x)

        return x

class CLS(nn.Module):
    def __init__(self, in_dim, n_classes,bottle_neck_dim=256):
        super(CLS, self).__init__()
        self.main = nn.Sequential(
                        nn.Linear(in_dim, bottle_neck_dim),
                        nn.BatchNorm1d(bottle_neck_dim),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Linear(bottle_neck_dim, n_classes)
                )

    def forward(self, x):

        x = self.main(x)

        return x
