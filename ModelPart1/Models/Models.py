import torch.nn as nn
import torchvision.transforms
from torchvision.models import resnet18
from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_integrate, so3_integrate_only_gamma
from s2cnn import so3_near_identity_grid
from s2cnn import s2_near_identity_grid
import torch.nn.functional as F
import torch
import math

class SphericalModelDeep(nn.Module):

    def __init__(self, bandwidth=30):
        super(SphericalModelDeep, self).__init__()

        grid_s2 = s2_near_identity_grid()  # roughly a 5x5 filter
        grid_so3_4 = so3_near_identity_grid()  # roughly 6x6 filter

        self.conv1 = S2Convolution(
            nfeature_in=1,
            nfeature_out=64,
            b_in=bandwidth,
            b_out=bandwidth // 2,
            grid=grid_s2)
        self.maxpool1 = nn.MaxPool3d((1, 1, bandwidth))
        self.conv2 = S2Convolution(
            nfeature_in=64,
            nfeature_out=128,
            b_in=bandwidth // 2,
            b_out=bandwidth // 2,
            grid=grid_s2)
        self.maxpool2 = nn.MaxPool3d((1, 1, bandwidth))

        self.conv3 = S2Convolution(
            nfeature_in=128,
            nfeature_out=256,
            b_in=bandwidth // 2,
            b_out=bandwidth // 4,
            grid=grid_s2)

        self.conv4 = SO3Convolution(
            nfeature_in=256,
            nfeature_out=512,
            b_in=bandwidth // 4,
            b_out=bandwidth // 4,
            grid=grid_so3_4)

        self.conv5 = SO3Convolution(
            nfeature_in=512,
            nfeature_out=512,
            b_in=bandwidth // 4,
            b_out=bandwidth // 4,
            grid=grid_so3_4)

        self.conv6 = SO3Convolution(
            nfeature_in=512,
            nfeature_out=1,
            b_in=bandwidth // 4,
            b_out=bandwidth,
            grid=grid_so3_4)
        self.maxpool3 = nn.MaxPool3d((1, 1, bandwidth * 2))

    # S2->S2->S2->SO3->S03
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = so3_integrate_only_gamma(x)
        x = self.maxpool1(x)
        x = torch.squeeze(x, dim=-1)

        x = self.conv2(x)
        x = F.relu(x)
        # x = so3_integrate_only_gamma(x)
        x = self.maxpool2(x)
        x = torch.squeeze(x, dim=-1)

        x = self.conv3(x)
        x = F.relu(x)

        # fully connected model
        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.maxpool3(x)
        # x = so3_integrate_only_gamma(x)
        x = torch.squeeze(x, dim=-1)

        return x


class ResNetModelBased(nn.Module):

    def __init__(self):
        super(ResNetModelBased, self).__init__()
        # Computational Graph Version

        self.model = resnet18(pretrained=True)

        # Last Layers

        self.convd1 = nn.Conv2d(512, 512, 9)
        self.convd2 = nn.Conv2d(512, 1, 9)

    def forward(self, x):
        img_xformer = torchvision.transforms.Resize(
            (x.shape[2] * 6, x.shape[3] * 6))  # since height and width need at least 224
        x = img_xformer(x)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # x = self.model.avgpool(x)
        preXformer = torchvision.transforms.Resize((30, 30))
        x = preXformer(x)

        x = self.convd1(x)
        x = F.relu(x)
        x = self.convd2(x)
        finalXformer = torchvision.transforms.Resize((60, 60))
        x = finalXformer(x)

        return x


class ModelBasedOnPaperGitHubVersionSINGLEBRANCH(nn.Module):

    def __init__(self, config):
        super(ModelBasedOnPaperGitHubVersionSINGLEBRANCH, self).__init__()
        # Computational Graph Version
        # original filters, 5 5 5 9 9 9
        # changed filters, 3 3 3 5 5 5

        self.conva1 = nn.Conv2d(1, 64, config['num_filters'], dilation=2)

        self.poola = nn.MaxPool2d(2, 2)
        # self.bn1 = nn.BatchNorm2d(64,eps = 1)
        self.conva2 = nn.Conv2d(64, 128, config['num_filters'])
        self.poola1 = nn.MaxPool2d(2, 2)
        # self.bn2 = nn.BatchNorm2d(128,eps = 1)
        self.conva3 = nn.Conv2d(128, 256, config['num_filters'])
        # self.bn3 = nn.BatchNorm2d(256,eps = 1)
        self.conva4 = nn.Conv2d(256, 512, config['num_filters_FC'])
        # self.bn4 = nn.BatchNorm2d(512,eps = 1)
        # Last Layers
        self.upSam = nn.Upsample(scale_factor=2, mode='bilinear')

        self.convd1 = nn.Conv2d(512, 512, config['num_filters_FC'])
        # self.bn5 = nn.BatchNorm2d(512,eps = 1)
        self.convd2 = nn.Conv2d(512, 1, config['num_filters_FC'])

    def forward(self, x):
        img_xformer = torchvision.transforms.Resize((x.shape[2] * 3, x.shape[3] * 3))
        x = img_xformer(x)
        x1 = x
        # img_xformer2 = torchvision.transforms.Resize((x.shape[2] // 4, x.shape[3] // 4))
        x = self.conva1(x)
        x = F.relu(x)
        # x = self.bn1(x)
        x = self.poola(x)

        x = self.conva2(x)
        x = F.relu(x)
        # x = self.bn2(x)
        x = self.poola1(x)

        x = self.conva3(x)
        x = F.relu(x)
        # x = self.bn3(x)

        x = self.conva4(x)

        x = F.relu(x)
        # x = self.bn4(x)

        x = self.upSam(x)
        x = self.convd1(x)
        x = F.relu(x)
        # x = self.bn5(x)

        x = self.convd2(x)

        finalXformer = torchvision.transforms.Resize((60, 60))
        x = finalXformer(x)

        return x


class SingleBranchHpLp(nn.Module):

    def __init__(self, config):
        super(SingleBranchHpLp, self).__init__()
        # Computational Graph Version
        # original filters, 5 5 5 9 9 9
        # changed filters, 3 3 3 5 5 5

        self.conva1 = nn.Conv2d(1, 64, config['num_filters'])
        self.poola = nn.MaxPool2d(2, 2)
        # self.bn1 = nn.BatchNorm2d(64,eps = 1)
        self.conva2 = nn.Conv2d(64, 128, config['num_filters'])
        self.poola1 = nn.MaxPool2d(2, 2)
        # self.bn2 = nn.BatchNorm2d(128,eps = 1)
        self.conva3 = nn.Conv2d(128, 256, config['num_filters'])
        # self.bn3 = nn.BatchNorm2d(256,eps = 1)
        # self.conva4 = nn.Conv2d(256, 512, config['num_filters_FC'])
        # self.bn4 = nn.BatchNorm2d(512,eps = 1)

        self.convb1 = nn.Conv2d(1, 64, config['num_filters'])
        self.poolb = nn.MaxPool2d(2, 2)
        # self.bn1 = nn.BatchNorm2d(64,eps = 1)
        self.convb2 = nn.Conv2d(64, 128, config['num_filters'])
        self.poolb1 = nn.MaxPool2d(2, 2)
        # self.bn2 = nn.BatchNorm2d(128,eps = 1)
        self.convb3 = nn.Conv2d(128, 256, config['num_filters'])
        # self.bn3 = nn.BatchNorm2d(256,eps = 1)

        # self.convb4 = nn.Conv2d(256, 512, config['num_filters_FC'])
        # self.bn4 = nn.BatchNorm2d(512,eps = 1)
        # Last Layers
        self.upSam = nn.Upsample(scale_factor=2, mode='bilinear')

        self.convd1 = nn.Conv2d(512, 512, config['num_filters_FC'])
        # self.bn5 = nn.BatchNorm2d(512,eps = 1)
        self.convd2 = nn.Conv2d(512, 1, config['num_filters_FC'])

    def forward(self, x):
        img_xformer = torchvision.transforms.Resize((x.shape[3] * 3, x.shape[4] * 3))
        x1 = img_xformer(x[:, :, 0, :, :])
        x2 = img_xformer(x[:, :, 1, :, :])
        x = None
        # img_xformer2 = torchvision.transforms.Resize((x.shape[2] // 4, x.shape[3] // 4))

        # HighPass
        x1 = self.conva1(x1)
        x1 = F.relu(x1)
        x1 = self.poola(x1)

        x1 = self.conva2(x1)
        x1 = F.relu(x1)
        x1 = self.poola1(x1)

        x1 = self.conva3(x1)
        x1 = F.relu(x1)

        # lowpass
        x2 = self.convb1(x2)
        x2 = F.relu(x2)
        x2 = self.poolb(x2)

        x2 = self.convb2(x2)
        x2 = F.relu(x2)
        x2 = self.poolb1(x2)

        x2 = self.convb3(x2)
        x2 = F.relu(x2)

        # recombine

        x = torch.cat((x1, x2), dim=1)

        x = self.upSam(x)
        x = self.convd1(x)
        x = F.relu(x)
        # x = self.bn5(x)

        x = self.convd2(x)

        finalXformer = torchvision.transforms.Resize((60, 60))
        x = finalXformer(x)

        return x


class ModelBasedOnPaperGitHubVersion(nn.Module):

    def __init__(self, config):
        super(ModelBasedOnPaperGitHubVersion, self).__init__()
        # Computational Graph Version

        # Full Res
        self.conva1 = nn.Conv2d(1, 64, config['num_filters'])
        self.poola = nn.MaxPool2d(2, 2)
        self.conva2 = nn.Conv2d(64, 128, config['num_filters'])
        self.poola1 = nn.MaxPool2d(2, 2)
        self.conva3 = nn.Conv2d(128, 256, config['num_filters'])
        # self.conva4 = nn.Conv2d(256, 512, config['num_filters_FC'])

        # Half Res
        self.convb1 = nn.Conv2d(1, 64, config['num_filters'], 1)
        self.poolb = nn.MaxPool2d(2, 2)
        self.convb2 = nn.Conv2d(64, 128, config['num_filters'])
        self.poolb1 = nn.MaxPool2d(2, 2)
        self.convb3 = nn.Conv2d(128, 256, config['num_filters'])
        # self.convb4 = nn.Conv2d(256, 512, config['num_filters_FC'])

        # Quarter Res

        self.convc1 = nn.Conv2d(1, 64, config['num_filters'] - 1, 1)
        self.poolc = nn.MaxPool2d(2, 2)
        self.convc2 = nn.Conv2d(64, 128, config['num_filters'] - 1)
        self.poolc1 = nn.MaxPool2d(2, 2)
        self.convc3 = nn.Conv2d(128, 256, config['num_filters'] - 1)
        # self.convc4 = nn.Conv2d(256, 512,config['num_filters_FC'])

        # Last Layers

        self.convd1 = nn.Conv2d(768, 512, config['num_filters_FC'])
        self.convd2 = nn.Conv2d(512, 1, config['num_filters_FC'])

        # self.convd3 = nn.Conv2d(1, 1,1)

    def forward(self, x):
        img_xformer = torchvision.transforms.Resize((x.shape[2] * 2, x.shape[3] * 2))
        x = img_xformer(x)
        img_xformer = None
        img_xformer1 = torchvision.transforms.Resize(((x.shape[2]) // 2, (x.shape[3]) // 2))
        img_xformer2 = torchvision.transforms.Resize(((x.shape[2]) // 4, (x.shape[3]) // 4))

        x1 = x

        x1 = self.conva1(x1)
        x1 = F.relu(x1)
        x1 = self.poola(x1)
        x1 = self.conva2(x1)
        x1 = F.relu(x1)
        x1 = self.poola1(x1)
        x1 = self.conva3(x1)
        x1 = F.relu(x1)
        # x1 = self.conva4(x1)
        # x1 = F.relu(x1)

        img_xformer1transpose = torchvision.transforms.Resize((x1.shape[2], x1.shape[3]))

        x2 = img_xformer1(x)
        img_xformer1 = None
        x2 = self.convb1(x2)
        x2 = F.relu(x2)
        x2 = self.poolb(x2)
        x2 = self.convb2(x2)
        x2 = F.relu(x2)
        x2 = self.poolb1(x2)
        x2 = self.convb3(x2)
        x2 = F.relu(x2)
        # x2 = self.convb4(x2)
        # x2 = F.relu(x2)
        x2 = img_xformer1transpose(x2)

        x3 = img_xformer2(x)
        img_xformer2 = None
        x3 = self.convc1(x3)
        x3 = F.relu(x3)
        x3 = self.convc2(x3)
        x3 = F.relu(x3)
        x3 = self.poolc1(x3)
        x3 = self.convc3(x3)
        x3 = F.relu(x3)
        # x3 = self.convc4(x3)
        # x3 = F.relu(x3)
        x3 = img_xformer1transpose(x3)

        # x = x1 + x2  + x3
        # x /= 2
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.convd1(x)
        x = F.relu(x)
        x = self.convd2(x)
        finalXformer = torchvision.transforms.Resize((60, 60))
        x = finalXformer(x)
        # x = self.convd3(x)

        return x


class ModelBasedOnPaperNoneSpherical(nn.Module):

    def __init__(self, bandwidth=30):
        super(ModelBasedOnPaperNoneSpherical, self).__init__()

        self.pool = nn.MaxPool2d(2)

        self.conv1 = nn.Conv2d(1, 128, 5, padding=2)
        self.conv2 = nn.Conv2d(128, 128, 5)
        self.conv3 = nn.Conv2d(128, 128, 5)

        self.convFullyConnected1 = nn.Conv2d(128, 512, 9)
        self.convFullyConnected2 = nn.Conv2d(512, 256, 1)
        self.convFullyConnected3 = nn.Conv2d(256, 256, 1)

        self.upsample1 = nn.ConvTranspose2d(256, 128, 5, stride=5)  # ,padding =2, output_padding=1)

        self.upsample2 = nn.ConvTranspose2d(128, 64, 6, stride=6)
        self.upsample3 = nn.ConvTranspose2d(64, 1, 2, stride=2)

    def forward(self, x):
        # residual1 = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)  # 30 30
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)  # 15 15
        x = F.relu(x)

        x = self.convFullyConnected1(x)
        x = F.relu(x)

        x = self.convFullyConnected2(x)
        x = F.relu(x)

        x = self.convFullyConnected3(x)
        x = F.relu(x)

        x = self.upsample1(x)

        x = F.relu(x)

        # 15 15
        x = self.upsample2(x)
        # 30 30
        x = F.relu(x)

        x = self.upsample3(x)
        # x = x + residual1
        x = F.relu(x)
        return x


class FirstWorkingModelForPos_NoDense(nn.Module):

    def __init__(self, config):
        super(FirstWorkingModelForPos_NoDense, self).__init__()

        self.conva1 = nn.Conv2d(1, 8, config['num_filters'], padding=2, padding_mode=config['padding_mode'])
        self.poola = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(8)

        self.conva2 = nn.Conv2d(8, 32, config['num_filters'], padding=2, padding_mode=config['padding_mode'])
        self.poola1 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conva3 = nn.Conv2d(32, 128, config['num_filters'], padding=2, padding_mode=config['padding_mode'])
        self.poola2 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.conva4 = nn.Conv2d(128, 256, config['num_filters'], padding=2, padding_mode=config['padding_mode'])
        self.bn4 = nn.BatchNorm2d(256)
        self.conva5 = nn.Conv2d(256, 512, config['num_filters_FC'], padding=2, padding_mode=config['padding_mode'])
        # Last Layers

        self.upSam = nn.Upsample(scale_factor=2, mode=config['upsample'])
        self.convd1 = nn.Conv2d(512, 512, config['num_filters_FC'])

        self.convd2 = nn.Conv2d(512, 1, config['num_filters_FC'])

    def forward(self, x):
        img_xformer = torchvision.transforms.Resize((x.shape[2] * 3, x.shape[3] * 3))
        x = img_xformer(x)
        # img_xformer2 = torchvision.transforms.Resize((x.shape[2] // 4, x.shape[3] // 4))
        x = self.conva1(x)
        x = F.relu(x)
        x = self.poola(x)
        x = self.bn1(x)

        x = self.conva2(x)
        x = F.relu(x)
        x = self.poola1(x)
        x = self.bn2(x)

        x = self.conva3(x)
        x = F.relu(x)
        x = self.poola2(x)
        x = self.bn3(x)

        x = self.conva4(x)
        x = F.relu(x)
        x = self.bn4(x)

        x = self.conva5(x)
        x = F.relu(x)

        x = self.upSam(x)
        x = self.convd1(x)
        x = F.relu(x)

        x = self.convd2(x)
        x = F.relu(x)

        finalXformer = torchvision.transforms.Resize((60, 60))
        x = finalXformer(x)

        return x


class BestPosModel_WithDenseLayer(nn.Module):

    def __init__(self, config):
        super(BestPosModel_WithDenseLayer, self).__init__()

        self.conva1 = nn.Conv2d(1, 8, config['num_filters'], padding=2, padding_mode=config['padding_mode'])
        self.poola = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(8)

        self.conva2 = nn.Conv2d(8, 32, config['num_filters'], padding=2, padding_mode=config['padding_mode'])
        self.poola1 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conva3 = nn.Conv2d(32, 128, config['num_filters'], padding=2, padding_mode=config['padding_mode'])
        self.poola2 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.conva4 = nn.Conv2d(128, 256, config['num_filters'], padding=2, padding_mode=config['padding_mode'])
        self.bn4 = nn.BatchNorm2d(256)

        self.upSam = nn.Upsample(scale_factor=2, mode=config['upsample'])
        self.convd1 = nn.Conv2d(256, 256, config['num_filters_FC'])  # ?? needed?

        self.convd2 = nn.Conv2d(256, 32, config['num_filters_FC'])

        self.dropout1 = nn.Dropout(p=config['dropout'])
        self.fc1 = nn.Linear(82944 // 2, 1800)
        self.fc3 = nn.Linear(1800, 3600)

        self.gaussian_filter = self.create_gaussian()
        self.leakyRelu = nn.LeakyReLU(config['leakySlope'])

    def create_gaussian(self):
        kernel_size = 5
        sigma = 3
        channels = 1

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        return gaussian_filter

    def forward(self, x):
        img_xformer = torchvision.transforms.Resize((x.shape[2] * 3, x.shape[3] * 3))
        x = img_xformer(x)
        x = self.conva1(x)
        x = self.leakyRelu(x)
        x = self.poola(x)
        x = self.bn1(x)

        x = self.conva2(x)
        x = self.leakyRelu(x)
        x = self.poola1(x)
        x = self.bn2(x)

        x = self.conva3(x)
        x = self.leakyRelu(x)
        x = self.poola2(x)
        x = self.bn3(x)

        x = self.conva4(x)
        x = self.leakyRelu(x)
        x = self.bn4(x)

        x = self.upSam(x)
        x = self.convd1(x)
        x = self.leakyRelu(x)

        x = self.convd2(x)
        x = self.leakyRelu(x)  # for 60x60 input, here we will have 16,64,36,36

        # dense Layers
        sz = x.size(0)
        x = x.view(sz, -1)

        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.leakyRelu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = x.view(sz, 1, 60, 60)

        x = self.gaussian_filter(x)

        finalXformer = torchvision.transforms.Resize((60, 60))
        x = finalXformer(x)

        return x


class testModel(nn.Module):

    def __init__(self, config):
        super(testModel, self).__init__()

        self.conva1 = nn.Conv2d(1, 8, config['num_filters'], padding=2, padding_mode=config['padding_mode'])
        self.poola = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(8)

        self.conva2 = nn.Conv2d(8, 32, config['num_filters'], padding=2, padding_mode=config['padding_mode'])
        self.poola1 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conva3 = nn.Conv2d(32, 128, config['num_filters'], padding=2, padding_mode=config['padding_mode'])
        self.poola2 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.conva4 = nn.Conv2d(128, 256, config['num_filters'], padding=2, padding_mode=config['padding_mode'])
        self.bn4 = nn.BatchNorm2d(256)

        self.upSam = nn.Upsample(scale_factor=2, mode=config['upsample'])
        self.convd1 = nn.Conv2d(256, 256, config['num_filters_FC'])

        self.convd2 = nn.Conv2d(256, 32, config['num_filters_FC'])

        self.dropout1 = nn.Dropout(p=config['dropout'])
        self.fc1 = nn.Linear(82944 // 2, 3600)
        self.fc3 = nn.Linear(3600, 3600)

        self.gaussian_filter = self.create_gaussian()
        self.leakyRelu = nn.LeakyReLU(config['leakySlope'])

    def create_gaussian(self):
        kernel_size = 5
        sigma = 3
        channels = 1

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        return gaussian_filter

    def forward(self, x):
        img_xformer = torchvision.transforms.Resize((x.shape[2] * 3, x.shape[3] * 3))
        x = img_xformer(x)
        # img_xformer2 = torchvision.transforms.Resize((x.shape[2] // 4, x.shape[3] // 4))
        x = self.conva1(x)
        x = self.leakyRelu(x)
        x = self.poola(x)
        x = self.bn1(x)

        x = self.conva2(x)
        x = self.leakyRelu(x)
        x = self.poola1(x)
        x = self.bn2(x)

        x = self.conva3(x)
        x = self.leakyRelu(x)
        x = self.poola2(x)
        x = self.bn3(x)

        x = self.conva4(x)
        x = self.leakyRelu(x)
        x = self.bn4(x)

        x = self.upSam(x)
        x = self.convd1(x)
        x = self.leakyRelu(x)

        x = self.convd2(x)
        x = self.leakyRelu(x)  # for 60x60 input, here we will have 16,64,36,36

        # dense Layers
        sz = x.size(0)
        x = x.view(sz, -1)

        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.leakyRelu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = x.view(sz, 1, 60, 60)

        x = self.gaussian_filter(x)

        finalXformer = torchvision.transforms.Resize((60, 60))
        x = finalXformer(x)

        return x

if __name__ == '__main__':
    pass