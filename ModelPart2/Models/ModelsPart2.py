import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms
import math

class DilatedCNN(nn.Module):

    def __init__(self, config):
        super(DilatedCNN, self).__init__()

        self.conva1 = nn.Conv2d(2, 8, config['num_filters'], padding=2, padding_mode=config['padding_mode'], dilation=2)
        # self.poola = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(8)

        self.conva2 = nn.Conv2d(8, 32, config['num_filters'], padding=2, padding_mode=config['padding_mode'],
                                dilation=2)
        # self.poola1 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conva3 = nn.Conv2d(32, 128, config['num_filters'], padding=2, padding_mode=config['padding_mode'],
                                dilation=2)
        # self.poola2 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.convd1 = nn.Conv2d(128, 256, config['num_filters_FC'], padding=4, dilation=4)
        self.bn4 = nn.BatchNorm2d(256)
        self.convd2 = nn.Conv2d(256, 512, config['num_filters_FC'], padding=4, dilation=4)
        # self.convd3 = nn.Conv2d(512, 32, config['num_filters_FC'], padding=4, dilation = 4)
        self.convd3 = nn.Conv2d(512, 32, 1)
        self.bn5 = nn.BatchNorm2d(32)

        self.finalConv = nn.Conv2d(64, 6, config['num_filters_FC'], padding=4, dilation=4)

        # self.dropout1 = nn.Dropout(p=config['dropout'])
        # self.fc1 = nn.Linear(10368, 5400)
        # self.fc3 = nn.Linear(5400, 5400)

        self.gaussian_filter = self.create_gaussian()
        self.leakyRelu = nn.LeakyReLU(config['leakySlope'])

    def create_gaussian(self):
        kernel_size = 3
        sigma = 3
        channels = 6

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
                                    kernel_size=kernel_size, groups=channels, bias=False, padding=1)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        return gaussian_filter

    def forward(self, x):
        # img_xformer = torchvision.transforms.Resize((x.shape[2] * 3, x.shape[3] * 3))
        # x = img_xformer(x)
        x = self.conva1(x)
        x = self.leakyRelu(x)
        # x = self.poola(x)
        x = self.bn1(x)

        x = self.conva2(x)

        # x1 = self.poola1(x1)

        x = self.leakyRelu(x)
        # x = self.poola1(x)
        x = self.bn2(x)
        x1 = x

        x = self.conva3(x)
        x = self.leakyRelu(x)
        # x = self.poola2(x)
        x = self.bn3(x)

        x = self.convd1(x)
        x = self.leakyRelu(x)
        x = self.bn4(x)

        x = self.convd2(x)
        x = self.leakyRelu(x)

        x = self.convd3(x)
        x = self.leakyRelu(x)
        x = self.bn5(x)
        # resizer = torchvision.transforms.Resize((45, 45))
        # x = resizer(x)
        x = torch.cat((x, x1), dim=1)

        x = self.finalConv(x)
        x = F.relu(x)

        x = self.gaussian_filter(x)
        x = F.relu(x)

        # finalXformer = torchvision.transforms.Resize((60, 60))
        # x = finalXformer(x)

        return x


class SignificantlySimple(nn.Module):

    def __init__(self, config):
        super(SignificantlySimple, self).__init__()

        self.conva1 = nn.Conv2d(2, 8, 3, padding=int(math.floor(3 / 2)), padding_mode=config['padding_mode'])
        self.poola = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(8)

        self.conva2 = nn.Conv2d(8, 32, 7, padding=int(math.floor(7 / 2)),
                                padding_mode=config['padding_mode'])
        self.poola = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(7200, 7200)
        self.bn3 = nn.BatchNorm1d(7200)
        self.fc2 = nn.Linear(7200, 1800)

    def create_gaussian(self):
        kernel_size = 3
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
                                    kernel_size=kernel_size, groups=channels, bias=False, padding=1)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        return gaussian_filter

    def forward(self, x):
        x = self.conva1(x)
        x = F.relu(x)
        x = self.poola(x)
        x = self.bn1(x)

        x = self.conva2(x)
        x = F.relu(x)
        x = self.poola(x)
        x = self.bn2(x)

        sz = x.size(0)
        x = x.view(sz, -1)

        x = self.fc1(x)
        x = self.bn3(x)
        x = self.fc2(x)

        x = x.view(sz, 2, 30, 30)
        finalXformer = torchvision.transforms.Resize((60, 60))
        x = finalXformer(x)

        return x


class OrientationTestClass(nn.Module):

    def __init__(self, config):
        super(OrientationTestClass, self).__init__()

        self.conva1 = nn.Conv2d(2, 8, 3, padding=int(math.floor(config['num_filters'] / 2)),
                                padding_mode=config['padding_mode'])
        self.poola = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(8)

        self.conva2 = nn.Conv2d(8, 32, config['num_filters'], padding=int(math.floor(config['num_filters'] / 2)),
                                padding_mode=config['padding_mode'])
        self.poola1 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conva3 = nn.Conv2d(32, 128, config['num_filters'], padding=int(math.floor(config['num_filters'] / 2)),
                                padding_mode=config['padding_mode'])
        self.poola2 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.convd1 = nn.Conv2d(128, 256, config['num_filters_FC'],
                                padding=int(math.floor(config['num_filters_FC'] / 2)))
        self.bn4 = nn.BatchNorm2d(256)
        self.convd2 = nn.Conv2d(256, 512, config['num_filters_FC'],
                                padding=int(math.floor(config['num_filters_FC'] / 2)))

        self.convd3 = nn.Conv2d(512, 32, 1)
        # adds in output of conva2
        self.finalConv = nn.Conv2d(64, 2, 1)

        # self.dropout1 = nn.Dropout(p=config['dropout'])
        # self.fc1 = nn.Linear(10368, 5400)
        # self.fc3 = nn.Linear(5400, 5400)

        self.gaussian_filter = self.create_gaussian()
        self.leakyRelu = nn.LeakyReLU(config['leakySlope'])

    def create_gaussian(self):
        kernel_size = 3
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
                                    kernel_size=kernel_size, groups=channels, bias=False, padding=1)

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
        x1 = x
        x1 = self.poola1(x1)

        x = self.leakyRelu(x)
        x = self.poola1(x)
        x = self.bn2(x)

        x = self.conva3(x)
        x = self.leakyRelu(x)
        x = self.poola2(x)
        x = self.bn3(x)

        x = self.convd1(x)
        x = self.leakyRelu(x)
        x = self.bn4(x)

        x = self.convd2(x)
        x = self.leakyRelu(x)

        x = self.convd3(x)
        x = self.leakyRelu(x)
        resizer = torchvision.transforms.Resize((45, 45))
        x = resizer(x)
        x = torch.cat((x, x1), dim=1)

        x = self.finalConv(x)
        x = F.relu(x)

        # dense Layers
        # sz = x.size(0)
        # x = x.view(sz, -1)

        # x = self.dropout1(x)
        # x = self.fc1(x)
        # x = self.leakyRelu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        # x = x.view(sz, 6, 30, 30)
        xn = torch.split(x, 1, dim=1)
        x1 = xn[0]
        x2 = xn[1]
        xn = None
        x2 = self.gaussian_filter(x2)

        x2 = F.relu(x2)
        x = torch.cat((x1, x2), dim=1)

        finalXformer = torchvision.transforms.Resize((60, 60))
        x = finalXformer(x)

        return x


class RBranchDCNN(nn.Module):

    def __init__(self, config=None):
        super(RBranchDCNN, self).__init__()

        self.conv13 = nn.Conv2d(2, 64, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv21 = nn.Conv2d(96, 64, 5, padding=2)
        self.bn21 = nn.BatchNorm2d(64)
        self.conv22 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn22 = nn.BatchNorm2d(64)
        self.conv23 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn23 = nn.BatchNorm2d(64)
        self.conv24 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn24 = nn.BatchNorm2d(128)
        self.conv25 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn25 = nn.BatchNorm2d(128)
        self.conv26 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn26 = nn.BatchNorm2d(128)
        self.conv27 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn27 = nn.BatchNorm2d(64)

        self.Unpool1 = nn.MaxUnpool2d(2, 2)
        self.deconv1 = nn.ConvTranspose2d(64, 2, 5, padding=2)

        self.convB11 = nn.Conv2d(64, 32, 3, padding=2)
        self.bn11 = nn.BatchNorm2d(32)
        self.poolB1 = nn.MaxPool2d(2, 2, return_indices=True)
        self.convB12 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(32)
        self.UnpoolB1 = nn.MaxUnpool2d(2, 2)
        self.deconvB11 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.bnb11 = nn.BatchNorm2d(32)

        self.convB21 = nn.Conv2d(32, 32, 3, padding=1)
        self.bnb21 = nn.BatchNorm2d(32)
        self.poolB2 = nn.MaxPool2d(2, 2, return_indices=True)
        self.convB22 = nn.Conv2d(32, 32, 3, padding=1)
        self.bnb22 = nn.BatchNorm2d(32)
        self.UnpoolB2 = nn.MaxUnpool2d(2, 2)
        self.deconvB21 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.bnb21 = nn.BatchNorm2d(32)

        self.gaussian_filter = self.create_gaussian()
        self.sigmoid = nn.Sigmoid()

    def create_gaussian(self):
        kernel_size = 5
        sigma = 3
        channels = 2

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
                                    kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size // 2)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        return gaussian_filter

    def forward(self, x):
        x = self.conv13(x)
        x = F.relu(x)
        x, indicesX = self.pool1(x)
        x = self.bn1(x)
        # go to branches. branch 1 = x1, branch 2 = x2
        x1 = x
        x1 = self.convB11(x1)
        x1 = F.relu(x1)
        x1, indicesX1 = self.poolB1(x1)
        x1 = self.bn11(x1)
        # x2
        x2 = x1
        x2 = self.convB21(x2)
        x2 = F.relu(x2)
        x2, indicesX2 = self.poolB2(x2)
        x2 = self.bnb21(x2)
        x2 = self.convB22(x2)
        x2 = F.relu(x2)
        x2 = self.bnb22(x2)
        # x2finalXformer = torchvision.transforms.Resize((x1.shape[2], x1.shape[3]))
        # x2 = x2finalXformer(x2)
        x2 = self.UnpoolB2(x2, indicesX2)
        x2 = self.deconvB21(x2)
        x2 = F.relu(x2)
        x2 = self.bnb21(x2)
        x2finalXformer = torchvision.transforms.Resize((x1.shape[2], x1.shape[3]))
        x2 = x2finalXformer(x2)
        # end of branch 2

        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.convB12(x1)
        x1 = F.relu(x1)
        x1 = self.bn12(x1)
        x1 = self.UnpoolB1(x1, indicesX1)
        x1 = self.deconvB11(x1)
        x1 = F.relu(x1)
        x1 = self.bnb11(x1)
        x1finalXformer = torchvision.transforms.Resize((x.shape[2], x.shape[3]))
        x1 = x1finalXformer(x1)

        x = torch.cat((x, x1), dim=1)
        x = self.conv21(x)
        x = F.relu(x)
        x = self.bn21(x)
        x = self.conv22(x)
        x = F.relu(x)
        x = self.bn22(x)
        x = self.conv23(x)
        x = F.relu(x)
        x = self.bn23(x)

        x = self.conv24(x)
        x = F.relu(x)
        x = self.bn24(x)
        x = self.conv25(x)
        x = F.relu(x)
        x = self.bn25(x)
        x = self.conv26(x)
        x = F.relu(x)
        x = self.bn26(x)
        x = self.conv27(x)
        x = F.relu(x)
        x = self.bn27(x)

        x = self.Unpool1(x, indicesX)

        x = self.deconv1(x)
        x = self.gaussian_filter(x)
        x = self.sigmoid(x)

        return x


class RBranchEarlyExit(nn.Module):

    def __init__(self, config=None):
        super(RBranchEarlyExit, self).__init__()
        self.maskvalue = config['maskvalue']

        self.conv13 = nn.Conv2d(2, 64, 5, padding=2) # 2 for pos map input. 1 for no pos map input
        self.sigma = config['gaussianSigma']
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)
        self.dropoutlayer = nn.Dropout2d(p=config['dropout'])

        self.conv21 = nn.Conv2d(96, 64, 5, padding=2)
        self.bn21 = nn.BatchNorm2d(64)
        self.conv22 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
        self.bn22 = nn.BatchNorm2d(64)
        self.conv23 = nn.Conv2d(64, 64, 3, padding=4, dilation=4)
        self.bn23 = nn.BatchNorm2d(64)

        self.conv_deconv1AngThetaPhi = nn.Conv2d(64, 2, 3, padding=1)
        self.conv_deconv1Gamma = nn.Conv2d(64, 1, 3, padding=1)
        self.conv_deconv1Hm = nn.Conv2d(64, 1, 3, padding=1)

        self.convB11 = nn.Conv2d(64, 32, 3, padding=2)
        self.bn11 = nn.BatchNorm2d(32)
        self.poolB1 = nn.MaxPool2d(2, 2, return_indices=True)
        self.convB12 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(32)

        self.conv_deconvreplaceB11 = nn.Conv2d(32, 32, 3, padding=1)
        self.bnb11 = nn.BatchNorm2d(32)

        self.convB21 = nn.Conv2d(32, 32, 3, padding=1)
        self.bnb21 = nn.BatchNorm2d(32)
        self.poolB2 = nn.MaxPool2d(2, 2, return_indices=True)
        self.convB22 = nn.Conv2d(32, 32, 3, padding=1)
        self.bnb22 = nn.BatchNorm2d(32)

        self.conv_deconvreplaceB21 = nn.Conv2d(32, 32, 3, padding=1)
        self.bnb21 = nn.BatchNorm2d(32)

        self.gaussian_filter1 = self.create_gaussian(channelsize=1, filtersize=5)

        self.sigmoid = nn.Sigmoid()
        self.leakyRelu = nn.LeakyReLU(config['leakySlope'])
        self.tanh = nn.Tanh()

    def create_gaussian(self, channelsize, filtersize, learnable_bias=False):
        kernel_size = filtersize

        channels = channelsize

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = self.sigma ** 2.

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
                                    kernel_size=kernel_size, groups=channels, bias=learnable_bias,
                                    padding=kernel_size // 2)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        return gaussian_filter

    def forward(self, x):
        x = self.conv13(x)
        x = self.leakyRelu(x)
        x, indicesX = self.pool1(x)
        x = self.bn1(x)
        # go to branches. branch 1 = x1, branch 2 = x2
        x1 = x
        x1 = self.convB11(x1)
        x1 = self.leakyRelu(x1)
        x1, indicesX1 = self.poolB1(x1)
        x1 = self.bn11(x1)
        # x2
        x2 = x1
        x2 = self.convB21(x2)
        x2 = self.leakyRelu(x2)
        x2, indicesX2 = self.poolB2(x2)
        x2 = self.bnb21(x2)
        x2 = self.convB22(x2)
        x2 = self.leakyRelu(x2)
        x2 = self.bnb22(x2)
        x2finalXformer = torchvision.transforms.Resize((x1.shape[2], x1.shape[3]))
        x2 = x2finalXformer(x2)

        x2 = self.conv_deconvreplaceB11(x2)
        x2 = self.leakyRelu(x2)
        x2 = self.bnb21(x2)
        # end of branch 2

        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.convB12(x1)
        x1 = self.leakyRelu(x1)
        x1 = self.bn12(x1)

        x1finalXformer = torchvision.transforms.Resize((x.shape[2], x.shape[3]))
        x1 = x1finalXformer(x1)
        x1 = self.conv_deconvreplaceB21(x1)

        x1 = self.leakyRelu(x1)
        x1 = self.bnb11(x1)

        x = torch.cat((x, x1), dim=1)
        #x = self.dropoutlayer(x)  # spatial dropout layer
        x = self.conv21(x)
        x = self.leakyRelu(x)
        x = self.bn21(x)
        x = self.conv22(x)
        x = self.leakyRelu(x)
        x = self.bn22(x)
        x = self.conv23(x)
        x = self.leakyRelu(x)
        x = self.bn23(x)

        # exit and unpool for HM
        x3finalXformer = torchvision.transforms.Resize((60, 60))
        x = x3finalXformer(x)
        x3 = x

        x3 = self.conv_deconv1Hm(x3)
        x3 = self.sigmoid(x3)
        x3 = self.gaussian_filter1(x3)
        #x3 = (1 / torch.amax(x3)) * x3

        x_gamma = self.conv_deconv1Gamma(x)
        x_gamma = self.sigmoid(x_gamma) * np.pi

        x_Theta_phi = self.conv_deconv1AngThetaPhi(x)
        x_Theta_phi = self.tanh(x_Theta_phi) * np.pi

        x = torch.cat((x_Theta_phi, x_gamma), dim=1)

        filterX3 = x3 < self.maskvalue  # only areas with low probability selected

        x0_temp = x[:, 0, :, :]
        x0_temp = x0_temp[:, None, :, :]
        x0_temp[filterX3] = 0

        x1_temp = x[:, 1, :, :]
        x1_temp = x1_temp[:, None, :, :]
        x1_temp[filterX3] = 0

        x2_temp = x[:, 2, :, :]
        x2_temp = x2_temp[:, None, :, :]
        x2_temp[filterX3] = 0

        x = torch.cat((x0_temp, x1_temp, x2_temp), dim=1)

        x = torch.cat((x, x3), dim=1)

        return x


class HMToAngle(nn.Module):

    def __init__(self, config):
        super(HMToAngle, self).__init__()

        self.conva1 = nn.Conv2d(2, 8, 3, padding=int(math.floor(config['num_filters'] / 2)),
                                padding_mode=config['padding_mode'])
        self.poola = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(8)

        self.conva2 = nn.Conv2d(8, 32, config['num_filters'], padding=int(math.floor(config['num_filters'] / 2)),
                                padding_mode=config['padding_mode'])
        self.poola1 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conva3 = nn.Conv2d(32, 128, config['num_filters'], padding=int(math.floor(config['num_filters'] / 2)),
                                padding_mode=config['padding_mode'])
        # residual Connect1
        self.poola2 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.convHm = nn.Conv2d(128, 3, 1)
        # concat here with spherical Data

        self.convd1 = nn.Conv2d(4, 16, config['num_filters_FC'], padding=int(math.floor(config['num_filters_FC'] / 2)))
        self.bn4 = nn.BatchNorm2d(16)
        self.convd2 = nn.Conv2d(16, 64, config['num_filters_FC'], padding=int(math.floor(config['num_filters_FC'] / 2)))
        self.bn5 = nn.BatchNorm2d(64)

        self.convd3 = nn.Conv2d(64, 128, config['num_filters_FC'],
                                padding=int(math.floor(config['num_filters_FC'] / 2)))
        # residual Connect 1
        self.bn6 = nn.BatchNorm2d(256)
        # adds in output of conva2
        self.finalConv = nn.Conv2d(259, 6, 1)

        # self.dropout1 = nn.Dropout(p=config['dropout'])
        # self.fc1 = nn.Linear(10368, 5400)
        # self.fc3 = nn.Linear(5400, 5400)

        self.gaussian_filter = self.create_gaussian()
        self.leakyRelu = nn.LeakyReLU(config['leakySlope'])

    def create_gaussian(self):
        kernel_size = 3
        sigma = 3
        channels = 3

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
                                    kernel_size=kernel_size, groups=channels, bias=False, padding=1)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        return gaussian_filter

    def forward(self, x):
        img_xformer = torchvision.transforms.Resize((x.shape[2] * 3, x.shape[3] * 3))
        x1 = torch.split(x, 1, dim=1)
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
        x2 = x
        x = self.leakyRelu(x)
        x = self.poola2(x)
        x = self.bn3(x)

        x = self.convHm(x)
        x = F.relu(x)
        xHm = x

        resizer = torchvision.transforms.Resize((60, 60))
        resizer2 = torchvision.transforms.Resize((30, 30))
        x = resizer(x)
        x2 = resizer2(x2)
        x = torch.cat((x, x1[1]), dim=1)  # 4x60x60

        x = self.convd1(x)
        x = self.leakyRelu(x)
        x = self.poola1(x)
        x = self.bn4(x)

        x = self.convd2(x)
        x = self.leakyRelu(x)
        x = self.bn5(x)

        x = self.convd3(x)
        x = torch.cat((x, x2), dim=1)
        x = self.leakyRelu(x)
        x = self.bn6(x)
        xHm = resizer2(xHm)
        x = torch.cat((xHm, x), dim=1)
        x = self.finalConv(x)
        x = F.relu(x)

        # dense Layers
        # sz = x.size(0)
        # x = x.view(sz, -1)

        # x = self.dropout1(x)
        # x = self.fc1(x)
        # x = self.leakyRelu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        # x = x.view(sz, 6, 30, 30)
        xn = torch.split(x, 3, dim=1)
        x1 = xn[0]
        x2 = xn[1]
        xn = None
        x2 = self.gaussian_filter(x2)

        x2 = F.relu(x2)
        x = torch.cat((x1, x2), dim=1)

        finalXformer = torchvision.transforms.Resize((60, 60))
        x = finalXformer(x)

        return x


if __name__ == '__main__':
    pass