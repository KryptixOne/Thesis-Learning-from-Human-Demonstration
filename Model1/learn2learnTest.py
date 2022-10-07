# !/usr/bin/env python3

"""
Trains a 3 layer MLP with MAML on Sine Wave Regression Dataset.
We use the Sine Wave dataloader from the torchmeta package.
Torchmeta: https://github.com/tristandeleu/pytorch-meta
"""

import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import learn2learn as l2l

from torch import nn, optim
import wandb
from torchmeta.toy import Sinusoid
from torchmeta.utils.data import BatchMetaDataLoader
from torch.utils.data import Dataset
from pathlib import Path
import pickle
import torch.nn.functional as F
import torchvision.transforms
import math
from random import shuffle
import shutil
from skimage.util import random_noise


class FolderData(Dataset):
    """Load Data

        DataPath should be towards the correct test/train/validation folder"""

    def __init__(self, DataPath, train_testFlag='train', ModelSelect='Part1', Singleimg=False, AddNoise=False,
                 ):
        path = Path(DataPath)
        self.ModelSelect = ModelSelect
        self.Singleimg = Singleimg
        self.AddNoise = AddNoise
        self.variance = 1e-4

        if not path.is_dir():
            raise RuntimeError(f'Invalid directory "{path}"')

        self.samples = []
        for e in path.iterdir():
            tempsamp = [f for f in e.iterdir() if f.is_file()]
            if train_testFlag == 'train':  # even distribution of training data
                # each category has 15 objects with 6 rotations
                tempsamp = tempsamp[0:90]
                self.samples = self.samples + tempsamp
            else:
                self.samples = self.samples + tempsamp
            # create subgroups based on the item
        if train_testFlag == 'train':
            self.samples = [self.samples[i:i + 6] for i in range(0, len(self.samples), 6)]
            # shuffle the items groupped in 6 and flatten list
            shuffle(self.samples)
            self.samples = [item for sublist in self.samples for item in sublist]

    def __getitem__(self, index):

        if self.ModelSelect == 'Part1':  # return Inputspherical + Target heatmaps
            with open(self.samples[index], 'rb') as f:
                curdata = pickle.load(f)

                depthimages = curdata['depthImages']
                posHeatMap = curdata['positionHeatMap']

                inputData = curdata['depthImages']
                inputData = torch.from_numpy(inputData[:, :, :].astype(np.float32))

                targetHM = curdata['positionHeatMap']
                targetHM = torch.from_numpy(targetHM[:, :, :].astype(np.float32))
                outputData = targetHM

        else:  # ModelSelect == 'Part2' return Input Sppherical + Positional HM , Target Orientation HM
            with open(self.samples[index], 'rb') as f:
                curdata = pickle.load(f)

            inputImg = curdata['depthImages']
            inputHM = curdata['positionHeatMap']

            inputImg = torch.from_numpy(inputImg[:, :, :].astype(np.float32))

            inputHM = torch.from_numpy(inputHM[:, :, :].astype(np.float32))

            outputposHM = curdata['positionHeatMap']

            inputData = torch.cat((inputImg, inputHM), dim=0)
            # make anngle data between 0 -180. If angle is 0, we make it 180

            ThetaAng = curdata['ThetaAng']
            # ThetaHm = curdata['ThetaHm']
            # ThetaHm[ThetaHm < 0.5] = 0

            PhiAngle = curdata['PhiAngle']
            # PhiHm = curdata['PhiHm']
            # PhiHm[PhiHm < 0.5] = 0

            GammaAngle = curdata['GammaAngle']
            Less_than = GammaAngle < 0
            GammaAngle[Less_than] = GammaAngle[Less_than] + 180

            # GammaHm = curdata['GammaHm']

            # GammaHm[GammaHm < 0.5] = 0

            ThetaAng = torch.from_numpy(ThetaAng[:, :, :].astype(np.float32))
            # ThetaHm = torch.from_numpy(ThetaHm[:, :, :].astype(np.float32))

            PhiAngle = torch.from_numpy(PhiAngle[:, :, :].astype(np.float32))
            # PhiHm = torch.from_numpy(PhiHm[:, :, :].astype(np.float32))

            GammaAngle = torch.from_numpy(GammaAngle[:, :, :].astype(np.float32))
            # GammaHm = torch.from_numpy(GammaHm[:, :, :].astype(np.float32))

            OutPosHm = torch.from_numpy(outputposHM[:, :, :].astype(np.float32))

            if self.Singleimg == False:
                # outputData = torch.cat((ThetaAng, PhiAngle, GammaAngle, ThetaHm, PhiHm, GammaHm, OutPosHm), dim=0)
                outputData = torch.cat((ThetaAng, PhiAngle, GammaAngle, OutPosHm), dim=0)
            else:
                # outputData = torch.cat((ThetaAng, ThetaHm), dim=0)
                pass
            # outputData = torch.cat((ThetaAng, PhiAngle, GammaAngle), dim=0)

        if self.AddNoise == True:
            inputData = inputData + (self.variance ** 0.5) * torch.rand(inputData.shape)
            # make sure data is capped at  despite addition of noise
            data1 = inputData > 1
            inputData[data1] = 1

        return (inputData, outputData)

    def __len__(self):
        return len(self.samples)


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

        self.conv13 = nn.Conv2d(2, 64, 5, padding=2)
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

        x_gamma = self.conv_deconv1Gamma(x)
        x_gamma = self.sigmoid(x_gamma) * np.pi

        x_Theta_phi = self.conv_deconv1AngThetaPhi(x)
        x_Theta_phi = self.tanh(x_Theta_phi) * np.pi

        x = torch.cat((x_Theta_phi, x_gamma), dim=1)

        filterX3 = x3 < 0.2  # only areas with low probability selected

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


VAL1SAME = 0
VAL2SAME = 0


def AugmentTasks(labels, NoiseToRegressionTargetAngles=False, LikelihoodsUpdates=False, UseSameNoise=False, ChooseRandomPoints=False, NumRandomPoints =None):
    augmentedLabels = labels.detach().clone()
    if NoiseToRegressionTargetAngles == True:
        ThetaAng = augmentedLabels[:, 0, :, :].detach().clone()  # -180 to 180
        noiseTheta = torch.randint_like(ThetaAng, low=-5, high=20)
        ThetaAng = ThetaAng + noiseTheta
        ThetaAng = torch.deg2rad(ThetaAng)

        PhiAngle = augmentedLabels[:, 1, :, :].detach().clone()
        noisePhi = torch.randint_like(ThetaAng, low=-5, high=20)
        PhiAngle = PhiAngle + noisePhi
        PhiAngle = torch.deg2rad(PhiAngle)

        GammaAngle = augmentedLabels[:, 2, :, :].detach().clone()
        noiseGamma = torch.randint_like(ThetaAng, low=-2, high=10)
        GammaAngle = GammaAngle + noiseGamma
        GammaAngle = torch.deg2rad(GammaAngle)

        augmentedLabels = torch.cat((ThetaAng[:, None, :, :], PhiAngle[:, None, :, :], GammaAngle[:, None, :, :],
                                     (augmentedLabels[:, 3, :, :])[:, None, :, :]), dim=1)
    else:
        ThetaAng = augmentedLabels[:, 0, :, :].detach().clone()  # -180 to 180
        ThetaAng = torch.deg2rad(ThetaAng)

        PhiAngle = augmentedLabels[:, 1, :, :].detach().clone()
        PhiAngle = torch.deg2rad(PhiAngle)

        GammaAngle = augmentedLabels[:, 2, :, :].detach().clone()
        GammaAngle = torch.deg2rad(GammaAngle)

        augmentedLabels = torch.cat((ThetaAng[:, None, :, :], PhiAngle[:, None, :, :], GammaAngle[:, None, :, :],
                                     (augmentedLabels[:, 3, :, :])[:, None, :, :]), dim=1)

    if LikelihoodsUpdates == True :
        outputHm = augmentedLabels[:, 3, :, :].detach().clone()
        global VAL2SAME
        global VAL1SAME
        if UseSameNoise == True:
            val1 = VAL1SAME
            val2 = VAL2SAME
        else:

            val1 = random.uniform(0, 1)
            val2 = random.uniform(val1 + 0.1, 1)

            VAL1SAME = val1
            VAL2SAME = val2

        boolHm1 = outputHm < val1
        boolHm2 = outputHm > val2
        outputHm[boolHm1] = 0
        outputHm[boolHm2] = 0
        augmentedLabels[:, 3, :, :] = outputHm[:, :, :]

        if ChooseRandomPoints == True:
            augmentedLabels[:, 3, :, :] = 1 / (torch.amax(augmentedLabels[:, 3, :, :])) * augmentedLabels[:, 3, :, :]
        else:
            # Rescale such that selected area is now the high probability area

            ThetaAng = augmentedLabels[:, 0, :, :].detach().clone()  # -180 to 180
            ThetaAng[boolHm1] = 0
            ThetaAng[boolHm2] = 0
            PhiAngle = augmentedLabels[:, 1, :, :].detach().clone()
            PhiAngle[boolHm1] = 0
            PhiAngle[boolHm2] = 0
            GammaAngle = augmentedLabels[:, 2, :, :].detach().clone()
            GammaAngle[boolHm1] = 0
            GammaAngle[boolHm2] = 0

            augmentedLabels[:, 3, :, :] = 1 / (torch.amax(augmentedLabels[:, 3, :, :])) * augmentedLabels[:, 3, :, :]
            augmentedLabels[:, 0, :, :] = ThetaAng
            augmentedLabels[:, 1, :, :] = PhiAngle
            augmentedLabels[:, 2, :, :] = GammaAngle
    elif ChooseRandomPoints == True:
        #choose few random points in likelihood map, hopefully will represent an update hm
        if NumRandomPoints is None:
            NumRandomPoints = 5
        blur = torchvision.transforms.GaussianBlur(3, sigma=1.0)
        filterX3 = augmentedLabels[:, 3, :, :] < 0.5  # only areas with low probability selected
        filterX3 = filterX3[:, None, :, :]

        newOnes = torch.zeros(filterX3.shape)
        for imgNum in range(filterX3.shape[0]):
            curFilter = filterX3[imgNum,:,:,:]
            curFilter = curFilter[None,:,:,:]
            onesmade = torch.where(curFilter == False, torch.ones(curFilter.shape), torch.zeros(curFilter.shape))

            if torch.count_nonzero(onesmade) < (2* NumRandomPoints):
                newOnes[imgNum,:,:,:] = onesmade
                continue

            saveProb =NumRandomPoints/ torch.count_nonzero(onesmade)

            idx = torch.where(onesmade ==1)
            lengthTorch = idx[0].size()
            randIdxtorch = torch.randperm(lengthTorch[0])
            randIdxtorch = randIdxtorch[:int(lengthTorch[0] * saveProb)]
            random_indices = (idx[0][randIdxtorch], idx[1][randIdxtorch],idx[2][randIdxtorch],idx[3][randIdxtorch])
            z = torch.zeros(onesmade.shape)
            z[random_indices] = 1
            z= blur(z)
            z = (1/torch.amax(z))*z
            newOnes[imgNum, :, :, :] =z

        #now use new HM to filter out crap
        filterX3 = newOnes == 0

        x0_temp = augmentedLabels[:, 0, :, :].detach().clone()
        x0_temp = x0_temp[:, None, :, :]
        x0_temp[filterX3] = 0

        x1_temp = augmentedLabels[:, 1, :, :].detach().clone()
        x1_temp = x1_temp[:, None, :, :]
        x1_temp[filterX3] = 0

        x2_temp = augmentedLabels[:, 2, :, :].detach().clone()
        x2_temp = x2_temp[:, None, :, :]
        x2_temp[filterX3] = 0

        x = torch.cat((x0_temp, x1_temp, x2_temp), dim=1)

        augmentedLabels = torch.cat((x, newOnes), dim=1)

    else:
        filterX3 = augmentedLabels[:,3,:,:] < 0.5  # only areas with low probability selected
        filterX3 =filterX3[:, None, :, :]
        x0_temp = augmentedLabels[:, 0, :, :].detach().clone()
        x0_temp = x0_temp[:, None, :, :]
        x0_temp[filterX3] = 0

        x1_temp = augmentedLabels[:, 1, :, :].detach().clone()
        x1_temp = x1_temp[:, None, :, :]
        x1_temp[filterX3] = 0

        x2_temp = augmentedLabels[:, 2, :, :].detach().clone()
        x2_temp = x2_temp[:, None, :, :]
        x2_temp[filterX3] = 0

        x = torch.cat((x0_temp, x1_temp, x2_temp), dim=1)
        x3 = augmentedLabels[:,3,:,:][:,None,:,:].detach().clone()
        x3[filterX3] =0

        augmentedLabels = torch.cat((x, x3), dim=1)

    return augmentedLabels


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device,
               NoiseToTarget=False, LikelihoodUpdate=False, UseSameNoise=False,ChooseRandomPoints=False):
    data, labels = batch
    labels = AugmentTasks(labels, NoiseToRegressionTargetAngles=NoiseToTarget, LikelihoodsUpdates=LikelihoodUpdate,
                          UseSameNoise=UseSameNoise,ChooseRandomPoints=ChooseRandomPoints)
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    # adaptation_indices[np.arange(shots * ways)] = True
    adaptation_indices[np.arange(int((2 / 3) * data.size(0)))] = True

    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
    # Adapt the model
    for step in range(adaptation_steps):
        pred = learner(adaptation_data)
        train_error = loss(pred, adaptation_labels)
        learner.adapt(train_error)
        #print('adapt step:' , step)
        #print('train error:' ,train_error)

        #print('valid error:',valid_error )

    # Evaluate the adapted model
    pred = learner(adaptation_data)
    train_error = loss(pred, adaptation_labels)
    predictions = learner(evaluation_data)
    #valid_error = loss(predictions, evaluation_labels)
    #print(valid_error)

    return train_error  # , valid_accuracy


# create_images(predictions, 'predictions', ManualSubplotValue=(10,4))
def create_images(TensorImg, nameOfFile, ManualSubplotValue=None):
    if ManualSubplotValue == None:
        subplotValue = math.ceil(math.sqrt(TensorImg.shape[0] * TensorImg.shape[1]))
        fig1, axs = plt.subplots(subplotValue, subplotValue, figsize=(30, 30))
    else:
        subplotValue = math.ceil(math.sqrt(TensorImg.shape[0] * TensorImg.shape[1]))
        fig1, axs = plt.subplots(ManualSubplotValue[0], ManualSubplotValue[1], figsize=(30, 30))
    startx = 0
    starty = 0

    for x in range(TensorImg.shape[0]):
        for y in range(TensorImg.shape[1]):

            if starty % (ManualSubplotValue[1] - 1) == 0 and starty != 0:
                img = np.asarray(TensorImg[x, y, :, :].detach().cpu())
                axs[startx, starty].imshow(img)
                im = axs[startx, starty].imshow(img)
                im.set_clim(0, 1)
            elif starty % (ManualSubplotValue[1] - 2) == 0 and starty != 0:
                img = np.asarray(TensorImg[x, y, :, :].detach().cpu())
                axs[startx, starty].imshow(img)
                im = axs[startx, starty].imshow(img)
                im.set_clim(0, np.pi)
            else:
                img = np.asarray(torch.cos(TensorImg[x, y, :, :]).detach().cpu())
                axs[startx, starty].imshow(img)
                im = axs[startx, starty].imshow(img)
                im.set_clim(-1, 1)
            plt.colorbar(im, ax=axs[startx, starty])

            if starty % (ManualSubplotValue[1] - 1) == 0 and starty != 0:
                starty = 0
                startx = startx + 1
            else:
                starty = starty + 1
    plt.tight_layout()
    plt.savefig(r'/mnt/d/Thesis/ThesisCode_Models/Model1/CreatedImagesFromScript/'+str(nameOfFile))


def create_feature_images(TensorImg, nameOfFile):
    subplotValue = math.ceil(math.sqrt(TensorImg.shape[1]))
    fig1, axs = plt.subplots(subplotValue, subplotValue, figsize=(30, 30))
    startx = 0
    starty = 0

    for x in range(TensorImg.shape[1]):
        img = np.asarray(TensorImg[0, x, :, :].detach().cpu())
        axs[startx, starty].imshow(img)
        if starty % (subplotValue - 1) == 0 and starty != 0:
            starty = 0
            startx = startx + 1
        else:
            starty = starty + 1
    plt.tight_layout()
    plt.savefig(str(nameOfFile))


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")


class customLoss_7Channel(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(customLoss_7Channel, self).__init__()

    def forward(self, inputTensor, TargetTensor, ):
        TensorAnglesThetaPhi = inputTensor[:, 0:2, :, :]
        TensorAnglesGamma = inputTensor[:, 2, :, :]
        TensorHMs = inputTensor[:, 3, :, :]
        TargetAnglesThetaPhi = TargetTensor[:, 0:2, :, :]
        TargetAnglesGamma = TargetTensor[:, 2, :, :]
        """
        a = TensorAngles > np.pi / 2
        b = TargetAngles < np.pi / 2
        mapping = torch.logical_and(a, b)
        TensorAngles[mapping] = TensorAngles[mapping] + np.pi
        """
        critMSE = nn.MSELoss(reduction='mean')
        lossMSE = critMSE(TensorHMs, TargetTensor[:, 3, :, :])

        lossAngularThetaPhi = ((2 - (
                2 * torch.cos(TensorAnglesThetaPhi - TargetTensor[:, 0:2, :, :]))).sum()) / (
                                  torch.numel(TensorAnglesThetaPhi))  # input should be in radians

        lossAngularGamma = ((2 - 2 * torch.square(
            torch.cos(TensorAnglesGamma - TargetTensor[:, 2, :, :]))).sum()) / (
                               torch.numel(TensorAnglesGamma))  # input should be in radians
        # critnew = nn.L1Loss()
        # lossMSEAngular =  critnew(TensorAngles, TargetTensor[:, 0:3, :, :])
        loss = lossMSE + (1 / 3) * lossAngularGamma + (2 / 3) * lossAngularThetaPhi  # post Furkan Loss
        # loss = lossMSE + lossAngularGamma + 2*lossAngularThetaPhi  #post Furkan Loss

        return loss


def main(
        config=None, trainloader=None, valloader=None, testloader=None, testtestloader=None
):
    shots = 20
    ways = 1
    meta_batch_size = int(config['meta_batch_size'])  # since total of 15 objects per category/ task object
    batch_size = int(config['batch_size'])  # 6 per object, so 5 objects
    adapt_lr = config['adapt_lr']
    meta_lr = config['meta_lr']
    adaptation_steps = int(config['adaptation_steps'])
    test_steps = 10
    seed = 42
    Create_checkpoint = True
    load_checkpoint = False
    testing = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        device = torch.device("cuda:0")

    # create the model
    model = RBranchEarlyExit(config)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=True, allow_unused=True, allow_nograd=True)
    opt = optim.SGD(maml.parameters(), meta_lr)
    # loss = nn.MSELoss(reduction='mean')
    loss = customLoss_7Channel()
    trainbest_loss = float("inf")
    valbest_loss = float("inf")
    CHECKPOINT_PATH = r'/mnt/d/Thesis/ThesisCode_Models/Model1/MODEL2/PostFurkanLoss/Post_Furkan_chat_DualLoss_Best_DeepNet_checkpoint.pth.tar'
    if load_checkpoint == True:
        print('loading checkpoint from ' + CHECKPOINT_PATH)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict((checkpoint["optimizer"]))
        maml.load_state_dict(checkpoint['maml_state_dict'])
        checkpoint = None  # Free up GPU memory

    val_iter = iter(valloader)
    train_iter = iter(trainloader)
    test_iter = iter(testloader)
    testtest_iter = iter(testtestloader)

    for epoch in range(int(config['iterations'])):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_valid_error = 0.0
        meta_test_error = 0.0
        meta_testtest_error = 0.0
        test_size = 1
        gradSteps = 5

        if testing == True:
            try:
                batch = next(testtest_iter)
            except StopIteration:
                testtest_iter = iter(testtestloader)
                batch = next(testtest_iter)
            for _ in range(gradSteps):
                opt.zero_grad()
                meta_test_error = 0.0
                meta_testtest_error =0.0
                for task in range(1):  # 14 since first category will have 84 total objects 6*14 =84
                    # Compute meta-testing loss
                    learner = maml.clone()

                    evaluation_error = fast_adapt(batch,
                                                  learner,
                                                  loss,
                                                  test_steps,
                                                  shots,
                                                  ways,
                                                  device,
                                                  NoiseToTarget=False, LikelihoodUpdate=False
                                                  )
                    meta_test_error += evaluation_error.item()
                    # meta_test_accuracy += evaluation_accuracy.item()
                print('Meta Test Error', meta_test_error / test_size)
                evaluation_error.backward()
                for p in maml.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(1.0 / test_size)
                opt.step()

                learner = maml.clone()


                evaluation_error = fast_adapt(batch,
                                              learner,
                                              loss,
                                              test_steps,
                                              shots,
                                              ways,
                                              device,
                                              LikelihoodUpdate=False, UseSameNoise=False)
                meta_testtest_error += evaluation_error.item()
                print(meta_testtest_error)
                # print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)
            exit()

        for task in range(meta_batch_size):  # get train and validation Data
            # Compute meta-training loss
            learner = maml.clone()
            # learner.train()
            # model.train()
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(trainloader)
                batch = next(train_iter)

            evaluation_error = fast_adapt(batch,
                                          learner,
                                          loss,
                                          adaptation_steps,
                                          shots,
                                          ways,
                                          device, NoiseToTarget=True, LikelihoodUpdate=False,ChooseRandomPoints=True
                                          )

            evaluation_error.backward()
            meta_train_error += evaluation_error.item()

            # Compute meta-validation loss
            learner = maml.clone()
            # learner.eval()
            # model.eval()

            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(valloader)
                batch = next(val_iter)

            evaluation_error = fast_adapt(batch,
                                          learner,
                                          loss,
                                          adaptation_steps,
                                          shots,
                                          ways,
                                          device, NoiseToTarget=True, LikelihoodUpdate=False,ChooseRandomPoints=True
                                          )

            meta_valid_error += evaluation_error.item()

        # Print some metrics
        print('\n')
        print('epoch', epoch)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

        train_is_best = (meta_train_error / meta_batch_size) < trainbest_loss
        val_is_best = (meta_valid_error / meta_batch_size) < valbest_loss
        valbest_loss = min(meta_valid_error / meta_batch_size, valbest_loss)
        trainbest_loss = min(meta_train_error / meta_batch_size, trainbest_loss)

        # check if both trainingloss and testloss are decreasing
        is_best = (val_is_best == True) and (train_is_best == True)
        #if epoch % 50 == 0:
        #    wandb.log({"Val_loss": (meta_valid_error / meta_batch_size),
        #               "Train_loss": (meta_train_error / meta_batch_size)})

            # Optional
        wandb.watch(model)
        if Create_checkpoint == True:
            save_checkpoint(
                {
                    "iterations": epoch,
                    "maml_state_dict": maml.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "valloss": valbest_loss,
                    'trainloss': trainbest_loss,
                    "optimizer": opt.state_dict(),
                },
                is_best, filename=" AugmentedData_RandomPoints_checkpoint.pth.tar"
            )


CONFIG = {
    'batch_size': 32,
    'meta_batch_size': 32,
    'iterations': 1000,
    'adaptation_steps': 60,
    'num_filters': 5,
    'num_filters_FC': 5,
    'momentum': 0.9,
    'meta_lr': 0.04,
    'adapt_lr': 0.1,
    'leakySlope': 0.08,
    'gaussianSigma': 0.8,
    'dropout': 0.8

}
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--HSweep",
                        help="To sweep or not",
                        default='NoSweep',
                        choices=['Sweep', 'NoSweep'])

    args = parser.parse_args()
    print('loading data')

    DataPathTrain = r'/mnt/d/Thesis/ThesisCode_Models/Model1/DataWithOrientationHM_normalized/Data/Train'
    DataPathVal = r'/mnt/d/Thesis/ThesisCode_Models/Model1/DataWithOrientationHM_normalized/Data/Val'
    DataPathTestadapt = r'/mnt/d/Thesis/ThesisCode_Models/Model1/DataWithOrientationHM_normalized/Data/TestAdapt'
    DataPathTesttest = r'/mnt/d/Thesis/ThesisCode_Models/Model1/DataWithOrientationHM_normalized/Data/TestTest'
    traintasksets = FolderData(DataPathTrain, ModelSelect='Part2', Singleimg=False, AddNoise=False)
    trainloader = torch.utils.data.DataLoader(traintasksets, batch_size=CONFIG['batch_size'])

    valtasksets = FolderData(DataPathVal, ModelSelect='Part2', Singleimg=False)
    valloader = torch.utils.data.DataLoader(valtasksets, batch_size=CONFIG['batch_size'])

    testtasksets = FolderData(DataPathTestadapt, ModelSelect='Part2', train_testFlag='test', Singleimg=False)
    testloader = torch.utils.data.DataLoader(testtasksets, batch_size=72)

    testtesttasksets = FolderData(DataPathTesttest, ModelSelect='Part2', train_testFlag='test', Singleimg=False)
    testtestloader = torch.utils.data.DataLoader(testtasksets, batch_size=6)


    def train_function():

        with wandb.init(project='AugmentedData_MetaAdpt', config=CONFIG, entity='kryptixone'):
            main(wandb.config, trainloader, valloader, testloader, testtestloader)


    if args.HSweep == 'Sweep':

        SWEEP_CONFIG = {'method': 'bayes'}  # random, bayes, grid
        metric = {'name': 'Val_loss',
                  # 'train_loss':'Train_loss',
                  'goal': 'minimize'}
        SWEEP_CONFIG['metric'] = metric

        parameters_dict = {
            'batch_size': {'value': 32},
            'meta_batch_size': {'value': 32},
            'meta_lr': {'distribution': 'uniform',
                        'min': 0.001,
                        'max': 0.05},
            'adapt_lr': {'distribution': 'uniform',
                         'min': 0.1,
                         'max': 0.9},
            'momentum': {'value': 0.9},
            'adaptation_steps': {'value': 60},
            'leakySlope': {'value': 0.08},
            'gaussianSigma': {'value': 1.6},
            'dropout': {'value': 0.8}

        }
        SWEEP_CONFIG['parameters'] = parameters_dict

        parameters_dict.update({'iterations': {'value': 1}})
        sweep_id = wandb.sweep(SWEEP_CONFIG, project="AugmentedDataSweep_MLR")
        wandb.agent(sweep_id, function=train_function, count=50)  # main(args.network)
    else:
        train_function()
