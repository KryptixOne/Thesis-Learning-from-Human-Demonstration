import numpy as np
import torch
import random
import numpy as np
import torch
import learn2learn as l2l
import math
from torch import nn, optim
import torchvision
from torchmeta.toy import Sinusoid
from torchmeta.utils.data import BatchMetaDataLoader
import torch.nn.functional as F





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

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def ImageSampler_SingleTask(inputImgTensor, block_size =10, num_subtasks=5,non_zero_block_factor =3):
    """
    For a single input task, this will sample the heatmap accordingly and return the sampled heatmaps
    return: returns the sampled heatmaps
    """

    szTensor= inputImgTensor.size() #get input size
    inputImgTensor = np.asarray(inputImgTensor.detach().cpu()) #convert to numpy array
    numElementsXform = (szTensor[1]//block_size)**2
    nonZeroBlocks = numElementsXform //non_zero_block_factor
    zArray = np.zeros(numElementsXform)
    zArray[0:nonZeroBlocks]=1

    #create Md-Matrix of xform Matrices Used to sample img blocks.
    for x in range(num_subtasks):
        if x ==0:
            XformMat = np.expand_dims(np.random.permutation(zArray),axis =0)
        else:
            temp = np.random.permutation(zArray)
            XformMat= np.concatenate((XformMat,np.expand_dims(temp,axis=0)),axis=0)

    blocksofInput= blockshaped(inputImgTensor[1,:,:],block_size,block_size)
    #origblocks =  np.copy(blocksofInput)
    sampledInput =  np.expand_dims(np.copy(blocksofInput),axis =0)
    for x in range(XformMat.shape[0]):
        arraytoUse = XformMat[x,:]
        for y in range (arraytoUse.shape[0]):
            blocksofInput[y,:,:] = blocksofInput[y,:,:]*arraytoUse[y]
        if x ==0:
            sampledInput[x,:,:,:] = blocksofInput
        else:
            sampledInput = np.concatenate((sampledInput,np.expand_dims(blocksofInput,axis=0)))

    sampledHeatmaps = np.zeros((XformMat.shape[0],inputImgTensor.shape[1],inputImgTensor.shape[1]))

    """
    Restructuring of the data such that it is similar to input data. Currently done manually due to annoyance.
    """
    for x in range(sampledInput.shape[0]):
        toXformThis = sampledInput[x,:,:,:]
        a=np.hstack(toXformThis[0:6,:,:])
        b=np.hstack(toXformThis[6:12, :, :])
        c=np.hstack(toXformThis[12:18 :, :])
        d=np.hstack(toXformThis[18:24, :, :])
        e= np.hstack(toXformThis[24:30, :, :])
        f = np.hstack(toXformThis[30:36, :, :])

        imgnew = np.vstack((a, b, c, d, e, f))
        sampledHeatmaps[x,:,:] =imgnew


    return sampledHeatmaps




#inputimg = torch.rand(2,60,60)
#ImageSampler_SingleTask(inputImgTensor =inputimg)
if __name__ == '__main__':

    CONFIG = {
        'batch_size': 16,
        'epochs': 1000,
        'num_filters': 5,
        'num_filters_FC': 5,
        'momentum': 0.9,
        'learning_rate': 5e-3,
        'padding_mode': 'replicate',
        'upsample': 'bilinear',
        'leakySlope': 0.0189,
        'dropout': 0.25

    }