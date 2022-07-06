# pylint: disable=E1101,R,C
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
import torch.utils.data as data_utils
import pickle
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from math import isnan, sqrt, ceil
from random import shuffle
import shutil
import wandb
import time
from matplotlib import pyplot as plt
import cv2
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import math

# temp path for Data
DATASET_PATH_win = r"D:/Thesis/ThesisCode_Models/DataToWorkWith/Data_Spherical_With_PosPmaps60.pickle"
DATASET_PATH_lin = '/mnt/d/Thesis/Thesis Code/DataBackup/Data_Spherical_With_PosPmaps60KDE.pickle'
DATASET_SMALL_LIN_for_split = '/mnt/d/Thesis/ThesisCode_Models/Model1/Datasmall/dataDepthAndPosHM2000.pickle'
DATASET_LARGE_LIN_for_split = '/mnt/d/Thesis/ThesisCode_Models/Model1/Datasmall/dataDepthAndPosHM.pickle'
DATASET_TO_SPLIT_NORMALIZED = '/mnt/d/Thesis/ThesisCode_Models/Model1/DataToPreSplit/dataDepthAndPosHM_normalized6000.pickle'
KEYS_USED = '/mnt/d/Thesis/ThesisCode_Models/Model1/DataToPreSplit/keysUsed_normalized6000.npy'
BAD_KEYS = '/mnt/d/Thesis/ThesisCode_Models/Model1/DataToPreSplit/badKeys_normalized6000.npy'
MNIST_PATH = "s2_mnist.gz"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(path, batch_size, keys_path=None, bad_keys_path=None, network=None):
    pre_split_available = True
    data_not_formatted = False
    start_training = True
    use_residuals = False

    # run After Data_not formatted
    if pre_split_available == False:  # if pre-split data is not available,
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        with open(keys_path, 'rb') as f:
            goodKeys = np.load(keys_path)
        with open(bad_keys_path, 'rb') as f:
            badKeys = np.load(bad_keys_path)
        idxlist = []
        for x in range(len(badKeys)):
            idx = np.where(goodKeys == badKeys[x])[0][0]
            idx = idx * 6
            idxlist.append(idx)

        # This delete List will be useful later for determining name of model when we do reconstruction, If we do reconstruction
        delete_list = []
        for x in idxlist:
            delete_list.append(x)
            delete_list.append(x + 1)
            delete_list.append(x + 2)
            delete_list.append(x + 3)
            delete_list.append(x + 4)
            delete_list.append(x + 5)
        print()
        # removes indices with bad Data
        dataset['depthImages'] = np.delete(dataset['depthImages'], delete_list, axis=0)
        dataset['positionHeatMap'] = np.delete(dataset['positionHeatMap'], delete_list, axis=0)

        indices = np.arange(len(dataset['depthImages']))
        X_train, X_testVal, Y_train, Y_testVal, Idx_train, Idx_testVal = train_test_split(dataset['depthImages'],
                                                                                          dataset['positionHeatMap'],
                                                                                          indices,
                                                                                          test_size=0.3,
                                                                                          random_state=42)
        dataset = None

        X_test, X_val, Y_test, Y_val, Idx_test, Idx_val = train_test_split(X_testVal, Y_testVal, Idx_testVal,
                                                                           test_size=0.5,
                                                                           random_state=42)

        np.save('X_train_6000_PosDepth', X_train)
        np.save('X_val_6000_PosDepth', X_val)
        np.save('X_test_6000_PosDepth', X_test)
        np.save('Y_train_6000_PosDepth', Y_train)
        np.save('Y_val_6000_PosDepth', Y_val)
        np.save('Y_test_6000_PosDepth', Y_test)
        np.save('Idx_test_6000_PosDepth', Idx_test)
        np.save('Idx_val_6000_PosDepth', Idx_val)
        np.save('Idx_train_6000_PosDepth', Idx_train)
    else:
        if use_residuals == False:
            if RUN_TEST_DATA == True:
                Y_test = np.load(
                    '/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/Y_test_6000_PosDepth.npy')
                X_test = np.load(
                    '/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/X_test_6000_PosDepth.npy')
            else:
                X_train = np.load(
                    '/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/X_train_6000_PosDepth.npy')
                X_val = np.load(
                    '/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/X_val_6000_PosDepth.npy')

                Y_train = np.load(
                    '/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/Y_train_6000_PosDepth.npy')
                Y_val = np.load(
                    '/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/Y_val_6000_PosDepth.npy')

            # Idx_test= np.load('Idx_test_6000_PosDepth.npy')
            # Idx_val = np.load('Idx_val_6000_PosDepth.npy')
            # Idx_train = np.load('Idx_train_6000_PosDepth.npy')
        else:
            from os import listdir
            from os.path import isfile, join

            pathTrainDataX = '/mnt/d/Thesis/ThesisCode_Models/Model1/ResidualData/TrainData_X'
            pathTrainResidualsY = '/mnt/d/Thesis/ThesisCode_Models/Model1/ResidualData/TrainResiduals_Y'
            pathValDataX = '/mnt/d/Thesis/ThesisCode_Models/Model1/ResidualData/ValData_X'
            pathValResidualsY = '/mnt/d/Thesis/ThesisCode_Models/Model1/ResidualData/ValResiduals_Y'

            onlyfiles = [f for f in listdir(pathTrainDataX) if isfile(join(pathTrainDataX, f))]
            count = 0
            for x in onlyfiles:
                if count == 0:
                    X_train = np.load(join(pathTrainDataX, x))
                    count = 1
                else:
                    temp = np.load(join(pathTrainDataX, x))
                    X_train = np.concatenate((X_train, temp), axis=0)

            onlyfiles = [f for f in listdir(pathTrainResidualsY) if isfile(join(pathTrainResidualsY, f))]
            count = 0
            for x in onlyfiles:
                if count == 0:
                    Y_train = np.load(join(pathTrainResidualsY, x))
                    count = 1
                else:
                    temp = np.load(join(pathTrainResidualsY, x))
                    Y_train = np.concatenate((Y_train, temp), axis=0)

            onlyfiles = [f for f in listdir(pathValDataX) if isfile(join(pathValDataX, f))]
            count = 0
            for x in onlyfiles:
                if count == 0:
                    X_val = np.load(join(pathValDataX, x))
                    count = 1
                else:
                    temp = np.load(join(pathValDataX, x))
                    X_val = np.concatenate((X_val, temp), axis=0)

            onlyfiles = [f for f in listdir(pathValResidualsY) if isfile(join(pathValResidualsY, f))]
            count = 0
            for x in onlyfiles:
                if count == 0:
                    Y_val = np.load(join(pathValResidualsY, x))
                    count = 1
                else:
                    temp = np.load(join(pathValResidualsY, x))
                    Y_val = np.concatenate((Y_val, temp), axis=0)

            # go to each path, get each item combine and
            print('hi')

            pass
    # run before pre_split_available
    if data_not_formatted == True:
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        # For Data creation when data isn't formatted
        k = 0
        # shuffles keylist due to occasional pycharm "killed'. We want random assortment of objects
        keylist = list(dataset.keys())
        shuffle(keylist)

        keys_used = []
        bad_keys = []

        for item in keylist:
            for x in range(6):
                currobj = dataset[item][x]
                currdepth = currobj[None, 0, :, :]
                currPosHeatMap = currobj[None, 5, :, :]
                # normalizing and nan handeling
                if (not isnan(np.amax(currPosHeatMap))) and (np.amax(currPosHeatMap) != 0):
                    currPosHeatMap = currPosHeatMap * 1 / (np.amax(currPosHeatMap))  # normalize to 0 -1

                mask = np.isnan(currPosHeatMap)
                mask2 = np.isnan(currdepth)

                if np.any(np.isnan(currdepth)) == True:
                    try:
                        currdepth[mask2] = np.interp(np.flatnonzero(mask2), np.flatnonzero(~mask2),
                                                     currdepth[~mask2])
                    except:
                        print(item)
                        bad_keys.append(item)

                        pass

                if np.any(np.isnan(currPosHeatMap)) == True:
                    try:
                        currPosHeatMap[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                                                         currPosHeatMap[~mask])

                    except:
                        if isnan(np.amax(currPosHeatMap)):
                            currPosHeatMap = np.zeros_like(currPosHeatMap)

                if item == keylist[0] and x == 0:
                    depthimages = currdepth
                    posHeatMap = currPosHeatMap

                else:
                    depthimages = np.concatenate((depthimages, currdepth), axis=0)
                    posHeatMap = np.concatenate((posHeatMap, currPosHeatMap), axis=0)

            keys_used.append(item)
            if k % 1000 == 0:
                data_dict = {'depthImages': depthimages, 'positionHeatMap': posHeatMap}
                with open('dataDepthAndPosHM_normalized' + str(k) + '.pickle', 'wb') as handle:
                    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                npKeys = np.asarray(keys_used)

                bad_keys = list(dict.fromkeys(bad_keys))
                npBadKeys = np.asarray(bad_keys)

                # With order of keysused, we can then find which indexes to remove after data creation
                with open('keysUsed_normalized' + str(k) + '.npy', 'wb') as f:
                    np.save(f, npKeys)
                with open('badKeys_normalized' + str(k) + '.npy', 'wb') as f:
                    np.save(f, npBadKeys)
                print(k)

            del dataset[item]
            k = k + 1

            # data_dict = {'depthImages':depthimages, 'positionHeatMap': posHeatMap}

    if start_training == True:
        if RUN_TEST_DATA == False:
            assert not np.any(np.isnan(X_train))
            assert not np.any(np.isnan(Y_train))
            assert not np.any(np.isnan(X_val))
            assert not np.any(np.isnan(Y_val))

            # Y_train = Y_train+X_train
            if network == 'SingleBranchHpLp':
                X_train_Hp = np.zeros_like(X_train)
                X_train_Lp = np.zeros_like(X_train)
                for x in range(X_train.shape[0]):
                    X_train_Hp[x, :, :] = cv2.Laplacian(X_train[x, :, :], cv2.CV_64F)
                    X_train_Lp[x, :, :] = cv2.GaussianBlur(X_train[x, :, :], (5, 5), 0)
                X_train_Hp = np.expand_dims(X_train_Hp, axis=1)
                X_train_Lp = np.expand_dims(X_train_Lp, axis=1)
                X_train = np.concatenate((X_train_Hp, X_train_Lp), axis=1)
                X_train_Hp = None
                X_train_Lp = None

                train_data = torch.from_numpy(
                    X_train[:, None, :, :, :].astype(np.float32))  # creates extra dimension [60000, 1, 60,60]
            else:
                train_data = torch.from_numpy(
                    X_train[:, None, :, :].astype(np.float32))

            X_train = None  # free up memory

            train_labels = torch.from_numpy(
                Y_train[:, None, :, :].astype(np.float32))

            Y_train = None  # free up memory
            if network == 'ResNetBased':
                train_data = (1 / 2) * train_data
                train_data = torch.cat((train_data, train_data, train_data), dim=1)

            train_dataset = data_utils.TensorDataset(train_data, train_labels)
            train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            if network == 'SingleBranchHpLp':
                X_val_Hp = np.zeros_like(X_val)
                X_val_Lp = np.zeros_like(X_val)
                for x in range(X_val.shape[0]):
                    X_val_Hp[x, :, :] = cv2.Laplacian(X_val[x, :, :], cv2.CV_64F)
                    X_val_Lp[x, :, :] = cv2.GaussianBlur(X_val[x, :, :], (5, 5), 0)
                X_val_Hp = np.expand_dims(X_val_Hp, axis=1)
                X_val_Lp = np.expand_dims(X_val_Lp, axis=1)
                X_val = np.concatenate((X_val_Hp, X_val_Lp), axis=1)
                X_val_Hp = None
                X_val_Lp = None

                val_data = torch.from_numpy(
                    X_val[:, None, :, :, :].astype(np.float32))
            else:
                val_data = torch.from_numpy(
                    X_val[:, None, :, :].astype(np.float32))
            X_val = None  # free up memory1

            val_labels = torch.from_numpy(
                Y_val[:, None, :, :].astype(np.float32))
            Y_val = None  # free up memory

            if network == 'ResNetBased':
                val_data = (1 / 2) * val_data
                val_data = torch.cat((val_data, val_data, val_data), dim=1)

            val_dataset = data_utils.TensorDataset(val_data, val_labels)
            val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

            return train_loader, val_loader, train_dataset, val_dataset
        else:
            test_data = torch.from_numpy(
                X_test[:, None, :, :].astype(np.float32))
            X_test = None  # free up memory1

            test_labels = torch.from_numpy(
                Y_test[:, None, :, :].astype(np.float32))
            Y_test = None  # free up memory

            test_dataset = data_utils.TensorDataset(test_data, test_labels)
            test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            return test_loader, test_dataset


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


class ShallowModelBasedOnPaperNoneSpherical(nn.Module):

    def __init__(self, bandwidth=30):
        super(ShallowModelBasedOnPaperNoneSpherical, self).__init__()

        self.pool = nn.MaxPool2d(2)

        self.conv1 = nn.Conv2d(1, 64, 5, padding=2)
        # self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.convFullyConnected1 = nn.Conv2d(128, 512, 2)
        # self.convFullyConnected2 = nn.Conv2d(512, 256, 1)
        self.convFullyConnected3 = nn.Conv2d(512, 128, 1)

        self.upsample1 = nn.ConvTranspose2d(128, 1, 5, stride=4, padding=0, output_padding=3)
        # self.upsample2 = nn.ConvTranspose2d(32, 1, 2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(1, 1, 1, stride=1)

    def forward(self, x):
        residual1 = x
        x = self.conv1(x)
        x = self.pool(x)
        x = F.relu(x)

        # x = self.conv2(x) #30 30
        # x = self.pool(x)
        # x = F.relu(x)

        x = self.conv3(x)  # 15 15
        x = self.pool(x)
        x = F.relu(x)

        x = self.convFullyConnected1(x)
        x = F.relu(x)

        # x = self.convFullyConnected2(x)
        # x = F.relu(x)

        x = self.convFullyConnected3(x)
        x = F.relu(x)

        x = self.upsample1(x)

        x = F.relu(x)

        # 15 15
        # x = self.upsample2(x)
        # 30 30
        # x = F.relu(x)

        # x= self.upsample3(x)
        x = x + residual1
        x = self.upsample3(x)
        x = F.relu(x)
        x = x.squeeze(dim=1)
        return x


class SingleBranchWithDenseFCLayers(nn.Module):

    def __init__(self, config):
        super(SingleBranchWithDenseFCLayers, self).__init__()
        # Computational Graph Version
        # original filters, 5 5 5 9 9 9
        # changed filters, 3 3 3 5 5 5
        # 1 4  4 16 16 64  64 128

        self.conva1 = nn.Conv2d(1, 8, config['num_filters'], padding=2, padding_mode='replicate')
        self.poola = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conva2 = nn.Conv2d(8, 32, config['num_filters'])
        self.poola1 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conva3 = nn.Conv2d(32, 128, config['num_filters'])
        self.conva4 = nn.Conv2d(128, 256, config['num_filters_FC'])

        self.FC1 = nn.Conv2d(256, 256, 1)
        self.FC2 = nn.Conv2d(256, 128, 1)
        self.FC3 = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        img_xformer = torchvision.transforms.Resize((x.shape[2] * 3, x.shape[3] * 3))
        x = img_xformer(x)
        # img_xformer2 = torchvision.transforms.Resize((x.shape[2] // 4, x.shape[3] // 4))
        x = self.conva1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.poola(x)

        x = self.conva2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.poola1(x)

        x = self.conva3(x)
        x = F.relu(x)

        x = self.conva4(x)
        x = F.relu(x)

        # x = self.upSam(x)
        x = self.FC1(x)
        x = F.relu(x)
        x = self.FC2(x)
        x = F.relu(x)
        # x = self.bn5(x)

        x = self.FC3(x)
        x = F.relu(x)

        finalXformer = torchvision.transforms.Resize((60, 60))
        x = finalXformer(x)

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
        # self.conva5 = nn.Conv2d(256, 512, config['num_filters_FC'], padding=2, padding_mode=config['padding_mode'])
        # Last Layers

        self.upSam = nn.Upsample(scale_factor=2, mode=config['upsample'])
        self.convd1 = nn.Conv2d(256, 256, config['num_filters_FC'])

        self.convd2 = nn.Conv2d(256, 32, config['num_filters_FC'])

        self.dropout1 = nn.Dropout(p=config['dropout'])
        self.fc1 = nn.Linear(82944 // 2, 3600)
        # self.fc2 = nn.Linear(7200, 3600)
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

        # x = self.conva5(x)
        # x = F.relu(x)

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


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")


def create_images(TensorImg, nameOfFile):
    subplotValue = ceil(sqrt(TensorImg.shape[0]))
    fig1, axs = plt.subplots(subplotValue, subplotValue, figsize=(30, 30))
    startx = 0
    starty = 0

    for x in range(TensorImg.shape[0]):
        img = np.asarray(TensorImg[x, :, :, :].detach().cpu()).squeeze(0)
        axs[startx, starty].imshow(img)
        if starty % (subplotValue - 1) == 0 and starty != 0:
            starty = 0
            startx = startx + 1
        else:
            starty = starty + 1
    plt.tight_layout()
    plt.savefig(str(nameOfFile))


def create_feature_images(TensorImg, nameOfFile):
    subplotValue = ceil(sqrt(TensorImg.shape[1]))
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


def main(network, config=None):
    if RUN_TEST_DATA == True:
        test_loader, test_dataset = load_data(
            DATASET_TO_SPLIT_NORMALIZED, config['batch_size'], KEYS_USED, BAD_KEYS, network)
    else:
        train_loader, val_loader, train_dataset, _ = load_data(
            DATASET_TO_SPLIT_NORMALIZED, config['batch_size'], KEYS_USED, BAD_KEYS, network)

    if network == 'SphericalModel':
        model = SphericalModelDeep()
    elif network == 'PaperBased':
        model = ModelBasedOnPaperNoneSpherical()
    elif network == 'ShallowPaper':
        model = ShallowModelBasedOnPaperNoneSpherical()
    elif network == 'GithubBasedPaper':
        model = ModelBasedOnPaperGitHubVersion(config)
    elif network == 'testModel':
        model = testModel(config)
    elif network == 'SingleBranchWithDenseFCLayers':
        model = SingleBranchWithDenseFCLayers(config)
    elif network == 'ResNetBased':

        model = ResNetModelBased()
        model.model.conv1.requires_grad_(False)
        model.model.bn1.requires_grad_(False)
        model.model.maxpool.requires_grad_(False)
        model.model.layer1.requires_grad_(False)
        model.model.layer2.requires_grad_(False)
        model.model.layer3.requires_grad_(False)
    elif network == 'GithubBasedPaperSingleBranch':
        model = ModelBasedOnPaperGitHubVersionSINGLEBRANCH(config)
    elif network == 'SingleBranchHpLp':
        model = SingleBranchHpLp(config)
    elif network == 'FirstWorkingModelForPos':
        model = FirstWorkingModelForPos_NoDense(config)
    else:
        raise ValueError('Unknown network architecture')
    model.to(DEVICE)

    print("#params", sum(x.numel() for x in model.parameters()))

    criterion = nn.MSELoss()
    criterion = criterion.to(DEVICE)

    # optimizer = torch.optim.Adam(
    #    model.parameters(),
    #    lr=LEARNING_RATE)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], nesterov=True,
                                momentum=config['momentum'])
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr = 0.04,step_size_up=6300)
    last_epoch = 0
    if LOAD_CHECKPOINT == True:
        print('loading checkpoint from ' + CHECKPOINT_PATH)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        last_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict((checkpoint["optimizer"]))
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        checkpoint = None  # Free up GPU memory

    # time.sleep(2)

    if RUN_TEST_DATA == False:

        best_loss = float("inf")
        train_best_loss = float("inf")
        for epoch in range(last_epoch, config['epochs']):
            count = 0
            traintotalloss = 0
            curepoch = 0

            for i, (images, labels) in enumerate(train_loader):
                model.train()

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)

                loss = criterion(outputs, labels)  # * abs((60*60*32)-int(torch.count_nonzero(outputs.detach().cpu())))
                # loss = 1- ms_ssim_module(upSam(outputs),upSam(labels))
                traintotalloss = traintotalloss + loss
                loss.backward()

                optimizer.step()
                # forCyclic
                lr_scheduler.step()
                curepoch = curepoch + 1

                # ForCyclicLR

                print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.10f}'.format(
                    epoch + 1, config['epochs'], i + 1, len(train_dataset) // config['batch_size'],
                    loss.item()), end="")
            print("")
            traintotalloss = traintotalloss / curepoch
            train_is_best = traintotalloss < train_best_loss
            train_best_loss = min(traintotalloss, train_best_loss)

            totalloss = 0
            curepoch = 0
            count = 0
            for images, labels in val_loader:
                model.eval()

                with torch.no_grad():
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)

                    outputs = model(images)
                    valloss = criterion(outputs,
                                        labels)  # * abs((60*60*32)-int(torch.count_nonzero(outputs.detach().cpu())))

                    # valloss = 1- ms_ssim_module(upSam(outputs),upSam(labels))

                    curepoch = curepoch + 1
                    totalloss = totalloss + valloss

            totalloss = totalloss / curepoch
            # for reduce on plateu
            #lr_scheduler.step(totalloss)

            wandb.log({"Val_loss": totalloss,
                       "Train_loss": traintotalloss})

            # Optional
            wandb.watch(model)

            test_is_best = totalloss < best_loss
            best_loss = min(totalloss, best_loss)

            # check if both trainingloss and testloss are decreasing
            is_best = (test_is_best == True) and (train_is_best == True)

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "loss": totalloss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best, filename=str(network) + "checkpoint.pth.tar"
            )

            print('Validation MSE Loss: ', np.asarray(totalloss.detach().cpu()))
            # print('learning rate: ', optimizer.param_groups[0]['lr'])
    else:

        totalloss = 0
        curepoch = 0
        count = 0
        for images, labels in test_loader:
            model.eval()

            with torch.no_grad():
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                testloss = criterion(outputs,
                                     labels)  # * abs((60*60*32)-int(torch.count_nonzero(outputs.detach().cpu())))

                # valloss = 1- ms_ssim_module(upSam(outputs),upSam(labels))

                curepoch = curepoch + 1
                totalloss = totalloss + testloss

        totalloss = totalloss / curepoch
        print('Test MSE Loss: ', np.asarray(totalloss.detach().cpu()))


LOAD_CHECKPOINT = False
CHECKPOINT_PATH = r'/mnt/d/Thesis/ThesisCode_Models/Model1//checkpoint_best_loss.pth.tar'
RUN_TEST_DATA = False

CONFIG = {
    'batch_size': 16,
    'epochs': 16,
    'num_filters': 5,
    'num_filters_FC': 5,
    'momentum': 0.9,
    'learning_rate': 5e-3,
    'padding_mode': 'replicate',
    'upsample': 'bilinear',
    'leakySlope': 0.0189,
    'dropout':0.25

}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network",
                        help="network architecture to use",
                        default='testModel',
                        choices=['SphericalModel', 'PaperBased', 'ShallowPaper', 'GithubBasedPaper', 'ResNetBased',
                                 'GithubBasedPaperSingleBranch', 'SingleBranchHpLp', 'testModel',
                                 'SingleBranchWithDenseFCLayers', 'FirstWorkingModelForPos'])
    parser.add_argument("--HSweep",
                        help="To sweep or not",
                        default='NoSweep',
                        choices=['Sweep', 'NoSweep'])

    args = parser.parse_args()


    def train_function():
        with wandb.init(project='FinalTest_pos_model', config=CONFIG, entity='kryptixone'):
            main(args.network, wandb.config)


    if args.HSweep == 'Sweep':

        SWEEP_CONFIG = {'method': 'bayes'} #random, bayes, grid
        metric = {'name': 'loss',
                  'goal': 'minimize'}
        SWEEP_CONFIG['metric'] = metric

        parameters_dict = {
            'num_filters': {'value': 5},
            'num_filters_FC': {'value': 5},
            # 'learning_rate': {'distribution': 'uniform',
            #                  'min': 0,
            #                  'max': 0.1},
            'learning_rate': {'value': 5e-3},
            'momentum': {'value': 0.9},
            'padding_mode': {'value': 'replicate'},
            'upsample': {'value': 'bilinear'},
            'leakySlope': {'value':0.0189},
            'dropout':{'distribution':'uniform',
                       'min':0,
                       'max':0.5}

        }
        SWEEP_CONFIG['parameters'] = parameters_dict

        parameters_dict.update({'epochs': {'value': 16}})
        parameters_dict.update({'batch_size': {'value': 16}})
        sweep_id = wandb.sweep(SWEEP_CONFIG, project="FinalTest_pos_model")
        wandb.agent(sweep_id, function=train_function, count=25)  # main(args.network)
    else:
        train_function()
