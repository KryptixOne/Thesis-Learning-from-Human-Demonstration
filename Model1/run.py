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

# temp path for Data
DATASET_PATH_win = r"D:/Thesis/ThesisCode_Models/DataToWorkWith/Data_Spherical_With_PosPmaps60.pickle"
DATASET_PATH_lin = '/mnt/d/Thesis/Thesis Code/DataBackup/Data_Spherical_With_PosPmaps60.pickle'
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
    load_test_data = False
    start_training = True

    if pre_split_available == False:  # if pre-split data is not available
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
        X_train = np.load('/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/X_train_6000_PosDepth.npy')
        X_val = np.load('/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/X_val_6000_PosDepth.npy')

        Y_train = np.load('/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/Y_train_6000_PosDepth.npy')
        Y_val = np.load('/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/Y_val_6000_PosDepth.npy')

        if load_test_data == True:
            Y_test = np.load('/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/Y_test_6000_PosDepth.npy')
            X_test = np.load('/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/X_test_6000_PosDepth.npy')

        # Idx_test= np.load('Idx_test_6000_PosDepth.npy')
        # Idx_val = np.load('Idx_val_6000_PosDepth.npy')
        # Idx_train = np.load('Idx_train_6000_PosDepth.npy')

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

            del dataset[item]
            k = k + 1

            # data_dict = {'depthImages':depthimages, 'positionHeatMap': posHeatMap}

    if start_training == True:
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


class SphericalModelDeep(nn.Module):

    def __init__(self, bandwidth=30):
        super(SphericalModelDeep, self).__init__()

        grid_s2 = s2_near_identity_grid() #roughly a 5x5 filter
        grid_so3_4 = so3_near_identity_grid()#roughly 6x6 filter

        self.conv1 = S2Convolution(
            nfeature_in=1,
            nfeature_out=64,
            b_in=bandwidth,
            b_out=bandwidth//2,
            grid=grid_s2)
        self.maxpool1 = nn.MaxPool3d((1,1,bandwidth))
        self.conv2 = S2Convolution(
            nfeature_in=64,
            nfeature_out=128,
            b_in=bandwidth//2 ,
            b_out=bandwidth //2,
            grid=grid_s2)
        self.maxpool2 = nn.MaxPool3d((1, 1, bandwidth))

        self.conv3 = S2Convolution(
            nfeature_in=128,
            nfeature_out=256,
            b_in=bandwidth //2,
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
            b_out=bandwidth ,
            grid=grid_so3_4)
        self.maxpool3 = nn.MaxPool3d((1, 1, bandwidth * 2))



    # S2->S2->S2->SO3->S03
    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        #x = so3_integrate_only_gamma(x)
        x = self.maxpool1(x)
        x = torch.squeeze(x,dim = -1)


        x = self.conv2(x)
        x = F.relu(x)
        #x = so3_integrate_only_gamma(x)
        x= self.maxpool2(x)
        x = torch.squeeze(x,dim = -1)

        x = self.conv3(x)
        x = F.relu(x)




        # fully connected model
        x = self.conv4(x)
        x = F.relu(x)


        x = self.conv5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.maxpool3(x)
        #x = so3_integrate_only_gamma(x)
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

        self.conva1 = nn.Conv2d(1, 64, config['num_filters'])
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

    def __init__(self, bandwidth=30):
        super(ModelBasedOnPaperGitHubVersion, self).__init__()
        # Computational Graph Version

        # Full Res
        self.conva1 = nn.Conv2d(1, 64, 5, 1)
        self.poola = nn.MaxPool2d(2, 2)
        self.conva2 = nn.Conv2d(64, 128, 5)
        self.poola1 = nn.MaxPool2d(2, 2)
        self.conva3 = nn.Conv2d(128, 256, 5)
        self.conva4 = nn.Conv2d(256, 512, 9)

        # Half Res
        self.convb1 = nn.Conv2d(1, 64, 5, 1)
        self.poolb = nn.MaxPool2d(2, 2)
        self.convb2 = nn.Conv2d(64, 128, 5)
        self.poolb1 = nn.MaxPool2d(2, 2)
        self.convb3 = nn.Conv2d(128, 256, 5)
        self.convb4 = nn.Conv2d(256, 512, 9)

        # Quarter Res
        """

        self.convc1 = nn.Conv2d(1, 64, 5, 1)
        self.poolc = nn.MaxPool2d(2, 2)
        self.convc2 = nn.Conv2d(64, 128, 5)
        self.poolc1 = nn.MaxPool2d(2, 2)
        self.convc3 = nn.Conv2d(128, 256, 5)
        self.convc4 = nn.Conv2d(256, 512, 9)
        """

        # Last Layers

        self.convd1 = nn.Conv2d(512, 512, 9)
        self.convd2 = nn.Conv2d(512, 1, 9)

    def forward(self, x):
        img_xformer = torchvision.transforms.Resize((x.shape[2] * 3, x.shape[3] * 3))
        x = img_xformer(x)
        img_xformer1 = torchvision.transforms.Resize((x.shape[2] // 2, x.shape[3] // 2))
        # img_xformer2 = torchvision.transforms.Resize((x.shape[2] // 4, x.shape[3] // 4))

        x1 = x

        x1 = self.conva1(x1)
        x1 = F.relu(x1)
        x1 = self.poola(x1)
        x1 = self.conva2(x1)
        x1 = F.relu(x1)
        x1 = self.poola1(x1)
        x1 = self.conva3(x1)
        x1 = F.relu(x1)
        x1 = self.conva4(x1)
        x1 = F.relu(x1)

        img_xformer1transpose = torchvision.transforms.Resize((x1.shape[2], x1.shape[3]))
        # img_xformer2transpose = torchvision.transforms.Resize((x1.shape[2], x1.shape[3]))

        x2 = img_xformer1(x)
        x2 = self.convb1(x2)
        x2 = F.relu(x2)
        x2 = self.poolb(x2)
        x2 = self.convb2(x2)
        x2 = F.relu(x2)
        x2 = self.poolb1(x2)
        x2 = self.convb3(x2)
        x2 = F.relu(x2)
        x2 = self.convb4(x2)
        x2 = F.relu(x2)
        x2 = img_xformer1transpose(x2)
        """
        x3 = img_xformer2(x)
        x3 = self.convc1(x3)
        x3 = self.poolc(x3)
        x3 = self.convc2(x3)
        x3 = self.poolc1(x3)
        x3 = self.convc3(x3)
        x3 = self.convc4(x3)
        x3= img_xformer2transpose(x3)
        """

        x = x1 + x2  # + x3
        x /= 2

        x = self.convd1(x)
        x = F.relu(x)
        x = self.convd2(x)
        finalXformer = torchvision.transforms.Resize((60, 60))
        x = finalXformer(x)

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
    train_loader, test_loader, train_dataset, _ = load_data(
        DATASET_TO_SPLIT_NORMALIZED, config['batch_size'], KEYS_USED, BAD_KEYS, network)

    if network == 'SphericalModel':
        model = SphericalModelDeep()
    elif network == 'PaperBased':
        model = ModelBasedOnPaperNoneSpherical()
    elif network == 'ShallowPaper':
        model = ShallowModelBasedOnPaperNoneSpherical()
    elif network == 'GithubBasedPaper':
        model = ModelBasedOnPaperGitHubVersion()
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



    else:
        raise ValueError('Unknown network architecture')
    model.to(DEVICE)

    print("#params", sum(x.numel() for x in model.parameters()))

    criterion = nn.MSELoss()
    # criterion =nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)

    # optimizer = torch.optim.Adam(
    #    model.parameters(),
    #    lr=LEARNING_RATE)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], nesterov=True,
                                momentum=config['momentum'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
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

    best_loss = float("inf")
    for epoch in range(last_epoch, config['epochs']):

        for i, (images, labels) in enumerate(train_loader):
            # break
            model.train()

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)  # * abs((60*60*32)-int(torch.count_nonzero(outputs.detach().cpu())))
            # loss = (outputs-labels)**2
            loss.backward()

            optimizer.step()

            print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.10f}'.format(
                epoch + 1, config['epochs'], i + 1, len(train_dataset) // config['batch_size'],
                loss.item()), end="")
        print("")

        totalloss = 0
        curepoch = 0

        for images, labels in test_loader:
            model.eval()

            with torch.no_grad():
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                valloss = criterion(outputs,
                                    labels)  # * abs((60*60*32)-int(torch.count_nonzero(outputs.detach().cpu())))

                curepoch = curepoch + 1
                totalloss = totalloss + valloss
        totalloss = totalloss / curepoch
        lr_scheduler.step(totalloss)

        wandb.log({"loss": totalloss})

        # Optional
        wandb.watch(model)

        is_best = totalloss < best_loss
        best_loss = min(totalloss, best_loss)

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
        print('learning rate: ', optimizer.param_groups[0]['lr'])

        # print('Test Accuracy: {0}'.format(100 * correct / total))


LOAD_CHECKPOINT = True
CHECKPOINT_PATH = r'/mnt/d/Thesis/ThesisCode_Models/Model1/SphericalModel_maxpool/checkpoint_best_loss.pth.tar'

CONFIG = {
    'batch_size': 4,
    'epochs': 10000,
    'num_filters': 5,
    'num_filters_FC': 5,
    'momentum': 0.9,
    'learning_rate': 5e-3

}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network",
                        help="network architecture to use",
                        default='SphericalModel',
                        choices=['SphericalModel', 'PaperBased', 'ShallowPaper', 'GithubBasedPaper', 'ResNetBased',
                                 'GithubBasedPaperSingleBranch', 'SingleBranchHpLp'])
    parser.add_argument("--HSweep",
                        help="To sweep or not",
                        default='NoSweep',
                        choices=['Sweep', 'NoSweep'])

    args = parser.parse_args()


    def train_function():
        with wandb.init(project='SingleBranchSpherical', config=CONFIG, entity='kryptixone'):
            main(args.network, wandb.config)


    if args.HSweep == 'Sweep':

        SWEEP_CONFIG = {'method': 'random'}
        metric = {'name': 'loss',
                  'goal': 'minimize'}
        SWEEP_CONFIG['metric'] = metric

        parameters_dict = {
            'num_filters': {'values': [1, 2, 3, 4, 5]},
            'num_filters_FC': {'values': [1, 2, 3, 4, 5, 6, 7, 8, 9]},
            # 'learning_rate': {'distribution': 'uniform',
            #                  'min': 0,
            #                  'max': 0.05},
            'learning_rate': {'value': 5e-3},
            'momentum': {'value': 0.9}
        }
        SWEEP_CONFIG['parameters'] = parameters_dict

        parameters_dict.update({'epochs': {'value': 5}})
        parameters_dict.update({'batch_size': {'value': 4}})
        sweep_id = wandb.sweep(SWEEP_CONFIG, project="HyperParametersSingleBranch")
        wandb.agent(sweep_id, function=train_function, count=100)  # main(args.network)
    else:
        train_function()
