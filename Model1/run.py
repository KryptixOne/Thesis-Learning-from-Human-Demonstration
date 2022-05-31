# pylint: disable=E1101,R,C
import numpy as np
import torch.nn as nn
import torchvision.transforms

from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_integrate
from s2cnn import so3_near_identity_grid
from s2cnn import s2_near_identity_grid
import torch.nn.functional as F
import torch
import torch.utils.data as data_utils
import gzip
import pickle
import numpy as np
from torch.autograd import Variable
import argparse
from sklearn.model_selection import train_test_split
from math import isnan
from random import shuffle
import shutil

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

NUM_EPOCHS = 2000
BATCH_SIZE = 8
LEARNING_RATE = 5e-3
LOAD_CHECKPOINT = True
CHECKPOINT_PATH = '/mnt/d/Thesis/ThesisCode_Models/Model1/CheckpointEpoch16/checkpoint_best_loss.pth.tar'


def load_data(path, batch_size, keys_path=None, bad_keys_path=None):
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

        train_data = torch.from_numpy(
            X_train[:, None, :, :].astype(np.float32))  # creates extra dimension [60000, 1, 60,60]

        X_train = None  # free up memory

        train_labels = torch.from_numpy(
            Y_train[:, None, :, :].astype(np.float32))

        Y_train = None  # free up memory

        train_dataset = data_utils.TensorDataset(train_data, train_labels)
        train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_data = torch.from_numpy(
            X_val[:, None, :, :].astype(np.float32))
        X_val = None  # free up memory

        val_labels = torch.from_numpy(
            Y_val[:, None, :, :].astype(np.float32))
        Y_val = None  # free up memory

        val_dataset = data_utils.TensorDataset(val_data, val_labels)
        val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader, train_dataset, val_dataset


class SphericalModelDeep(nn.Module):

    def __init__(self, bandwidth=30):
        super(SphericalModelDeep, self).__init__()

        grid_s2 = s2_near_identity_grid(n_alpha=6, max_beta=np.pi / 16, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi / 16, n_beta=1, max_gamma=2 * np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi / 8, n_beta=1, max_gamma=2 * np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi / 4, n_beta=1, max_gamma=2 * np.pi, n_gamma=6)
        grid_so3_4 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi / 2, n_beta=1, max_gamma=2 * np.pi, n_gamma=6)

        self.conv1 = S2Convolution(
            nfeature_in=1,
            nfeature_out=16,
            b_in=bandwidth,
            b_out=bandwidth,
            grid=grid_s2)

        self.m = nn.MaxPool3d((2))
        self.BatchNorm3d1 = nn.BatchNorm3d(16)
        self.conv2 = SO3Convolution(
            nfeature_in=16,
            nfeature_out=32,
            b_in=bandwidth // 2,
            b_out=bandwidth // 2,
            grid=grid_so3_2)

        self.conv3 = SO3Convolution(
            nfeature_in=32,
            nfeature_out=64,
            b_in=bandwidth // 2,
            b_out=bandwidth // 4,
            grid=grid_so3_3)
        self.BatchNorm3d2 = nn.BatchNorm3d(64)
        self.C3d1 = (nn.Conv3d(64, 64, 2)).to(DEVICE)

        self.conv4 = SO3Convolution(
            nfeature_in=64,
            nfeature_out=128,
            b_in=bandwidth // 8,
            b_out=bandwidth // 8,
            grid=grid_so3_4)

        self.conv5 = SO3Convolution(
            nfeature_in=128,
            nfeature_out=256,
            b_in=bandwidth // 8,
            b_out=bandwidth // 8,
            grid=grid_so3_4)
        self.BatchNorm3d3 = nn.BatchNorm3d(256)

        self.C3d2 = nn.Conv3d(256, 64, 1)
        self.C3d3 = nn.Conv3d(64, 32, 1)

        self.conv3inv = SO3Convolution(
            nfeature_in=32,
            nfeature_out=1,
            b_in=bandwidth // 8,
            b_out=bandwidth,
            grid=grid_so3_3)
        self.BatchNorm3d4 = nn.BatchNorm3d(1)

        self.dimRed = nn.Conv3d(1, 1, (1, 1, bandwidth * 2))

        """
        self.conv2inv = SO3Convolution(
            nfeature_in=16,
            nfeature_out=8,
            b_in=bandwidth // 2,
            b_out=bandwidth,
            grid=grid_so3_1)

        self.conv1inv = SO3Convolution(
            nfeature_in=8,
            nfeature_out=1,
            b_in=bandwidth,
            b_out=bandwidth,
            grid=grid_so3_1)
        """

        # tx = t.view(t.size(0), t.size(1), t.size(2),t.size(3),-1).max(-1)[0]

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)  # [batch, 8, b, b, b]

        x = self.m(x)  # [batch, 8, b/2, b/2, b/2]

        x = self.BatchNorm3d1(x)
        x = self.conv2(x)  # [30,30,30]
        x = F.relu(x)

        x = self.conv3(x)  # [14,14,14]
        x = F.relu(x)
        x = self.m(x)

        x = self.BatchNorm3d2(x)
        x = self.C3d1(x)
        x = self.conv4(x)

        # fully connected model
        x = self.conv5(x)
        x = F.relu(x)
        x = self.BatchNorm3d3(x)

        x = self.C3d2(x)
        x = F.relu(x)

        x = self.C3d3(x)
        x = F.relu(x)

        x = self.conv3inv(x)  # bring back to size [ batch 1 b,b,b]
        x = self.BatchNorm3d4(x)
        x = self.dimRed(x)
        x = F.relu(x)

        x = torch.squeeze(x, dim=4)
        x = torch.squeeze(x, dim=1)

        # print()
        # x = so3_integrate(x)
        # x = self.linear(x)
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


def main(network):
    train_loader, test_loader, train_dataset, _ = load_data(
        DATASET_TO_SPLIT_NORMALIZED, BATCH_SIZE, KEYS_USED, BAD_KEYS)

    if network == 'SphericalModel':
        model = SphericalModelDeep()
    elif network == 'PaperBased':
        model = ModelBasedOnPaperNoneSpherical()
    elif network == 'ShallowPaper':
        model = ShallowModelBasedOnPaperNoneSpherical()
    elif network == 'GithubBasedPaper':
        model = ModelBasedOnPaperGitHubVersion()

    else:
        raise ValueError('Unknown network architecture')
    model.to(DEVICE)

    print("#params", sum(x.numel() for x in model.parameters()))

    criterion = nn.MSELoss()
    criterion = criterion.to(DEVICE)

    # optimizer = torch.optim.Adam(
    #    model.parameters(),
    #    lr=LEARNING_RATE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, nesterov=True, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    last_epoch =0
    if LOAD_CHECKPOINT == True:
        print('loading checkpoint from '+ CHECKPOINT_PATH)
        checkpoint = torch.load(CHECKPOINT_PATH,map_location = DEVICE)
        last_epoch = checkpoint["epoch"]+1
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict((checkpoint["optimizer"]))
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        checkpoint = None #Free up GPU memory


    best_loss = float("inf")
    for epoch in range(last_epoch,NUM_EPOCHS):

        for i, (images, labels) in enumerate(train_loader):
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
                epoch + 1, NUM_EPOCHS, i + 1, len(train_dataset) // BATCH_SIZE,
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
            is_best, filename=str(network) + str(epoch) + "checkpoint.pth.tar"
        )

        print('Validation MSE Loss: ', np.asarray(totalloss.detach().cpu()))
        print('learning rate: ', optimizer.param_groups[0]['lr'])

        # print('Test Accuracy: {0}'.format(100 * correct / total))

    print('complete')
    input('press a key to end')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network",
                        help="network architecture to use",
                        default='GithubBasedPaper',
                        choices=['SphericalModel', 'PaperBased', 'ShallowPaper', 'GithubBasedPaper'])
    args = parser.parse_args()

    main(args.network)
