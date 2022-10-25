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
from torch.utils.data import Dataset
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
from os import listdir
from os.path import isfile, join
from pathlib import Path
from collections import Counter

from Dataloader.FolderDataPart1 import FolderData
from Models.Models import SphericalModelDeep, ResNetModelBased, ModelBasedOnPaperGitHubVersionSINGLEBRANCH, \
    SingleBranchHpLp, ModelBasedOnPaperGitHubVersion, ModelBasedOnPaperNoneSpherical, FirstWorkingModelForPos_NoDense, \
    BestPosModel_WithDenseLayer

# temp path for Data
DATASET_PATH_win = r"D:/Thesis/ThesisCode_Models/DataToWorkWith/Data_Spherical_With_PosPmaps60.pickle"
DATASET_PATH_lin = '/mnt/d/Thesis/Thesis Code/DataBackup/Data_Spherical_With_PosPmaps60KDE.pickle'
DATASET_PATH_lin_FolderData = r'/mnt/d/Thesis/ThesisCode_Models/DataWithOrientationHM_normalized/Data/'
DATASET_SMALL_LIN_for_split = '/mnt/d/Thesis/ThesisCode_Models/ModelPart1/Datasmall/dataDepthAndPosHM2000.pickle'
DATASET_LARGE_LIN_for_split = '/mnt/d/Thesis/ThesisCode_Models/ModelPart1/Datasmall/dataDepthAndPosHM.pickle'
DATASET_TO_SPLIT_NORMALIZED = '/mnt/d/Thesis/ThesisCode_Models/ModelPart1/DataToPreSplit/dataDepthAndPosHM_normalized6000.pickle'

KEYS_USED = '/mnt/d/Thesis/ThesisCode_Models/ModelPart1/DataToPreSplit/keysUsed_normalized6000.npy'
BAD_KEYS = '/mnt/d/Thesis/ThesisCode_Models/ModelPart1/DataToPreSplit/badKeys_normalized6000.npy'
MNIST_PATH = "s2_mnist.gz"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def load_data(path, batch_size, keys_path=None, bad_keys_path=None, network=None):
    pre_split_available = False
    data_not_formatted = False
    start_training = False
    use_residuals = False
    use_single_file_as_dataset = False
    use_FolderData = True
    # path_of_individual_items_not_normalized = r'/mnt/d/Thesis/Thesis Code/Data_Created/DataWithOrientationHM/grasps'
    path_of_individual_items_not_normalized = r'/mnt/d/Thesis/HumanDemonstrated/HumanDemonstrated_withKDE/Tooluse'
    path_of_individual_items_normalized = r'/mnt/d/Thesis/HumanDemonstrated/HumanDemonstrated_Normalized/Tooluse'

    # run After Data_not formatted
    # dataset_To_prepsplit
    if (pre_split_available == False) and (use_FolderData == False):  # if pre-split data is not available,
        if use_single_file_as_dataset == True:
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
                                                                                              dataset[
                                                                                                  'positionHeatMap'],
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
            items = [f for f in listdir(path_of_individual_items_normalized) if
                     isfile(join(path_of_individual_items_normalized, f))]
            shuffle(items)
            for item in items:
                with open(join(path_of_individual_items_normalized, item), 'rb') as f:
                    curdata = pickle.load(f)
                if item == items[0]:
                    depthimages = curdata['depthImages']
                    posHeatMap = curdata['positionHeatMap']
                    ThetaAng = curdata['ThetaAng']
                    ThetaHm = curdata['ThetaHm']
                    PhiAngle = curdata['PhiAngle']
                    PhiHm = curdata['PhiHm']
                    GammaAngle = curdata['GammaAngle']
                    GammaHm = curdata['GammaHm']

                    data_dict = {'depthImages': depthimages,
                                 'positionHeatMap': posHeatMap,
                                 'ThetaAng': ThetaAng,
                                 'ThetaHm': ThetaHm,
                                 'PhiAngle': PhiAngle,
                                 'PhiHm': PhiHm,
                                 'GammaAngle': GammaAngle,
                                 'GammaHm': GammaHm
                                 }
                else:
                    depthimages0 = curdata['depthImages']
                    posHeatMap0 = curdata['positionHeatMap']
                    ThetaAng0 = curdata['ThetaAng']
                    ThetaHm0 = curdata['ThetaHm']
                    PhiAngle0 = curdata['PhiAngle']
                    PhiHm0 = curdata['PhiHm']
                    GammaAngle0 = curdata['GammaAngle']
                    GammaHm0 = curdata['GammaHm']

                    data_dict = {'depthImages': np.concatenate((data_dict['depthImages'], depthimages0), axis=0),
                                 'positionHeatMap': np.concatenate((data_dict['positionHeatMap'], posHeatMap0), axis=0),
                                 'ThetaAng': np.concatenate((data_dict['ThetaAng'], ThetaAng0), axis=0),
                                 'ThetaHm': np.concatenate((data_dict['ThetaHm'], ThetaHm0), axis=0),
                                 'PhiAngle': np.concatenate((data_dict['PhiAngle'], PhiAngle0), axis=0),
                                 'PhiHm': np.concatenate((data_dict['PhiHm'], PhiHm0), axis=0),
                                 'GammaAngle': np.concatenate((data_dict['GammaAngle'], GammaAngle0), axis=0),
                                 'GammaHm': np.concatenate((data_dict['GammaHm'], GammaHm0), axis=0)
                                 }
                # if len(data_dict['depthImages']) % 18000 == 0:
                #    break
            indices = np.arange(len(data_dict['depthImages']))
            X_train_depth, \
            X_testVal_depth, \
            Y_train_position, \
            Y_testVal_position, \
            Idx_train, \
            Idx_testVal, \
            Y_train_ThetaAng, \
            Y_testVal_ThetaAng, \
            Y_train_ThetaHm, \
            Y_testVal_ThetaHm, \
            Y_train_PhiAngle, \
            Y_testVal_PhiAngle, \
            Y_train_PhiHm, \
            Y_testVal_PhiHm, \
            Y_train_GammaAngle, \
            Y_testVal_GammaAngle, \
            Y_train_GammaHm, \
            Y_testVal_GammaHm = train_test_split(
                data_dict['depthImages'],
                data_dict[
                    'positionHeatMap'],
                indices,
                data_dict[
                    'ThetaAng'],
                data_dict[
                    'ThetaHm'],
                data_dict[
                    'PhiAngle'],
                data_dict[
                    'PhiHm'],
                data_dict[
                    'GammaAngle'],
                data_dict[
                    'GammaHm'],
                test_size=0.3,
                random_state=42)
            data_dict = None

            X_test_depth, X_Val_depth, \
            Y_test_position, Y_Val_position, \
            Idx_test, Idx_Val, \
            Y_test_ThetaAng, Y_Val_ThetaAng, \
            Y_test_ThetaHm, Y_Val_ThetaHm, \
            Y_test_PhiAngle, Y_Val_PhiAngle, \
            Y_test_PhiHm, Y_Val_PhiHm, \
            Y_test_GammaAngle, Y_Val_GammaAngle, \
            Y_test_GammaHm, Y_Val_GammaHm = train_test_split(X_testVal_depth,
                                                             Y_testVal_position,
                                                             Idx_testVal,
                                                             Y_testVal_ThetaAng,
                                                             Y_testVal_ThetaHm,
                                                             Y_testVal_PhiAngle,
                                                             Y_testVal_PhiHm,
                                                             Y_testVal_GammaAngle,
                                                             Y_testVal_GammaHm,

                                                             test_size=0.9,
                                                             random_state=42)

            np.save('X_train_depth', X_train_depth)
            np.save('X_test_depth', X_test_depth)
            np.save('X_Val_depth', X_Val_depth)

            np.save('Y_train_position', Y_train_position)
            np.save('Y_Val_position', Y_Val_position)
            np.save('Y_test_position', Y_test_position)

            np.save('Idx_train', Idx_train)
            np.save('Idx_test', Idx_test)
            np.save('Idx_Val', Idx_Val)

            np.save('Y_train_ThetaAng', Y_train_ThetaAng)
            np.save('Y_test_ThetaAng', Y_test_ThetaAng)
            np.save('Y_Val_ThetaAng', Y_Val_ThetaAng)

            np.save('Y_train_ThetaHm', Y_train_ThetaHm)
            np.save('Y_test_ThetaHm', Y_test_ThetaHm)
            np.save('Y_Val_ThetaHm', Y_Val_ThetaHm)

            np.save('Y_train_PhiAngle', Y_train_PhiAngle)
            np.save('Y_test_PhiAngle', Y_test_PhiAngle)
            np.save('Y_Val_PhiAngle', Y_Val_PhiAngle)

            np.save('Y_train_PhiHm', Y_train_PhiHm)
            np.save('Y_test_PhiHm', Y_test_PhiHm)
            np.save('Y_Val_PhiHm', Y_Val_PhiHm)

            np.save('Y_train_GammaAngle', Y_train_GammaAngle)
            np.save('Y_test_GammaAngle', Y_test_GammaAngle)
            np.save('Y_Val_GammaAngle', Y_Val_GammaAngle)

            np.save('Y_train_GammaHm', Y_train_GammaHm)
            np.save('Y_test_GammaHm', Y_test_GammaHm)
            np.save('Y_Val_GammaHm', Y_Val_GammaHm)

            items_np = np.asarray(items)
            np.save('items_used__Use_Indices_On_This', items_np)

    else:
        if (data_not_formatted == False) and (use_FolderData == False):
            if use_residuals == False:
                if RUN_TEST_DATA == True:
                    Y_test = np.load(
                        '/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/Y_test_6000_PosDepth.npy')
                    X_test = np.load(
                        '/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/X_test_6000_PosDepth.npy')
                    Idx_test = np.load(
                        '/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/Idx_test_6000_PosDepth.npy')
                else:
                    X_train = np.load(
                        '/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/X_train_6000_PosDepth.npy')
                    X_val = np.load(
                        '/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/X_val_6000_PosDepth.npy')

                    Y_train = np.load(
                        '/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/Y_train_6000_PosDepth.npy')
                    Y_val = np.load(
                        '/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/Y_val_6000_PosDepth.npy')

                    Idx_val = np.load(
                        '/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/Idx_val_6000_PosDepth.npy')
                    Idx_train = np.load(
                        '/mnt/d/Thesis/ThesisCode_Models/Model1/TrainData_6000_Pos_Depth/Idx_train_6000_PosDepth.npy')
            else:

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
        if use_single_file_as_dataset == True:
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
        else:
            # get category names:
            create_categories = False
            # get all files in directory.

            keylist = [f for f in listdir(path_of_individual_items_not_normalized) if
                       isfile(join(path_of_individual_items_not_normalized, f))]
            keys_used = []
            bad_keys = []
            k = 0

            onlyfiles = [f.split('_')[0] for f in listdir(path_of_individual_items_not_normalized) if
                         isfile(join(path_of_individual_items_not_normalized, f))]
            categories = list(dict.fromkeys(onlyfiles))
            a = dict(Counter(onlyfiles))
            categories_sorted = dict(sorted(a.items(), key=lambda item: item[1]))
            #train_categories = {key: val for key, val in categories_sorted.items() if val >= 15}
            #testVal_categories = {key: val for key, val in categories_sorted.items() if val < 15}
            #val_categories, test_categories = train_test_split(list(testVal_categories.keys()), test_size=0.3)

            if create_categories == True:
                for category in categories:
                    if category in list(train_categories.keys()):
                        Path(
                            r'/mnt/d/Thesis/ThesisCode_Models/Model1/DataWithOrientationHM_normalized/Data/Train/' + str(
                                category)).mkdir(parents=True, exist_ok=True)
                    elif category in list(test_categories):
                        Path(
                            r'/mnt/d/Thesis/ThesisCode_Models/Model1/DataWithOrientationHM_normalized/Data/Test/' + str(
                                category)).mkdir(parents=True, exist_ok=True)
                    else:
                        Path(
                            r'/mnt/d/Thesis/ThesisCode_Models/Model1/DataWithOrientationHM_normalized/Data/Val/' + str(
                                category)).mkdir(parents=True, exist_ok=True)
            for item in keylist:
                with open(join(path_of_individual_items_not_normalized, item), 'rb') as f:
                    dataset = pickle.load(f)
                for x in range(dataset.shape[0]):
                    currobj = dataset[x, :, :, :]
                    currdepth = currobj[None, 0, :, :]
                    currPosHeatMap = currobj[None, 5, :, :]
                    currThetaAngle = currobj[None, 6, :, :]
                    currThetaHm = currobj[None, 7, :, :]
                    currPhiAngle = currobj[None, 8, :, :]
                    currPhiHm = currobj[None, 9, :, :]
                    currGammaAngle = currobj[None, 10, :, :]
                    CurrGammaHm = currobj[None, 11, :, :]

                    # normalizing and nan handeling
                    if (not isnan(np.amax(currPosHeatMap))) and (np.amax(currPosHeatMap) != 0):
                        currPosHeatMap = currPosHeatMap * 1 / (np.amax(currPosHeatMap))  # normalize to 0 -1

                    mask = np.isnan(currPosHeatMap)
                    mask2 = np.isnan(currdepth)

                    mask3 = np.isnan(currThetaAngle)
                    mask4 = np.isnan(currThetaHm)
                    mask5 = np.isnan(currPhiAngle)
                    mask6 = np.isnan(currPhiHm)
                    mask7 = np.isnan(currGammaAngle)
                    mask8 = np.isnan(CurrGammaHm)

                    if np.any(np.isnan(currdepth)) == True:
                        try:
                            currdepth[mask2] = np.interp(np.flatnonzero(mask2), np.flatnonzero(~mask2),
                                                         currdepth[~mask2])
                        except:
                            print(str(item)[:-3])
                            bad_keys.append(item)

                            pass

                    if np.any(np.isnan(currPosHeatMap)) == True:
                        try:
                            currPosHeatMap[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                                                             currPosHeatMap[~mask])

                        except:
                            if isnan(np.amax(currPosHeatMap)):
                                currPosHeatMap = np.zeros_like(currPosHeatMap)

                    if np.any(np.isnan(currThetaAngle)) == True:
                        try:
                            currThetaAngle[mask3] = np.interp(np.flatnonzero(mask3), np.flatnonzero(~mask3),
                                                              currThetaAngle[~mask3])

                        except:
                            if isnan(np.amax(currThetaAngle)):
                                currThetaAngle = np.zeros_like(currThetaAngle)
                                print(str(item)[:-3])

                    if np.any(np.isnan(currThetaHm)) == True:
                        try:
                            currThetaHm[mask4] = np.interp(np.flatnonzero(mask4), np.flatnonzero(~mask4),
                                                           currThetaHm[~mask4])

                        except:
                            if isnan(np.amax(currThetaHm)):
                                currThetaHm = np.zeros_like(currThetaHm)
                                print(str(item)[:-3])

                    if np.any(np.isnan(currPhiAngle)) == True:
                        try:
                            currPhiAngle[mask5] = np.interp(np.flatnonzero(mask5), np.flatnonzero(~mask5),
                                                            currPhiAngle[~mask5])

                        except:
                            if isnan(np.amax(currPhiAngle)):
                                currPhiAngle = np.zeros_like(currPhiAngle)
                                print(str(item)[:-3])

                    if np.any(np.isnan(currPhiHm)) == True:
                        try:
                            currPhiHm[mask] = np.interp(np.flatnonzero(mask6), np.flatnonzero(~mask6),
                                                        currPhiHm[~mask6])

                        except:
                            if isnan(np.amax(currPhiHm)):
                                currPhiHm = np.zeros_like(currPhiHm)
                                print(str(item)[:-3])

                    if np.any(np.isnan(currGammaAngle)) == True:
                        try:
                            currGammaAngle[mask7] = np.interp(np.flatnonzero(mask7), np.flatnonzero(~mask7),
                                                              currGammaAngle[~mask7])

                        except:
                            if isnan(np.amax(currGammaAngle)):
                                currGammaAngle = np.zeros_like(currGammaAngle)
                                print(str(item)[:-3])

                    if np.any(np.isnan(CurrGammaHm)) == True:
                        try:
                            CurrGammaHm[mask] = np.interp(np.flatnonzero(mask8), np.flatnonzero(~mask8),
                                                          CurrGammaHm[~mask8])

                        except:
                            if isnan(np.amax(CurrGammaHm)):
                                CurrGammaHm = np.zeros_like(CurrGammaHm)
                                print(str(item)[:-3])

                    # if x == 0:
                    depthimages = currdepth
                    posHeatMap = currPosHeatMap
                    ThetaAng = currThetaAngle
                    ThetaHm = currThetaHm
                    PhiAngle = currPhiAngle
                    PhiHm = currPhiHm
                    GammaAngle = currGammaAngle
                    GammaHm = CurrGammaHm

                    """
                    else:
                        depthimages = np.concatenate((depthimages, currdepth), axis=0)
                        posHeatMap = np.concatenate((posHeatMap, currPosHeatMap), axis=0)

                        ThetaAng = np.concatenate((ThetaAng, currThetaAngle), axis=0)
                        ThetaHm = np.concatenate((ThetaHm, currThetaHm), axis=0)
                        PhiAngle = np.concatenate((PhiAngle, currPhiAngle), axis=0)
                        PhiHm = np.concatenate((PhiHm, currPhiHm), axis=0)
                        GammaAngle = np.concatenate((GammaAngle, currGammaAngle), axis=0)
                        GammaHm = np.concatenate((GammaHm, CurrGammaHm), axis=0)
                    """

                    keys_used.append(item)

                    data_dict = {'depthImages': depthimages,
                                 'positionHeatMap': posHeatMap,
                                 'ThetaAng': ThetaAng,
                                 'ThetaHm': ThetaHm,
                                 'PhiAngle': PhiAngle,
                                 'PhiHm': PhiHm,
                                 'GammaAngle': GammaAngle,
                                 'GammaHm': GammaHm
                                 }
                    if create_categories == True:
                        if str(item).split('_')[0] in list(train_categories.keys()):
                            with open(
                                    r'/mnt/d/Thesis/ThesisCode_Models/Model1/DataWithOrientationHM_normalized/Data/Train/' +
                                    str(item).split('_')[0] + r'/' + str(item)[
                                                                     :-7] + 'vers_' + str(x) + '.pickle',
                                    'wb') as handle:
                                pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        elif str(item).split('_')[0] in list(test_categories):
                            with open(
                                    r'/mnt/d/Thesis/ThesisCode_Models/Model1/DataWithOrientationHM_normalized/Data/Test/' +
                                    str(item).split('_')[0] + r'/' + str(item)[
                                                                     :-7] + 'vers_' + str(x) + '.pickle',
                                    'wb') as handle:
                                pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        else:
                            with open(
                                    r'/mnt/d/Thesis/ThesisCode_Models/Model1/DataWithOrientationHM_normalized/Data/Val/' +
                                    str(item).split('_')[0] + r'/' + str(item)[
                                                                     :-7] + 'vers_' + str(x) + '.pickle',
                                    'wb') as handle:
                                pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        with open(
                                r'/mnt/d/Thesis/HumanDemonstrated/HumanDemonstrated_Normalized/Tooluse/' +
                                str(item).split('_')[0] + r'/' + str(item)[
                                                                 :-7] + 'vers_' + str(x) + '.pickle',
                                'wb') as handle:
                            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    npKeys = np.asarray(keys_used)

                    bad_keys = list(dict.fromkeys(bad_keys))
                    npBadKeys = np.asarray(bad_keys)
                k = k + 1
                """
                # With order of keysused, we can then find which indexes to remove after data creation
                with open(
                        r'/mnt/d/Thesis/ThesisCode_Models/Model1/DataWithOrientationHM_normalized/KeysUsed/keysUsed_normalized' + str(
                            k) + '.npy', 'wb') as f:
                    np.save(f, npKeys)
                with open(
                        r'/mnt/d/Thesis/ThesisCode_Models/Model1/DataWithOrientationHM_normalized/BadKeys/badKeys_normalized' + str(
                            k) + '.npy', 'wb') as f:
                    np.save(f, npBadKeys)
                # print(k)
                """

    if start_training == True:
        if use_FolderData == False:
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

                idx_data = torch.from_numpy(Idx_train)

                if network == 'ResNetBased':
                    train_data = (1 / 2) * train_data
                    train_data = torch.cat((train_data, train_data, train_data), dim=1)

                train_dataset = data_utils.TensorDataset(train_data, train_labels, idx_data)
                train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                idx_Valdata = torch.from_numpy(Idx_val)

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

                val_dataset = data_utils.TensorDataset(val_data, val_labels, idx_Valdata)
                val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

                return train_loader, val_loader, train_dataset, val_dataset
            else:
                test_data = torch.from_numpy(
                    X_test[:, None, :, :].astype(np.float32))
                X_test = None  # free up memory1

                test_labels = torch.from_numpy(
                    Y_test[:, None, :, :].astype(np.float32))
                Y_test = None  # free up memory
                idx_data = torch.from_numpy(Idx_test)
                test_dataset = data_utils.TensorDataset(test_data, test_labels, idx_data)
                test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

                return test_loader, test_dataset
        else:
            trainPath = path + r'/Train'
            valPath = path + r'/Val'

            train_dataset = FolderData(trainPath, train_testFlag='train', ModelSelect='Part1')
            val_dataset = FolderData(valPath, train_testFlag='val', ModelSelect='Part1')
            train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

            return train_loader, val_loader, train_dataset, val_dataset


def main(network, config=None):
    if RUN_TEST_DATA == True:
        test_loader, test_dataset = load_data(
            DATASET_TO_SPLIT_NORMALIZED, config['batch_size'], KEYS_USED, BAD_KEYS, network)
    else:
        train_loader, val_loader, train_dataset, _ = load_data(
            DATASET_PATH_lin_FolderData, config['batch_size'], KEYS_USED, BAD_KEYS, network)

    if network == 'SphericalModel':
        model = SphericalModelDeep()
    elif network == 'PaperBased':
        model = ModelBasedOnPaperNoneSpherical()
    elif network == 'GithubBasedPaper':
        model = ModelBasedOnPaperGitHubVersion(config)
    elif network == 'testModel':
        model = testModel(config)
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
    elif network == 'BestPosModel_WithDenseLayer':
        model = BestPosModel_WithDenseLayer(config)
    else:
        raise ValueError('Unknown network architecture')
    model.to(DEVICE)

    print("#params", sum(x.numel() for x in model.parameters()))

    criterion = nn.MSELoss()
    criterion = criterion.to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], nesterov=True,
                                momentum=config['momentum'])
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.04, step_size_up=6344,
                                                     mode='triangular2')
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

            # for i, (images, labels, idx) in enumerate(train_loader):
            for i, (images, labels) in enumerate(train_loader):
                if CREATE_IMAGE_PAIRS == False:
                    model.train()

                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)

                    optimizer.zero_grad()
                    outputs = model(images)

                    loss = criterion(outputs,
                                     labels)  # * abs((60*60*32)-int(torch.count_nonzero(outputs.detach().cpu())))
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
                else:
                    model.eval()
                    with torch.no_grad():

                        images = images.to(DEVICE)
                        labels = labels.to(DEVICE)

                        optimizer.zero_grad()
                        outputs = model(images)

                        loss = criterion(outputs,
                                         labels)  # * abs((60*60*32)-int(torch.count_nonzero(outputs.detach().cpu())))
                        # loss = 1- ms_ssim_module(upSam(outputs),upSam(labels))
                        traintotalloss = traintotalloss + loss
                        # forCyclic
                        curepoch = curepoch + 1

                        # save the PredictedHm, The Input Image and the IDx
                        imgToSave = np.asarray(images.detach().cpu())
                        OutputToSave = np.asarray(outputs.detach().cpu())
                        idxToSave = np.asarray(idx.detach().cpu())

                        np.save(
                            r'/mnt/d/Thesis/ThesisCode_Models/Model1/FirstModelThatWorked/Cyclic+LeakyReluModel/PredictedHeatmap_ImagePairs/Train/TrainImgs/Train_InputImg' + str(
                                curepoch) + '.npy', imgToSave)
                        np.save(
                            r'/mnt/d/Thesis/ThesisCode_Models/Model1/FirstModelThatWorked/Cyclic+LeakyReluModel/PredictedHeatmap_ImagePairs/Train/TrainHM/Train_PredictedHM' + str(
                                curepoch) + '.npy', OutputToSave)
                        np.save(
                            r'/mnt/d/Thesis/ThesisCode_Models/Model1/FirstModelThatWorked/Cyclic+LeakyReluModel/PredictedHeatmap_ImagePairs/Train/TrainIDX/Train_idxImg' + str(
                                curepoch) + '.npy', idxToSave)

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
            # for images, labels, idx in val_loader:
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

                    if CREATE_IMAGE_PAIRS == True:
                        imgToSave = np.asarray(images.detach().cpu())
                        OutputToSave = np.asarray(outputs.detach().cpu())
                        idxToSave = np.asarray(idx.detach().cpu())

                        np.save(
                            r'/mnt/d/Thesis/ThesisCode_Models/Model1/FirstModelThatWorked/Cyclic+LeakyReluModel/PredictedHeatmap_ImagePairs/Val/ValImgs/Val_InputImg' + str(
                                curepoch) + '.npy', imgToSave)
                        np.save(
                            r'/mnt/d/Thesis/ThesisCode_Models/Model1/FirstModelThatWorked/Cyclic+LeakyReluModel/PredictedHeatmap_ImagePairs/Val/ValHM/Val_PredictedHM' + str(
                                curepoch) + '.npy', OutputToSave)
                        np.save(
                            r'/mnt/d/Thesis/ThesisCode_Models/Model1/FirstModelThatWorked/Cyclic+LeakyReluModel/PredictedHeatmap_ImagePairs/Val/ValIDX/Val_idxImg' + str(
                                curepoch) + '.npy', idxToSave)

            totalloss = totalloss / curepoch
            # for reduce on plateu
            # lr_scheduler.step(totalloss)
            if CREATE_IMAGE_PAIRS == True:
                exit()

            wandb.log({"Val_loss": totalloss,
                       "Train_loss": traintotalloss})

            # Optional
            wandb.watch(model)

            test_is_best = totalloss < best_loss
            best_loss = min(totalloss, best_loss)

            # check if both trainingloss and testloss are decreasing
            is_best = (test_is_best == True) and (train_is_best == True)

            if CREATE_CHECKPOINT == True:
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
        for images, labels, idx in test_loader:
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

                if CREATE_IMAGE_PAIRS == True:
                    imgToSave = np.asarray(images.detach().cpu())
                    OutputToSave = np.asarray(outputs.detach().cpu())
                    idxToSave = np.asarray(idx.detach().cpu())

                    np.save(
                        r'/mnt/d/Thesis/ThesisCode_Models/Model1/FirstModelThatWorked/Cyclic+LeakyReluModel/PredictedHeatmap_ImagePairs/Test/TestImgs/Test_InputImg' + str(
                            curepoch) + '.npy', imgToSave)
                    np.save(
                        r'/mnt/d/Thesis/ThesisCode_Models/Model1/FirstModelThatWorked/Cyclic+LeakyReluModel/PredictedHeatmap_ImagePairs/Test/TestHM/Test_PredictedHM' + str(
                            curepoch) + '.npy', OutputToSave)
                    np.save(
                        r'/mnt/d/Thesis/ThesisCode_Models/Model1/FirstModelThatWorked/Cyclic+LeakyReluModel/PredictedHeatmap_ImagePairs/Test/TestIDX/Test_idxImg' + str(
                            curepoch) + '.npy', idxToSave)

        totalloss = totalloss / curepoch
        print('Test MSE Loss: ', np.asarray(totalloss.detach().cpu()))


CONFIG = {
    'batch_size': 8,
    'epochs': 1000,
    'num_filters': 5,
    'num_filters_FC': 5,
    'momentum': 0.9,
    'learning_rate': 5e-3,
    'padding_mode': 'replicate',
    'upsample': 'bilinear',
    'leakySlope': 0.0189,
    'dropout': 0.75

}

LOAD_CHECKPOINT = False
CHECKPOINT_PATH = r'/mnt/d/Thesis/ThesisCode_Models/ModelPart1/FirstModelThatWorked/Cyclic+LeakyReluModel/checkpoint_best_loss.pth.tar'
RUN_TEST_DATA = False
CREATE_IMAGE_PAIRS = False
CREATE_CHECKPOINT = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network",
                        help="network architecture to use",
                        default='BestPosModel_WithDenseLayer',
                        choices=['SphericalModel', 'PaperBased', 'BestPosModel_WithDenseLayer', 'GithubBasedPaper',
                                 'ResNetBased',
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

        SWEEP_CONFIG = {'method': 'bayes'}  # random, bayes, grid
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
            'leakySlope': {'value': 0.0189},
            'dropout': {'distribution': 'uniform',
                        'min': 0,
                        'max': 0.5}

        }
        SWEEP_CONFIG['parameters'] = parameters_dict

        parameters_dict.update({'epochs': {'value': 16}})
        parameters_dict.update({'batch_size': {'value': 16}})
        sweep_id = wandb.sweep(SWEEP_CONFIG, project="FinalTest_pos_model")
        wandb.agent(sweep_id, function=train_function, count=25)  # main(args.network)
    else:
        train_function()
