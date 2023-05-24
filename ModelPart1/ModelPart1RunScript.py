# pylint: disable=E1101,R,C
import torch.nn as nn
import torch
import torch.utils.data as data_utils
import numpy as np
import argparse
from math import sqrt, ceil
import shutil
import wandb
from matplotlib import pyplot as plt


from Dataloader.FolderDataPart1 import FolderData
from Models.Models import SphericalModelDeep, ResNetModelBased, ModelBasedOnPaperGitHubVersionSINGLEBRANCH, \
    SingleBranchHpLp, ModelBasedOnPaperGitHubVersion, ModelBasedOnPaperNoneSpherical, FirstWorkingModelForPos_NoDense, \
    BestPosModel_WithDenseLayer

# temp path for Data
DATASET_PATH_lin_FolderData = r'/mnt/d/Thesis/ThesisCode_Models/DataWithOrientationHM_normalized/Data/'


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


def load_data(path, batch_size, network=None):
    start_training = True

    if start_training == True:

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
            DATASET_PATH_lin_FolderData, config['batch_size'], network)
    else:
        train_loader, val_loader, train_dataset, _ = load_data(
            DATASET_PATH_lin_FolderData, config['batch_size'],  network)

    if network == 'SphericalModel':
        model = SphericalModelDeep()
    elif network == 'PaperBased':
        model = ModelBasedOnPaperNoneSpherical()
    elif network == 'GithubBasedPaper':
        model = ModelBasedOnPaperGitHubVersion(config)
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
                                     labels)

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
                                         labels)
                        traintotalloss = traintotalloss + loss
                        # forCyclic
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


            totalloss = totalloss / curepoch
            # for reduce on plateu
            # lr_scheduler.step(totalloss)

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
CREATE_IMAGE_PAIRS = False #keep false
CREATE_CHECKPOINT = True
DATASET_PATH_lin_FolderData = r'/mnt/d/Thesis/ThesisCode_Models/DataWithOrientationHM_normalized/Data/' #dataset Path
WanbEntity = 'InsertYourWanB_ID'

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
        with wandb.init(project='FinalTest_pos_model', config=CONFIG, entity=WanbEntity):
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
