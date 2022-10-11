import random
import argparse
import numpy as np
import torch
import learn2learn as l2l

from torch import nn, optim
import wandb
from torch.utils.data import Dataset
import torchvision.transforms
import shutil

from Models.ModelsPart2 import RBranchEarlyExit
from Dataloader.FolderDataLoader import FolderData
from ImageFunctions.ImageCreationFunctions import create_images

VAL1SAME = 0
VAL2SAME = 0

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


def AugmentTasks(labels, NoiseToRegressionTargetAngles=False, LikelihoodsUpdates=False, UseSameNoise=False,
                 ChooseRandomPoints=False, NumRandomPoints=None):
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

    if LikelihoodsUpdates == True:
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
        # choose few random points in likelihood map, hopefully will represent an update hm
        if NumRandomPoints is None:
            NumRandomPoints = 5
        blur = torchvision.transforms.GaussianBlur(3, sigma=1.0)
        filterX3 = augmentedLabels[:, 3, :, :] < 0.5  # only areas with low probability selected
        filterX3 = filterX3[:, None, :, :]

        newOnes = torch.zeros(filterX3.shape)
        for imgNum in range(filterX3.shape[0]):
            curFilter = filterX3[imgNum, :, :, :]
            curFilter = curFilter[None, :, :, :]
            onesmade = torch.where(curFilter == False, torch.ones(curFilter.shape), torch.zeros(curFilter.shape))

            if torch.count_nonzero(onesmade) < (2 * NumRandomPoints):
                newOnes[imgNum, :, :, :] = onesmade
                continue

            saveProb = NumRandomPoints / torch.count_nonzero(onesmade)

            idx = torch.where(onesmade == 1)
            lengthTorch = idx[0].size()
            randIdxtorch = torch.randperm(lengthTorch[0])
            randIdxtorch = randIdxtorch[:int(lengthTorch[0] * saveProb)]
            random_indices = (idx[0][randIdxtorch], idx[1][randIdxtorch], idx[2][randIdxtorch], idx[3][randIdxtorch])
            z = torch.zeros(onesmade.shape)
            z[random_indices] = 1
            z = blur(z)
            z = (1 / torch.amax(z)) * z
            newOnes[imgNum, :, :, :] = z

        # now use new HM to filter out crap
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
        filterX3 = augmentedLabels[:, 3, :, :] < 0.5  # only areas with low probability selected
        filterX3 = filterX3[:, None, :, :]
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
        x3 = augmentedLabels[:, 3, :, :][:, None, :, :].detach().clone()
        x3[filterX3] = 0

        augmentedLabels = torch.cat((x, x3), dim=1)

    return augmentedLabels


def fast_adapt(batch, learner, loss, adaptation_steps, device,
               NoiseToTarget=False, LikelihoodUpdate=False, UseSameNoise=False, ChooseRandomPoints=False):
    data, labels = batch
    labels = AugmentTasks(labels, NoiseToRegressionTargetAngles=NoiseToTarget, LikelihoodsUpdates=LikelihoodUpdate,
                          UseSameNoise=UseSameNoise, ChooseRandomPoints=ChooseRandomPoints)
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
        # print('adapt step:' , step)
        # print('train error:' ,train_error)

        # print('valid error:',valid_error )

    # Evaluate the adapted model
    pred = learner(adaptation_data)
    train_error = loss(pred, adaptation_labels)
    predictions = learner(evaluation_data)
    # valid_error = loss(predictions, evaluation_labels)
    # print(valid_error)

    return train_error, predictions, evaluation_labels  # , valid_accuracy


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")


def main(
        config=None, trainloader=None, valloader=None, testloader=None, testtestloader=None
):
    meta_batch_size = int(config['meta_batch_size'])  # since total of 15 objects per category/ task object
    batch_size = int(config['batch_size'])  # 6 per object, so 5 objects
    adapt_lr = config['adapt_lr']
    meta_lr = config['meta_lr']
    adaptation_steps = int(config['adaptation_steps'])
    test_steps = 10
    seed = 42
    Create_checkpoint = True
    load_checkpoint = True
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
    CHECKPOINT_PATH = r'/mnt/d/Thesis/ThesisCode_Models/ModelPart2/AugmentedData_RandomPoints_checkpoint.pth.tar'
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
                meta_testtest_error = 0.0
                for task in range(1):  # 14 since first category will have 84 total objects 6*14 =84
                    # Compute meta-testing loss
                    learner = maml.clone()

                    evaluation_error, predicted, evalLabels = fast_adapt(batch,
                                                                         learner,
                                                                         loss,
                                                                         test_steps,
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

                evaluation_error, predicted, evalLabels = fast_adapt(batch,
                                                                     learner,
                                                                     loss,
                                                                     test_steps,
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

            evaluation_error, predicted, evalLabels = fast_adapt(batch,
                                                                 learner,
                                                                 loss,
                                                                 adaptation_steps,
                                                                 device, NoiseToTarget=True, LikelihoodUpdate=False,
                                                                 ChooseRandomPoints=True
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

            evaluation_error, predicted, evalLabels = fast_adapt(batch,
                                                                 learner,
                                                                 loss,
                                                                 adaptation_steps,
                                                                 device, NoiseToTarget=True, LikelihoodUpdate=False,
                                                                 ChooseRandomPoints=True
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
        # if epoch % 50 == 0:
        #    wandb.log({"Val_loss": (meta_valid_error / meta_batch_size),
        #               "Train_loss": (meta_train_error / meta_batch_size)})

        # Optional
        # wandb.watch(model)
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
                is_best, filename="AugmentedData_RandomPoints_checkpoint.pth.tar"
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--HSweep",
                        help="To sweep or not",
                        default='NoSweep',
                        choices=['Sweep', 'NoSweep'])

    args = parser.parse_args()
    print('loading data')

    DataPathTrain = r'/mnt/d/Thesis/ThesisCode_Models/DataWithOrientationHM_normalized/Data/Train'
    DataPathVal = r'/mnt/d/Thesis/ThesisCode_Models/DataWithOrientationHM_normalized/Data/Val'
    DataPathTestadapt = r'/mnt/d/Thesis/ThesisCode_Models/DataWithOrientationHM_normalized/Data/TestAdapt'
    DataPathTesttest = r'/mnt/d/Thesis/ThesisCode_Models/DataWithOrientationHM_normalized/Data/TestTest'
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
