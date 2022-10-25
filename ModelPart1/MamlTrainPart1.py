from ModelPart1.Models.Models import EvenDataModel
from ModelPart2.Models.ModelsPart2 import RBranchEarlyExit
from ModelPart2.Dataloader.FolderDataLoader import FolderData
import torch
import learn2learn as l2l
import numpy as np
import random
import torchvision
import torch.utils.data as data_utils
import shutil

import math
from matplotlib import pyplot as plt
def create_images(TensorImg, nameOfFile):
    subplotValue = math.ceil(math.sqrt(TensorImg.shape[0]))
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


def fast_adapt(batch, learner, loss, adaptation_steps, device, ChooseRandomPoints=False):
    data, labels = batch
    labels = AugmentTasks(labels, ChooseRandomPoints=ChooseRandomPoints)

    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
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

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)


    return valid_error, predictions, evaluation_labels  # , valid_accuracy


def AugmentTasks(labels,
                 ChooseRandomPoints=False, NumRandomPoints=None):
    global config1
    augmentedLabels = labels.detach().clone()


    if ChooseRandomPoints == True:
        # choose few random points in likelihood map, hopefully will represent an update hm
        if NumRandomPoints is None:
            NumRandomPoints = 10
        blurredsize = random.randrange(5,10,2)
        blur = torchvision.transforms.GaussianBlur(9, sigma=1.0)
        #maskValue = random.uniform(0.25, 1) #thereby all point will exist within some range
        filterX3 = augmentedLabels[:, 0, :, :] < 0.5  # only areas with low probability selected
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

        augmentedLabels =  newOnes

    else:
        blur = torchvision.transforms.GaussianBlur(9, sigma=1.0)
        augmentedLabels[:,:,:,:] = blur(augmentedLabels[:,:,:,:])
    return augmentedLabels


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "MAMLTrainedPart1_checkpoint_best_loss.pth.tar")


config1 = {
    'batch_size': 4,
    'epochs': 3000,
    'num_filters': 5,
    'num_filters_FC': 5,
    'momentum': 0.9,
    'learning_rate': 5e-3,
    'padding_mode': 'replicate',
    'upsample': 'bilinear',
    'leakySlope': 0.0189,
    'dropout': 0.0,
    'iterations':3000,
    'meta_batch_size':4,
    'adaptation_steps':15,
    'maskvalue':0.25,
    'adapt_lr':0.1,
    'meta_lr':0.04

}

seed = 42

load_checkpoint = False
Create_checkpoint = True
testTheModel =False
path = r'/mnt/d/Thesis/ThesisCode_Models/DataWithOrientationHM_normalized/Data/'
trainPath = path + r'/Train'
valPath = path + r'/Val'

Test_path = r'/mnt/d/Thesis/HumanDemonstrated/HumanDemonstrated_Normalized/ToolUse'

if testTheModel == False:
    train_dataset = FolderData(trainPath, train_testFlag='train', ModelSelect='Part1')
    val_dataset = FolderData(valPath, train_testFlag='val', ModelSelect='Part1')
    train_loader = data_utils.DataLoader(train_dataset, batch_size=config1['batch_size'], shuffle=True)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=config1['batch_size'], shuffle=True)

    val_iter = iter(val_loader)
    train_iter = iter(train_loader)
    trainbest_loss = float("inf")
    valbest_loss = float("inf")
else:
    testTaskSets = FolderData(Test_path, ModelSelect='Part1', train_testFlag='test', Singleimg=False)
    testLoader = torch.utils.data.DataLoader(testTaskSets, batch_size=8)
    testIter = iter(testLoader)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda:0")

# define the models
modelpart1 = EvenDataModel(config1)
modelpart1.to(device)

maml = l2l.algorithms.MAML(modelpart1, lr=config1['adapt_lr'], first_order=True, allow_unused=True,
                           allow_nograd=True)
opt = torch.optim.SGD(maml.parameters(), config1['meta_lr'], momentum= config1['momentum'],nesterov =True, weight_decay =1e-4)

loss = torch.nn.MSELoss()

CHECKPOINT_PATH =  r'/mnt/d/Thesis/ThesisCode_Models/ModelPart1/MamlTrainedOnValid_noDropout_ModelPart1.pth.tar'
if load_checkpoint == True:
    print('loading checkpoint from ' + CHECKPOINT_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    modelpart1.load_state_dict(checkpoint["model_state_dict"])
    opt.load_state_dict((checkpoint["optimizer"]))
    maml.load_state_dict(checkpoint['maml_state_dict'])
    checkpoint = None  # Free up GPU memory


if testTheModel == False:
    for epoch in range(int(config1['iterations'])):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_valid_error = 0.0
        meta_test_error = 0.0
        meta_testtest_error = 0.0
        test_size = 1
        gradSteps = 5

        for task in range(config1['meta_batch_size']):  # get train and validation Data
            # Compute meta-training loss
            learner = maml.clone()
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            likeupdate = False
            randompoint = True

            evaluation_error, predicted, evalLabels = fast_adapt(batch,
                                                                 learner,
                                                                 loss,
                                                                 config1['adaptation_steps'],
                                                                 device,
                                                                 ChooseRandomPoints=randompoint
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
                val_iter = iter(val_loader)
                batch = next(val_iter)

            evaluation_error, predicted, evalLabels = fast_adapt(batch,
                                                                 learner,
                                                                 loss,
                                                                 config1['adaptation_steps'],
                                                                 device,
                                                                 ChooseRandomPoints=randompoint
                                                                 )

            meta_valid_error += evaluation_error.item()

        # Print some metrics
        print('\n')
        print('epoch', epoch)
        print('Meta Train Error', meta_train_error / config1['meta_batch_size'])
        print('Meta Valid Error', meta_valid_error / config1['meta_batch_size'])

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1.0 / config1['meta_batch_size'])
        opt.step()

        train_is_best = (meta_train_error / config1['meta_batch_size']) < trainbest_loss
        val_is_best = (meta_valid_error / config1['meta_batch_size']) < valbest_loss
        valbest_loss = min(meta_valid_error / config1['meta_batch_size'], valbest_loss)
        trainbest_loss = min(meta_train_error / config1['meta_batch_size'], trainbest_loss)

        # check if both trainingloss and testloss are decreasing
        is_best = (val_is_best == True) and (train_is_best == True)

        if Create_checkpoint == True:
            save_checkpoint(
                {
                    "iterations": epoch,
                    "maml_state_dict": maml.state_dict(),
                    "model_state_dict": modelpart1.state_dict(),
                    "valloss": valbest_loss,
                    'trainloss': trainbest_loss,
                    "optimizer": opt.state_dict(),
                },
                is_best, filename="MamlTrainedOnValid_WithMomentumAndL2Reg_ModelPart1.pth.tar"
            )

else:

    try:
        batch = next(testIter)
    except StopIteration:
        testIter = iter(testLoader)
        batch = next(testIter)
    test_size = 1
    gradSteps = 10
    for grad in range(gradSteps):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_valid_error = 0.0
        meta_test_error = 0.0
        meta_testtest_error = 0.0
        for task in range(test_size):  # get train and validation Data
            # Compute meta-training loss
            learner = maml.clone()


            likeupdate = False
            randompoint = False

            evaluation_error, predicted, evalLabels = fast_adapt(batch,
                                                                 learner,
                                                                 loss,
                                                                 config1['adaptation_steps'],
                                                                 device
                                                                 )

            evaluation_error.backward()
            meta_train_error += evaluation_error.item()

            # Compute meta-validation loss
            learner = maml.clone()


            for p in maml.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(1.0 / test_size)
            opt.step()
            learner.eval()
            evaluation_error, predicted, evalLabels = fast_adapt(batch,
                                                                 learner,
                                                                 loss,
                                                                 config1['adaptation_steps'],
                                                                 device
                                                                 )

            meta_valid_error += evaluation_error.item()

        # Print some metrics
        print('\n')
        print('grad', grad)
        print('Meta Train Error', meta_train_error / config1['meta_batch_size'])
        print('Meta Test Error', meta_valid_error / config1['meta_batch_size'])
        create_images(evalLabels, 'evalLabels_' + str(grad)+ '_withError_'+str(meta_valid_error / config1['meta_batch_size'])[2:5])
        create_images(predicted, 'predicted_' + str(grad)+'_withError_'+str(meta_valid_error / config1['meta_batch_size'])[2:5])
        print()

