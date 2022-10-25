from ModelPart1.Models.Models import EvenDataModel
from ModelPart2.Models.ModelsPart2 import RBranchEarlyExit
from ModelPart2.Dataloader.FolderDataLoader import FolderData
import torch
import learn2learn as l2l
import numpy as np
import random
import torchvision
from ModelPart2.ImageFunctions.ImageCreationFunctions import create_images

class customLoss_7Channel(torch.nn.Module):
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
        critMSE = torch.nn.MSELoss(reduction='mean')
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


def fast_adapt(batch, learner, loss, adaptation_steps, device,
               NoiseToTarget=False, LikelihoodUpdate=False, UseSameNoise=False, ChooseRandomPoints=False):
    data, labels = batch
    labels = AugmentTasks(labels, NoiseToRegressionTargetAngles=NoiseToTarget, LikelihoodsUpdates=LikelihoodUpdate,
                          UseSameNoise=UseSameNoise, ChooseRandomPoints=ChooseRandomPoints)

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

    return valid_error, predictions, evaluation_labels,evaluation_data  # , valid_accuracy

def AugmentTasks(labels, NoiseToRegressionTargetAngles=False, LikelihoodsUpdates=False, UseSameNoise=False,
                 ChooseRandomPoints=False, NumRandomPoints=None):
    global config2
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
        filterX3 = augmentedLabels[:, 3, :, :] < config2['maskvalue']  # only areas with low probability selected
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
        blur = torchvision.transforms.GaussianBlur(3, sigma=1.0)
        blurredaug = blur(augmentedLabels[:, 3, :, :])
        blurredaug = (1 / torch.amax(blurredaug)) * blurredaug
        filterX3 = blurredaug < config2['maskvalue']  # only areas with low probability selected
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
        x3 = blurredaug[:, None, :, :].detach().clone()
        x3[filterX3] = 0

        augmentedLabels = torch.cat((x, x3), dim=1)

    return augmentedLabels


config1 = {
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
config2 = {
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
    'dropout': 0.8,
    'maskvalue':0.5

}
seed = 42
Maml_test_Grad_steps = 50
test_adapt_steps = 60
loss = customLoss_7Channel()

UseBothParts =True


if UseBothParts ==True:

    CheckpointLoadModelPart1 = r'/mnt/d/Thesis/ThesisCode_Models/ModelPart1/Model1Results/FirstModelThatWorked/With_Even_Data/checkpoint_best_loss.pth.tar'
    CheckpointLoadModelPart2 = r'/mnt/d/Thesis/ThesisCode_Models/ModelPart2/Results/PostFurkanLoss/Post_Furkan_chat_DualLoss_Best_DeepNet_checkpoint.pth.tar'
    TestInputImagesDirectory = r'/mnt/d/Thesis/HumanDemonstrated/HumanDemonstrated_Normalized/ToolUse'

    # create test-dataloader
    testTaskSets = FolderData(TestInputImagesDirectory, ModelSelect='Full', train_testFlag='test', Singleimg=False)
    testLoader = torch.utils.data.DataLoader(testTaskSets, batch_size=32)
    testIter = iter(testLoader)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        device = torch.device("cuda:0")

    # define the models
    modelpart1 = EvenDataModel(config1)
    modelpart1.to(device)
    modelpart2 = RBranchEarlyExit(config2)
    modelpart2.to(device)
    maml = l2l.algorithms.MAML(modelpart2, lr=config2['adapt_lr'], first_order=True, allow_unused=True, allow_nograd=True)
    opt = torch.optim.SGD(maml.parameters(),config2['meta_lr'])

    # load checkpoint for part1
    checkpoint1 = torch.load(CheckpointLoadModelPart1, map_location=device)
    modelpart1.load_state_dict(checkpoint1["state_dict"])

    # load checkpoint for part2
    checkpoint2 = torch.load(CheckpointLoadModelPart2, map_location=device)
    modelpart2.load_state_dict(checkpoint2["model_state_dict"])
    maml.load_state_dict(checkpoint2['maml_state_dict'])

    # model initialization
    modelpart1.eval()
    modelpart2.train()

    # load the test data


    for curstep in range(Maml_test_Grad_steps):
        opt.zero_grad()
        meta_test_error = 0.0
        meta_testtest_error = 0.0
        try:
            batch = next(testIter)
        except StopIteration:
            testIter = iter(testLoader)
            batch = next(testIter)

        data, HumanDemonstratedlabels = batch
        data, HumanDemonstratedlabels = data.to(device), HumanDemonstratedlabels.to(device)

        # start predictions
        with torch.no_grad():
            predictedPositionalHM = modelpart1(data)

        # combine predicted positionalHM with the input Spherical Data
        dataCombined = torch.cat((data[:, :, :, :], predictedPositionalHM[:, :, :, :]), dim=1)

        TestUsingTree = True
        if TestUsingTree== True:
            X =dataCombined
            y =HumanDemonstratedlabels[:,3,:,:]
            from sklearn.ensemble import ExtraTreesRegressor
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.linear_model import RidgeCV

            ESTIMATORS = {
                "Extra trees": ExtraTreesRegressor(
                    n_estimators=10, max_features=32, random_state=0
                ),
                "K-nn": KNeighborsRegressor(),
                "Linear regression": LinearRegression(),
                "Ridge": RidgeCV(),
            }



        for task in range(1):  # 14 since first category will have 84 total objects 6*14 =84
            # Compute meta-testing loss
            learner = maml.clone()

            evaluation_error, predicted, EvalLabels,evaluation_data = fast_adapt((dataCombined,HumanDemonstratedlabels),
                                                learner,
                                                loss,
                                                test_adapt_steps,
                                                device,
                                                NoiseToTarget=False, LikelihoodUpdate=False
                                                )
            meta_test_error += evaluation_error.item()
            # meta_test_accuracy += evaluation_accuracy.item()
        print('Meta Test Error', meta_test_error / 1)

        evaluation_error.backward()
        for p in maml.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1.0 / 1)
        opt.step()
        learner = maml.clone()

        evaluation_error, predicted, EvalLabels,evaluation_data = fast_adapt((dataCombined,HumanDemonstratedlabels),
                                                             learner,
                                                             loss,
                                                             test_adapt_steps,
                                                             device,
                                                             LikelihoodUpdate=False, UseSameNoise=False)
        meta_testtest_error += evaluation_error.item()
        print('Post gradient error: ', meta_testtest_error)
        create_images(predicted,'PredPostGrad'+str(curstep),(11,4))
        create_images(EvalLabels,'EvaluationLabels'+str(curstep),(11,4))
        create_images(evaluation_data,'evaluation_data'+str(curstep),(11,2),ModelPart2=None)

        evaluation_error, predictedWithNoise, EvalLabelsWithNoise, evaluation_data = fast_adapt((dataCombined, HumanDemonstratedlabels),
                                                                              learner,
                                                                              loss,
                                                                              test_adapt_steps,
                                                                              device,
                                                                              LikelihoodUpdate=False, UseSameNoise=False,NoiseToTarget=True)

    print('Model Run Completed')
    input('press any key to end run: ...')

