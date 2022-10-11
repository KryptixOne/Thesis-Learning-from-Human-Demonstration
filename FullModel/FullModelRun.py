from ModelPart1.Models.Models import BestPosModel_WithDenseLayer
from ModelPart2.Models.ModelsPart2 import RBranchEarlyExit
from ModelPart2.Dataloader.FolderDataLoader import FolderData
import torch
import learn2learn as l2l
import numpy as np


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
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    # print(valid_error)

    return valid_error, predictions, evaluation_labels  # , valid_accuracy


config1 = []
config2 = []
seed = 42
adapt_lr = 0.1
meta_lr = 0.04
Maml_test_Grad_steps = 5
test_adapt_steps = 15
loss = customLoss_7Channel()

CheckpointLoadModelPart1 = r'/mnt/d/Thesis/ThesisCode_Models/ModelPart1/Model1Results/FirstModelThatWorked/With_Even_Data/checkpoint_best_loss.pth.tar'
CheckpointLoadModelPart2 = r'/mnt/d/Thesis/ThesisCode_Models/ModelPart2/AugmentedData_RandomPoints_checkpoint.pth.tar'
TestInputImagesDirectory = r'/mnt/d/Thesis/HumanDemonstrated/HumanDemonstrated_Normalized/'

# create test-dataloader
testTaskSets = FolderData(TestInputImagesDirectory, ModelSelect='Part1', train_testFlag='test', Singleimg=False)
testLoader = torch.utils.data.DataLoader(testTaskSets, batch_size=32)
testIter = iter(testLoader)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda:0")

# define the models
modelpart1 = BestPosModel_WithDenseLayer(config1)
modelpart2 = RBranchEarlyExit(config2)
maml = l2l.algorithms.MAML(modelpart2, lr=adapt_lr, first_order=True, allow_unused=True, allow_nograd=True)
opt = torch.optim.SGD(maml.parameters(), meta_lr)

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
batch = next(testIter)
data, HumanDemonstratedlabels = batch
data, HumanDemonstratedlabels = data.to(device), HumanDemonstratedlabels.to(device)

# start predictions
with torch.no_grad():
    predictedPositionalHM = modelpart1(data)

# combine predicted positionalHM with the input Spherical Data
dataCombined = torch.cat((data[:, :, :, :], predictedPositionalHM[:, :, :, :]), dim=1)

for _ in range(Maml_test_Grad_steps):
    opt.zero_grad()
    meta_test_error = 0.0
    meta_testtest_error = 0.0
    for task in range(1):  # 14 since first category will have 84 total objects 6*14 =84
        # Compute meta-testing loss
        learner = maml.clone()

        evaluation_error, predicted, EvalLabels = fast_adapt(dataCombined,
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

    evaluation_error, predicted, EvalLabels = fast_adapt(dataCombined,
                                                         learner,
                                                         loss,
                                                         test_adapt_steps,
                                                         device,
                                                         LikelihoodUpdate=False, UseSameNoise=False)
    meta_testtest_error += evaluation_error.item()
    print('Post gradient error: ', meta_testtest_error)

print('Model Run Completed')
input('press any key to end run: ...')