
import numpy as np
import torch


from torch.utils.data import Dataset
from pathlib import Path
import pickle

from random import shuffle

class FolderData(Dataset):
    """Load Data

        DataPath should be towards the correct test/train/validation folder"""

    def __init__(self, DataPath, train_testFlag='train', ModelSelect='Part1', Singleimg=False, AddNoise=False,
                 ):
        path = Path(DataPath)
        self.ModelSelect = ModelSelect
        self.Singleimg = Singleimg
        self.AddNoise = AddNoise
        self.variance = 1e-4

        if not path.is_dir():
            raise RuntimeError(f'Invalid directory "{path}"')

        self.samples = []
        for e in path.iterdir():
            tempsamp = [f for f in e.iterdir() if f.is_file()]
            if train_testFlag == 'train':  # even distribution of training data
                # each category has 15 objects with 6 rotations
                tempsamp = tempsamp[0:90]
                self.samples = self.samples + tempsamp
            else:
                self.samples = self.samples + tempsamp
            # create subgroups based on the item
        if train_testFlag == 'train':
            self.samples = [self.samples[i:i + 6] for i in range(0, len(self.samples), 6)]
            # shuffle the items groupped in 6 and flatten list
            shuffle(self.samples)
            self.samples = [item for sublist in self.samples for item in sublist]

    def __getitem__(self, index):

        if self.ModelSelect == 'Part1':  # return Inputspherical + Target heatmaps
            with open(self.samples[index], 'rb') as f:
                curdata = pickle.load(f)

                depthimages = curdata['depthImages']
                posHeatMap = curdata['positionHeatMap']

                inputData = curdata['depthImages']
                inputData = torch.from_numpy(inputData[:, :, :].astype(np.float32))

                targetHM = curdata['positionHeatMap']
                targetHM = torch.from_numpy(targetHM[:, :, :].astype(np.float32))
                outputData = targetHM


        elif self.ModelSelect == 'Part2': #return Input Sppherical + Positional HM , Target Orientation HM
            with open(self.samples[index], 'rb') as f:
                curdata = pickle.load(f)

            inputImg = curdata['depthImages']
            inputHM = curdata['positionHeatMap']

            inputImg = torch.from_numpy(inputImg[:, :, :].astype(np.float32))

            inputHM = torch.from_numpy(inputHM[:, :, :].astype(np.float32))

            outputposHM = curdata['positionHeatMap']

            inputData = torch.cat((inputImg, inputHM), dim=0)
            # make anngle data between 0 -180. If angle is 0, we make it 180

            ThetaAng = curdata['ThetaAng']
            # ThetaHm = curdata['ThetaHm']
            # ThetaHm[ThetaHm < 0.5] = 0

            PhiAngle = curdata['PhiAngle']
            # PhiHm = curdata['PhiHm']
            # PhiHm[PhiHm < 0.5] = 0

            GammaAngle = curdata['GammaAngle']
            Less_than = GammaAngle < 0
            GammaAngle[Less_than] = GammaAngle[Less_than] + 180

            # GammaHm = curdata['GammaHm']

            # GammaHm[GammaHm < 0.5] = 0

            ThetaAng = torch.from_numpy(ThetaAng[:, :, :].astype(np.float32))
            # ThetaHm = torch.from_numpy(ThetaHm[:, :, :].astype(np.float32))

            PhiAngle = torch.from_numpy(PhiAngle[:, :, :].astype(np.float32))
            # PhiHm = torch.from_numpy(PhiHm[:, :, :].astype(np.float32))

            GammaAngle = torch.from_numpy(GammaAngle[:, :, :].astype(np.float32))
            # GammaHm = torch.from_numpy(GammaHm[:, :, :].astype(np.float32))

            OutPosHm = torch.from_numpy(outputposHM[:, :, :].astype(np.float32))

            if self.Singleimg == False:
                # outputData = torch.cat((ThetaAng, PhiAngle, GammaAngle, ThetaHm, PhiHm, GammaHm, OutPosHm), dim=0)
                outputData = torch.cat((ThetaAng, PhiAngle, GammaAngle, OutPosHm), dim=0)
            else:
                # outputData = torch.cat((ThetaAng, ThetaHm), dim=0)
                pass
            # outputData = torch.cat((ThetaAng, PhiAngle, GammaAngle), dim=0)

        elif self.ModelSelect == 'Full':
            with open(self.samples[index], 'rb') as f:
                curdata = pickle.load(f)

                inputData = curdata['depthImages']
                inputData = torch.from_numpy(inputData[:, :, :].astype(np.float32))


                outputposHM = curdata['positionHeatMap']
                ThetaAng = curdata['ThetaAng']
                PhiAngle = curdata['PhiAngle']

                GammaAngle = curdata['GammaAngle']
                Less_than = GammaAngle < 0
                GammaAngle[Less_than] = GammaAngle[Less_than] + 180
                ThetaAng = torch.from_numpy(ThetaAng[:, :, :].astype(np.float32))
                PhiAngle = torch.from_numpy(PhiAngle[:, :, :].astype(np.float32))
                GammaAngle = torch.from_numpy(GammaAngle[:, :, :].astype(np.float32))
                OutPosHm = torch.from_numpy(outputposHM[:, :, :].astype(np.float32))

                outputData = torch.cat((ThetaAng, PhiAngle, GammaAngle, OutPosHm), dim=0)


        if self.AddNoise == True:
            inputData = inputData + (self.variance ** 0.5) * torch.rand(inputData.shape)
            # make sure data is capped at  despite addition of noise
            data1 = inputData > 1
            inputData[data1] = 1

        return (inputData, outputData)

    def __len__(self):
        return len(self.samples)

