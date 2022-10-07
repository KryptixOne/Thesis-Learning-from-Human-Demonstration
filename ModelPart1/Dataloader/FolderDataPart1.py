
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from pathlib import Path


class FolderData(Dataset):
    """Load Data

        DataPath should be towards the correct test/train/validation folder"""

    def __init__(self, DataPath, train_testFlag='train', ModelSelect='Part1'):
        path = Path(DataPath)
        self.ModelSelect = ModelSelect

        if not path.is_dir():
            raise RuntimeError(f'Invalid directory "{path}"')

        self.samples = []
        #get all item directories
        for e in path.iterdir():
            tempsamp = [f for f in e.iterdir() if f.is_file()]
            if train_testFlag == 'train':  # even distribution of training data
                tempsamp = tempsamp[0:90]
                self.samples = self.samples + tempsamp
            else:
                self.samples = self.samples + tempsamp

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
            Load in the correct file given the path.
            Identifies correct Data
        Returns:
            input (spherical img)
            targer (Heatmaps)

             test_data = torch.from_numpy(
                X_test[:, None, :, :].astype(np.float32))
            X_test = None  # free up memory1

            test_labels = torch.from_numpy(
                Y_test[:, None, :, :].astype(np.float32))

                Input Dictionary Structure:
                    depthimages = curdata['depthImages']
                    posHeatMap = curdata['positionHeatMap']
                    ThetaAng = curdata['ThetaAng']
                    ThetaHm = curdata['ThetaHm']
                    PhiAngle = curdata['PhiAngle']
                    PhiHm= curdata['PhiHm']
                    GammaAngle = curdata['GammaAngle']
                    GammaHm = curdata['GammaHm']
        """
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

        else:  # ModelSelect == 'Part2' return Input Sppherical + Positional HM , Target Orientation HM
            with open(self.samples[index], 'rb') as f:
                curdata = pickle.load(f)

            inputImg = curdata['depthImages']
            inputImg = torch.from_numpy(inputImg[:, :, :].astype(np.float32))

            inputHM = curdata['positionHeatMap']
            inputHM = torch.from_numpy(inputHM[:, :, :].astype(np.float32))

            inputData = torch.cat((inputImg, inputHM), dim=0)

            ThetaAng = curdata['ThetaAng']
            ThetaHm = curdata['ThetaHm']
            PhiAngle = curdata['PhiAngle']
            PhiHm = curdata['PhiHm']
            GammaAngle = curdata['GammaAngle']
            GammaHm = curdata['GammaHm']

            ThetaAng = torch.from_numpy(ThetaAng[:, :, :].astype(np.float32))
            ThetaHm = torch.from_numpy(ThetaHm[:, :, :].astype(np.float32))

            PhiAngle = torch.from_numpy(PhiAngle[:, :, :].astype(np.float32))
            PhiHm = torch.from_numpy(PhiHm[:, :, :].astype(np.float32))

            GammaAngle = torch.from_numpy(GammaAngle[:, :, :].astype(np.float32))
            GammaHm = torch.from_numpy(GammaHm[:, :, :].astype(np.float32))

            outputData = torch.cat((ThetaAng, ThetaHm, PhiAngle, PhiHm, GammaAngle, GammaHm), dim=0)

        return (inputData, outputData)

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':
    pass