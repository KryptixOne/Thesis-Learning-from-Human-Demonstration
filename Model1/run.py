# pylint: disable=E1101,R,C
import numpy as np
import torch.nn as nn
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

# temp path for Data
DATASET_PATH_win = r"D:/Thesis/ThesisCode_Models/DataToWorkWith/Data_Spherical_With_PosPmaps60.pickle"
DATASET_PATH_lin = '/mnt/d/Thesis/ThesisCode_Models/DataToWorkWith/Data_Spherical_With_PosPmaps60.pickle'
DATASET_SMALL_LIN = '/mnt/d/Thesis/ThesisCode_Models/Model1/Datasmall/dataDepthAndPosHM2000.pickle'
DATASET_LARGE_LIN = '/mnt/d/Thesis/ThesisCode_Models/Model1/Datasmall/dataDepthAndPosHM.pickle'
MNIST_PATH = "s2_mnist.gz"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 5e-3


def load_data(path, batch_size):
    pre_split_available = True
    data_not_formatted = False
    load_test_data = False

    if pre_split_available == False:  # if pre-split data is not available
        with open(path, 'rb') as f:
            dataset = pickle.load(f)

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
        # For Data creation when data isn't formatted
        k = 0
        keylist = list(dataset.keys())
        for item in keylist:
            for x in range(6):
                currobj = dataset[item][x]
                currdepth = currobj[None, 0, :, :]
                currPosHeatMap = currobj[None, 5, :, :]

                if item == keylist[0] and x == 0:
                    depthimages = currdepth
                    posHeatMap = currPosHeatMap
                else:
                    depthimages = np.concatenate((depthimages, currdepth), axis=0)
                    posHeatMap = np.concatenate((posHeatMap, currPosHeatMap), axis=0)

            if k % 1000 == 0:
                data_dict = {'depthImages': depthimages, 'positionHeatMap': posHeatMap}
                # with open('dataDepthAndPosHM.pickle', 'wb') as handle:
                #    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            del dataset[item]
            k = k + 1

            # data_dict = {'depthImages':depthimages, 'positionHeatMap': posHeatMap}


    train_data = torch.from_numpy(
        X_train[:, None, :, :].astype(np.float32))  # creates extra dimension [60000, 1, 60,60]
    X_train = None  # free up memory
    
    train_labels = torch.from_numpy(
        Y_train.astype(np.float32))

    Y_train = None  # free up memory

    # TODO normalize dataset
    # mean = train_data.mean()
    # stdv = train_data.std()

    train_dataset = data_utils.TensorDataset(train_data, train_labels)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_data = torch.from_numpy(
        X_val[:, None, :, :].astype(np.float32))
    X_val = None  # free up memory

    val_labels = torch.from_numpy(
        Y_val.astype(np.int64))
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

        self.convolutional = nn.Sequential(
            S2Convolution(
                nfeature_in=1,
                nfeature_out=8,
                b_in=bandwidth,
                b_out=bandwidth,
                grid=grid_s2),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in=8,
                nfeature_out=16,
                b_in=bandwidth,
                b_out=bandwidth // 2,
                grid=grid_so3_1),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in=16,
                nfeature_out=16,
                b_in=bandwidth // 2,
                b_out=bandwidth // 2,
                grid=grid_so3_2),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in=16,
                nfeature_out=24,
                b_in=bandwidth // 2,
                b_out=bandwidth // 4,
                grid=grid_so3_2),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in=24,
                nfeature_out=24,
                b_in=bandwidth // 4,
                b_out=bandwidth // 4,
                grid=grid_so3_3),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in=24,
                nfeature_out=32,
                b_in=bandwidth // 4,
                b_out=bandwidth // 8,
                grid=grid_so3_3),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in=32,
                nfeature_out=64,
                b_in=bandwidth // 8,
                b_out=bandwidth // 8,
                grid=grid_so3_4),
            nn.ReLU(inplace=False)
            ,

            # added parts
            SO3Convolution(
                nfeature_in=64,
                nfeature_out=32,
                b_in=bandwidth // 8,
                b_out=bandwidth // 8,
                grid=grid_so3_4),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in=32,
                nfeature_out=24,
                b_in=bandwidth // 8,
                b_out=bandwidth // 4,
                grid=grid_so3_3),
            nn.ReLU(inplace=False)
            , SO3Convolution(
                nfeature_in=24,
                nfeature_out=24,
                b_in=bandwidth // 4,
                b_out=bandwidth // 4,
                grid=grid_so3_3),
            nn.ReLU(inplace=False)
            ,
            SO3Convolution(
                nfeature_in=24,
                nfeature_out=16,
                b_in=bandwidth // 4,
                b_out=bandwidth // 2,
                grid=grid_so3_2),
            nn.ReLU(inplace=False)
            ,
            SO3Convolution(
                nfeature_in=16,
                nfeature_out=16,
                b_in=bandwidth // 2,
                b_out=bandwidth // 2,
                grid=grid_so3_2),
            nn.ReLU(inplace=False)
            ,
            SO3Convolution(
                nfeature_in=16,
                nfeature_out=8,
                b_in=bandwidth // 2,
                b_out=bandwidth,
                grid=grid_so3_1),
            nn.ReLU(inplace=False)
            ,
            SO3Convolution(
                nfeature_in=8,
                nfeature_out=1,
                b_in=bandwidth,
                b_out=bandwidth,
                grid=grid_so3_1),
            nn.ReLU(inplace=False)

            # tx = t.view(t.size(0), t.size(1), t.size(2),t.size(3),-1).max(-1)[0]
        )

        self.linear = nn.Sequential(
            # linear 1
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(inplace=False),
            # linear 2
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=False),
            # linear 3
            nn.BatchNorm1d(32),
            nn.Linear(in_features=32, out_features=10)
        )

    def forward(self, x):
        x = self.convolutional(x)
        print()
        x = so3_integrate(x)
        x = self.linear(x)
        return x


def main(network):
    train_loader, test_loader, train_dataset, _ = load_data(
        DATASET_LARGE_LIN, BATCH_SIZE)

    if network == 'SphericalModel':
        classifier = SphericalModelDeep()
    else:
        raise ValueError('Unknown network architecture')
    classifier.to(DEVICE)

    print("#params", sum(x.numel() for x in classifier.parameters()))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            classifier.train()

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
                epoch + 1, NUM_EPOCHS, i + 1, len(train_dataset) // BATCH_SIZE,
                loss.item()), end="")
        print("")
        correct = 0
        total = 0
        for images, labels in test_loader:
            classifier.eval()

            with torch.no_grad():
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = classifier(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).long().sum().item()

        print('Test Accuracy: {0}'.format(100 * correct / total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network",
                        help="network architecture to use",
                        default='SphericalModel',
                        choices=['SphericalModel'])
    args = parser.parse_args()

    main(args.network)
