
import matplotlib.pyplot as plt
import numpy as np
import torch
import math

def create_images(TensorImg, nameOfFile, ManualSubplotValue=None, ModelPart2='predLabel'):
    if ManualSubplotValue == None:
        subplotValue = math.ceil(math.sqrt(TensorImg.shape[0] * TensorImg.shape[1]))
        fig1, axs = plt.subplots(subplotValue, subplotValue, figsize=(30, 30))
    else:
        subplotValue = math.ceil(math.sqrt(TensorImg.shape[0] * TensorImg.shape[1]))
        fig1, axs = plt.subplots(ManualSubplotValue[0], ManualSubplotValue[1], figsize=(30, 30))
    startx = 0
    starty = 0
    if ModelPart2 == 'predLabel':
        for x in range(TensorImg.shape[0]):
            for y in range(TensorImg.shape[1]):

                if starty % (ManualSubplotValue[1] - 1) == 0 and starty != 0:
                    img = np.asarray(TensorImg[x, y, :, :].detach().cpu())
                    axs[startx, starty].imshow(img)
                    im = axs[startx, starty].imshow(img)
                    im.set_clim(0, 1)
                elif starty % (ManualSubplotValue[1] - 2) == 0 and starty != 0:
                    img = np.asarray(TensorImg[x, y, :, :].detach().cpu())
                    axs[startx, starty].imshow(img)
                    im = axs[startx, starty].imshow(img)
                    im.set_clim(0, np.pi)
                else:
                    img = np.asarray(torch.cos(TensorImg[x, y, :, :]).detach().cpu())
                    axs[startx, starty].imshow(img)
                    im = axs[startx, starty].imshow(img)
                    im.set_clim(-1, 1)
                plt.colorbar(im, ax=axs[startx, starty])

                if starty % (ManualSubplotValue[1] - 1) == 0 and starty != 0:
                    starty = 0
                    startx = startx + 1
                else:
                    starty = starty + 1
    else:
        for x in range(TensorImg.shape[0]):
            for y in range(TensorImg.shape[1]):
                img = np.asarray(TensorImg[x, y, :, :].detach().cpu())
                axs[startx, starty].imshow(img)

                if starty % (ManualSubplotValue[1] - 1) == 0 and starty != 0:
                    starty = 0
                    startx = startx + 1
                else:
                    starty = starty + 1

    plt.tight_layout()
    plt.savefig(r'/mnt/d/Thesis/ThesisCode_Models/ModelPart2/CreatedImagesFromScript/'+str(nameOfFile))


def create_feature_images(TensorImg, nameOfFile):
    subplotValue = math.ceil(math.sqrt(TensorImg.shape[1]))
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

if __name__ == '__main__':
    pass