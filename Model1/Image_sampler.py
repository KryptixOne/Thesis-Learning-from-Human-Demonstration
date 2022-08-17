import numpy as np
import torch

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def ImageSampler_SingleTask(inputImgTensor, block_size =10, num_subtasks=5,non_zero_block_factor =3):
    """
    For a single input task, this will sample the heatmap accordingly and return the sampled heatmaps
    return: returns the sampled heatmaps
    """

    szTensor= inputImgTensor.size() #get input size
    inputImgTensor = np.asarray(inputImgTensor.detach().cpu()) #convert to numpy array
    numElementsXform = (szTensor[1]//block_size)**2
    nonZeroBlocks = numElementsXform //non_zero_block_factor
    zArray = np.zeros(numElementsXform)
    zArray[0:nonZeroBlocks]=1

    #create Md-Matrix of xform Matrices Used to sample img blocks.
    for x in range(num_subtasks):
        if x ==0:
            XformMat = np.expand_dims(np.random.permutation(zArray),axis =0)
        else:
            temp = np.random.permutation(zArray)
            XformMat= np.concatenate((XformMat,np.expand_dims(temp,axis=0)),axis=0)

    blocksofInput= blockshaped(inputImgTensor[1,:,:],block_size,block_size)
    #origblocks =  np.copy(blocksofInput)
    sampledInput =  np.expand_dims(np.copy(blocksofInput),axis =0)
    for x in range(XformMat.shape[0]):
        arraytoUse = XformMat[x,:]
        for y in range (arraytoUse.shape[0]):
            blocksofInput[y,:,:] = blocksofInput[y,:,:]*arraytoUse[y]
        if x ==0:
            sampledInput[x,:,:,:] = blocksofInput
        else:
            sampledInput = np.concatenate((sampledInput,np.expand_dims(blocksofInput,axis=0)))

    sampledHeatmaps = np.zeros((XformMat.shape[0],inputImgTensor.shape[1],inputImgTensor.shape[1]))

    """
    Restructuring of the data such that it is similar to input data. Currently done manually due to annoyance.
    """
    for x in range(sampledInput.shape[0]):
        toXformThis = sampledInput[x,:,:,:]
        a=np.hstack(toXformThis[0:6,:,:])
        b=np.hstack(toXformThis[6:12, :, :])
        c=np.hstack(toXformThis[12:18 :, :])
        d=np.hstack(toXformThis[18:24, :, :])
        e= np.hstack(toXformThis[24:30, :, :])
        f = np.hstack(toXformThis[30:36, :, :])

        imgnew = np.vstack((a, b, c, d, e, f))
        sampledHeatmaps[x,:,:] =imgnew


    return sampledHeatmaps




#inputimg = torch.rand(2,60,60)
#ImageSampler_SingleTask(inputImgTensor =inputimg)
