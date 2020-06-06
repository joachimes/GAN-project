import sys
sys.path.insert(1,'../cycleGAN/')

import torch
import torch.nn as nn
import numpy as np
import torchvision

from model import Generator as gengen
from torch.utils.data import DataLoader, random_split
from dataloaderImg import *

from torch.autograd import Variable
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

import os
import matplotlib.pyplot as plt
from skimage.transform import resize


testSplit = .2
shuffle = True
batchSize = 32
nCPU = 4
imgShape = 128
imgChannels = 4
numResBlocks = 6
inputShape = (imgChannels, imgShape, imgShape)
modelPath = "../cycleGAN/"
epochEnded = 199 

G_AB = nn.DataParallel(gengen(inputShape, numResBlocks)).cuda()
G_BA = nn.DataParallel(gengen(inputShape, numResBlocks)).cuda()

G_AB = gengen(inputShape, numResBlocks).cuda()
G_BA = gengen(inputShape, numResBlocks).cuda()

G_AB.load_state_dict(torch.load(modelPath + "saved_models/G_AB_" + str(epochEnded), map_location=torch.device('cuda')))
G_BA.load_state_dict(torch.load(modelPath + "saved_models/G_BA_" + str(epochEnded), map_location=torch.device('cuda')))

### Load data
transforms_ = [
    #transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    #transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
]

dataset = ImageDataset('../processedData/HE',
                       '../processedData/IHC_Ki67',
                       transforms_=transforms_,#[lambda x: torchvision.transforms.ToTensor()(x)],
                       mode="embedded", exts='png', panNuke=True)
randomSeed = 42

if randomSeed is not None:
    np.random.seed(randomSeed)


datasetSize = len(dataset)

testSize = int(testSplit * datasetSize) #test size length
valiSize = testSize
trainSize = datasetSize - testSize - valiSize      #train data length


trainDataset, testValiDataset = random_split(dataset, [trainSize, valiSize + testSize])
validationDataset, testDataset = random_split(testValiDataset, [valiSize, testSize])

testLoader = DataLoader(testDataset, shuffle=shuffle,
                         batch_size=batchSize, num_workers=nCPU)#, sampler=test_sampler)

def convert_from_tanh(input_tensor):
    out_tensor = torch.add(input_tensor, 1)
    out_tensor = torch.mul(out_tensor, 255/2)
    return out_tensor
    return torch.mul(out_tensor, 255)



def sample_images(path):
    imgNumHE = 0
    imgNumIHC = 0
    mseAB = 0
    mseBA = 0
    for i in range(2):
        G_AB.eval()
        G_BA.eval()
        imgs = next(iter(testLoader))
        with torch.no_grad():
            real_A = Variable(imgs["A"].type(Tensor))
            real_B = Variable(imgs["B"].type(Tensor))
            fake_A = G_AB(real_A)
            fake_B = G_BA(real_B)
            for j in range(batchSize):
                npRealA = np.uint8(convert_from_tanh(real_A[j].permute(1,2,0)).cpu().numpy())
                npRealB = np.uint8(convert_from_tanh(real_B[j].permute(1,2,0)).cpu().numpy())
                fake_A_img = np.uint8(convert_from_tanh(fake_A[j].permute(1,2,0)).cpu().numpy())
                fake_B_img = np.uint8(convert_from_tanh(fake_B[j].permute(1,2,0)).cpu().numpy())
                mseAB += np.square(np.subtract((npRealA[:,:,3]>0.1).astype(int), (fake_B_img[:,:,3]>0.1).astype(int))).mean()
                mseBA += np.square(np.subtract((npRealB[:,:,3]>0.1).astype(int), (fake_A_img[:,:,3]>0.1).astype(int))).mean()
                plt.imsave(path + "HEIHC/" + str(imgNumHE) + "HE.png", npRealB[:,:,:3])
                plt.imsave(path + "HEIHC/" + str(imgNumIHC) + "IHC_Ki67.png", fake_B_img[:,:,:3])
                plt.imsave(path + "IHCHE/" + str(imgNumHE) + "HE.png", fake_A_img[:,:,:3])
                plt.imsave(path + "IHCHE/" + str(imgNumIHC) + "IHC_Ki67.png", npRealA[:,:,:3])
                imgNumHE  += 1
                imgNumIHC += 1
    print("The mean squared error of the label AB:"+str(mseAB/imgNumHE)+"\nThe mean squared error of the labels BA:"+str(mseBA/imgNumIHC))
    print("Now ther are " + str(imgNumHE) + " HE images and " + str(imgNumIHC) + " IHC images")

def crawler(targetPath,):
    sample_images(targetPath + "gp/")
    

dimension = 128
scaleDif = 2
crawler(targetPath="./../processedData/")

