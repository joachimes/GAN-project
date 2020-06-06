import sys
sys.path.insert(1,'../cycleGAN/')

import torch
import torch.nn as nn
import numpy as np
import torchvision

from model import Generator as genera
from torch.utils.data import DataLoader, random_split
from dataloaderImg import *

from torch.autograd import Variable
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

import os
import matplotlib.pyplot as plt
from skimage.transform import resize


testSplit = .1
shuffle = True
batchSize = 32
nCPU = 4
imgShape = 128
imgChannels = 4
numResBlocks = 6
inputShape = (imgChannels, imgShape, imgShape)
modelPath = "../cycleGAN/"
epochEnded = 199

G_AB = nn.DataParallel(genera(inputShape, numResBlocks)).cuda()
G_BA = nn.DataParallel(genera(inputShape, numResBlocks)).cuda()
#G_AB = genera(inputShape, numResBlocks).cuda()
#G_BA = genera(inputShape, numResBlocks).cuda()

G_AB.load_state_dict(torch.load(modelPath + "saved_models/G_AB_" + str(epochEnded), map_location=torch.device('cuda')))
G_BA.load_state_dict(torch.load(modelPath + "saved_models/G_BA_" + str(epochEnded), map_location=torch.device('cuda')))

### Load data
transforms_ = [
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
]

dataset = ImageDataset('../processedData/HE',
                       '../processedData/IHC_Ki67',
                       transforms_=transforms_,
                       mode="embedded", exts='png', panNuke=True)
randomSeed = 42

if randomSeed is not None:
    np.random.seed(randomSeed)


datasetSize = len(dataset)

testSize = int(testSplit * datasetSize) #test size length
trainSize = datasetSize - testSize      #train data length


trainDataset, testDataset = random_split(dataset, [trainSize, testSize])

testLoader = DataLoader(testDataset, shuffle=shuffle,
                         batch_size=batchSize, num_workers=nCPU)


def convert_from_tanh(input_tensor):
    out_tensor = torch.add(input_tensor, 1)
    out_tensor = torch.mul(out_tensor, 255/2)
    return out_tensor


def IHCSampled(path, targetPath, dim, step):
    imageCounter = 0
    print("going in IHC", path)
    for subdir, dirs, files in os.walk(path):
        print(len(files))
        for i in range(0, 400):
            if len(files) < 10:
                continue
            if imageCounter >= 300:
                break
            image = plt.imread(subdir + "/" + files[i])
            plt.imsave(targetPath + "IHC_Ki67_i" + str(imageCounter) + ".png", image[:,:,:3])
            imageCounter += 1
        print("Now there are " + str(imageCounter) + " IHC_Ki69 images")
    print()
    return imageCounter


def PanNukeCrawler(imagePath, imageType, targetPath, zoom=2):
    assert zoom > 0, "zoom must be greater than zero"
    imageCounter = 0
    imgData = np.load(imagePath+"images.npy")
    typeData = np.load(imagePath+"types.npy")
    for i in range(len(typeData)):
        if imageCounter >= 300:
            break
        if imageType == typeData[i]:
            image = imgData[i]/255
            image = resize(image, (image.shape[0]//zoom, image.shape[1]//zoom, image.shape[2]))
            plt.imsave(targetPath + "HE_i" + str(imageCounter) + ".png", image)
            imageCounter +=1
    print("Now there are " + str(imageCounter) + " PanNuke HE images")
    print()
    return imageCounter

def sample_images(startHE, startIHC, path):
    imgNumHE = startHE
    imgNumIHC = startIHC
    for i in range(6):
        G_AB.eval()
        G_BA.eval()
        imgs = next(iter(testLoader))
        print(imgs["A"].shape)
        with torch.no_grad():
            real_A = Variable(imgs["A"].type(Tensor))
            real_B = Variable(imgs["B"].type(Tensor))
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            for j in range(batchSize):
                imgNumHE  += 1
                imgNumIHC += 1
                fake_A_img = np.uint8(convert_from_tanh(fake_A[j].permute(1,2,0)).cpu().numpy()[:,:,:3])
                fake_B_img = np.uint8(convert_from_tanh(fake_B[j].permute(1,2,0)).cpu().numpy()[:,:,:3])
                plt.imsave(path + "HE/HE_i" + str(imgNumHE) + ".png", fake_A_img)
                plt.imsave(path + "IHC/IHC_Ki67_i" + str(imgNumIHC) + ".png", fake_B_img)
        print("Now ther are " + str(imgNumHE) + " HE images and " + str(imgNumIHC) + " IHC images")

def crawler(path, targetPath, dim, step, zoom=1):
    HEnum = 300# PanNukeCrawler(path+"PanNuke/fold1/images/fold1/", "Breast", targetPath+"gw/HE/")
    IHCnum = 300#IHCSampled(targetPath + "IHC_Ki67/embedded/", targetPath + "gw/IHC/", dim, step)
    sample_images(HEnum, IHCnum, targetPath + "gw/")
    

dimension = 128
scaleDif = 2
crawler(path="./../data/", targetPath="./../processedData/", dim=dimension, step=dimension, zoom=scaleDif)

