import sys
sys.path.insert(1,'../cycleGAN/')

import torch
import torch.nn as nn
import numpy as np
import torchvision

from PIL import Image
from model import Generator as gengen
from torch.utils.data import DataLoader, random_split
from dataloaderImg import *

from torch.autograd import Variable
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import io

nCPU = 4
imgShape = 128
imgChannels = 4
numResBlocks = 6
inputShape = (imgChannels, imgShape, imgShape)
modelPath = "../cycleGAN/"
epochEnded = 199
modelType = "goodGen"
#G_AB = nn.DataParallel(gengen(inputShape, numResBlocks)).cuda()
#G_BA = nn.DataParallel(gengen(inputShape, numResBlocks)).cuda()
G_AB = gengen(inputShape, numResBlocks).cuda()
G_BA = gengen(inputShape, numResBlocks).cuda()

G_AB.load_state_dict(torch.load(modelPath + "saved_models/" + modelType + "/G_AB_" + str(epochEnded), map_location=torch.device('cuda')))
G_BA.load_state_dict(torch.load(modelPath + "saved_models/" + modelType + "/G_BA_" + str(epochEnded), map_location=torch.device('cuda')))


### Load data
transforms_ = [
    #transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    #transforms.RandomCrop((opt.img_height, opt.img_width)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),

]


preprocess = torchvision.transforms.Compose(transforms_)


def convert_from_tanh(input_tensor):
    out_tensor = torch.add(input_tensor, 1)
    out_tensor = torch.mul(out_tensor, 255/2)
    return out_tensor
    return torch.mul(out_tensor, 255)


def IHCSampled(path, targetPath):
    imageCounter = 0
    mse_value = 0
    mse_value_zero = 0
    extraImgs = False
    print("going in IHC", path)
    for subdir, dirs, files in os.walk(path):
        print(len(files))
        for i in range(0, 400):
            if len(files) < 10:
                continue
            if imageCounter >= 300:
                break
            image = plt.imread(subdir + "/" + files[i])
            res = sample_image(G_BA, image, imageCounter, targetPath, files[i], extraImgs, extraImgs)
            mse_value += res[0]
            mse_value_zero += res[1]
            imageCounter += 1
            if i == 30:
                extraImgs = False
#                cycleThrough(targetPath, files[i], image, G_BA, G_AB)
        print("Now there are " + str(imageCounter) + " IHC_Ki69 images")
    print()
    print(mse_value/300)
    return (mse_value/300, mse_value_zero/300)

def genBatch(img):
    tensor_img = preprocess(Image.fromarray(np.uint8(img*255)))
    batch = DataLoader([[tensor_img]], batch_size=1)
    real_img = next(iter(batch))
    real_img = Variable(real_img[0].type(Tensor))
    return real_img


def sample_unembed_image(genNet, image, img_count, path, name,versus=False, multi_sample=False):
    print(name)
    real_img = genBatch(image)
    genNet.eval()
    with torch.no_grad():
        if versus:
            fake_img_m = genNet(real_img)
            fake_img = np.uint8(convert_from_tanh(fake_img_m[0].permute(1,2,0)).cpu().numpy()[:,:,:3])
            plt.imsave(path + "versus/" + modelType + '3d' + name[:-4] + "_i" + str(img_count) + ".jpg", fake_img)
            plt.imsave(path + "versus/" + modelType + '3d' + name[:-4] + "_i" + str(img_count) + "real.jpg", image)
    return 1 


def sample_image(genNet, image, img_count, path, name,versus=False, multi_sample=False):
    img = np.zeros(image.shape)
    img[:,:,:3] = image[:,:,:3]
    mask = image[:,:,3].astype(np.uint8)
    threshold = 0.3
    mse_value = 0
    real_img = genBatch(img)
#    plt.imsave(path + "maskMse/" + str(img_count) + name[:-4] + "_i" + str(img_count) + "real.jpg", image[:,:,:3])
    genNet.eval()
    with torch.no_grad():
        fake_img = genNet(real_img)
        fake_A_img = np.uint8(convert_from_tanh(fake_img[0].permute(1,2,0)).cpu().numpy()[:,:,:3])
        fake_A_mask = (convert_from_tanh(fake_img[0].permute(1,2,0)).cpu().numpy()[:,:,3]>threshold).astype(np.uint8)
        print(np.unique(fake_A_mask, return_counts=True), np.unique(mask, return_counts=True))

        mse_value = np.square(np.subtract(mask.astype(np.uint8), fake_A_mask)).mean()
        mse_value_zero = np.square(np.subtract((mask>threshold).astype(np.uint8), (np.zeros((128,128,1))>threshold).astype(np.uint8))).mean()
#        plt.imsave(path + "maskMse/" + str(img_count) + name[:-4] + "_i" + ".jpg", fake_A_img)
#        io.imsave(path + "maskMse/" + str(img_count) + name[:-4] + "_i" + "mask.jpg", mask3d(fake_A_mask))
#        io.imsave(path + "maskMse/" + str(img_count) + name[:-4] + "_i" + "realmask.jpg", mask3d(mask))
        if versus:
            real_img_new = genBatch(image)
            fake_img_m = genNet(real_img_new)
            fake_img = np.uint8(convert_from_tanh(fake_img_m[0].permute(1,2,0)).cpu().numpy()[:,:,:3])
            plt.imsave(path + "versus/" + str(img_count) + modelType + '4d' + name[:-4] + "_i.jpg", fake_img)
        if multi_sample:
            real_img_new = genBatch(image)
            for j in range(10):
                fake_img = genNet(real_img_new)
                fake_A_img = np.uint8(convert_from_tanh(fake_img[0].permute(1,2,0)).cpu().numpy()[:,:,:3])
                plt.imsave(path + "variation/" + str(img_count) + modelType + name[:-4] + "_i_v" + str(j) + ".jpg", fake_A_img)
    return (mse_value, mse_value_zero)


def mask3d(mask):
    res = np.zeros((128,128,3))
    print(mask.shape)
    for i in range(3):
        res[:,:,i] = mask*255
    return res

def cycleThrough(path, name, img, gen1, gen2):
    real_img = genBatch(img)
    gen1.eval()
    gen2.eval()
    threshold = 0.1
    for i in range(60):
        with torch.no_grad():
            fake_img_A = gen1(real_img)
            fake_img_B = gen2(fake_img_A)
            fake_A_img = np.uint8(convert_from_tanh(fake_img_A[0].permute(1,2,0)).cpu().numpy()[:,:,:3])
            fake_A_mask = (convert_from_tanh(fake_img_A[0].permute(1,2,0)).cpu().numpy()[:,:,3]>threshold).astype(int)
            fake_B_img = np.uint8(convert_from_tanh(fake_img_B[0].permute(1,2,0)).cpu().numpy()[:,:,:3])
            fake_B_mask = (convert_from_tanh(fake_img_B[0].permute(1,2,0)).cpu().numpy()[:,:,3]>threshold).astype(int)
            plt.imsave(path + "cycle/" + name[:-4] + "_Ai" + str(i) + ".jpg", fake_A_img)
            io.imsave(path + "cycle/mask" + name[:-4] + "_Ai" + str(i) + ".jpg", mask3d(fake_A_mask))
            plt.imsave(path + "cycle/" + name[:-4] + "_Bi" + str(i) + ".jpg", fake_B_img)
            io.imsave(path + "cycle/mask" + name[:-4] + "_Bi" + str(i) + ".jpg", mask3d(fake_B_mask))
            real_img= fake_img_B

def PanNukeCrawler(path, targetPath):
    imageCounter = 0
    mse_value = 0
    mse_value_zero = 0
    extraImgs = False
    print("going in HE", path)
    for subdir, dirs, files in os.walk(path):
        print(len(files))
        for i in range(0, 400):
            if len(files) < 10:
                continue
            if imageCounter >= 300:
                break
            image = plt.imread(subdir + "/" + files[i])
            res = sample_image(G_AB, image, imageCounter, targetPath, files[i],extraImgs, extraImgs) 
            mse_value += res[0]
            mse_value_zero += res[1]
            if i == 30:
                extraImgs = False
            #    cycleThrough(targetPath, files[i], image, G_AB, G_BA)
            imageCounter +=1
    print("Now there are " + str(imageCounter) + " PanNuke HE images")
    print()
    print(mse_value/300)
    return (mse_value/300, mse_value_zero/300)

def crawler(targetPath):
    HEnum = PanNukeCrawler(targetPath + "HE/embedded/PanNuke/", targetPath+"thesis/")
    IHCnum = IHCSampled(targetPath + "IHC_Ki67/embedded/", targetPath + "thesis/")
    print(HEnum, IHCnum)

crawler(targetPath="./../processedData/")

