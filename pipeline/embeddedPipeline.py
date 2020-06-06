import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


def ihcFormatter(imgIn, mask):
    ihcImgMasked = imgIn * mask[:, :, None]
    return ihcImgMasked


def subsamplerHE(path, imgIn, imgLabel, imgNum, dim, step):
    for i in range(0, len(imgIn) - dim - 1, step):
        for j in range(0, len(imgIn[i]) - dim - 1, step):
            subImg = imgIn[i:i + dim, j:j + dim]
            subImgLabel = imgLabel[i:i + dim, j:j + dim]
            plt.imsave(path + "embedded/Vis/HE_i" + str(imgNum) + ".png",
                       layerImages(subImg, subImgLabel))
            imgNum += 1
    return imgNum


def subsamplerIHC(path, imgIn, imgLabel, imgNum, dim, step):
    for i in range(0, len(imgIn) - dim - 1, step):
        for j in range(0, len(imgIn[i]) - dim - 1, step):
            subImg = imgIn[i:i + dim, j:j + dim]
            blackPix = np.count_nonzero(np.all(subImg == [0, 0, 0], axis=2))
            if blackPix == 0:
                subImgLabel = imgLabel[i:i + dim, j:j + dim]
                plt.imsave(path + "embedded/IHC_Ki67_i" + str(imgNum) + ".png",
                           layerImages(subImg/255, subImgLabel))
                imgNum += 1
    return imgNum


def layerImages(img1, img2):
    stackedImg = np.asarray(np.dstack((img1, img2)))
    return stackedImg


def HEcrawler(path, targetPath, dim, step, zoom):
    assert zoom > 0, "zoom must be greater than zero"
    imageCounter = 0
    for subdir, dirs, files in os.walk(path):
        if len(files) != 2:
            print("This folder " + subdir + " does not contain only 2 images", files)
            continue
        image = plt.imread(subdir + "/" + files[0])
        mask = (plt.imread(subdir + "/" + files[1]) > 1).astype(float)
        image = resize(image, (image.shape[0]//zoom, image.shape[1]//zoom, image.shape[2]))
        mask = resize(mask, (mask.shape[0]//zoom, mask.shape[1]//zoom))
        imageCounter = subsamplerHE(targetPath, image, mask, imageCounter, dim, step)
        print("Now there are " + str(imageCounter) + " HE images")
    print()


def IHCcrawler(path, targetPath, dim, step):
    imageCounter = 0
    for subdir, dirs, files in os.walk(path):
        imgAmount = int(len(files) / 3)
        if imgAmount <= 0:
            continue
        for i in range(imgAmount):
            files = sorted(files)
            image = plt.imread(subdir + "/" + files[i + imgAmount])
            segmentedAreaMask = plt.imread(subdir + "/" + files[i + imgAmount * 2])
            mask = (plt.imread(subdir + "/" + files[i]) > 0).astype(float)
            imageSegmented = ihcFormatter(image, segmentedAreaMask)
            imageCounter = subsamplerIHC(targetPath, imageSegmented, mask, imageCounter, dim, step)

        print("Now there are " + str(imageCounter) + " IHC_Ki69 images")
    return imageCounter


def hamamatsuCrawler(imgCount, path, targetPath, dim, step, zoom=2):
    assert zoom > 0, "zoom must be greater than zero"
    imageCounter = imgCount
    for subdir, dirs, files in os.walk(path):
        if len(files) < 1:
            continue
        offset = len(files)//2
        files = sorted(files)
        for i in range(0, 1000):
            image = plt.imread(subdir + "/" + files[i+offset])
            image = resize(image, (image.shape[0]//zoom, image.shape[1]//zoom, image.shape[2]))
            mask = plt.imread(subdir + "/" + files[i])
            mask = resize(mask, (mask.shape[0]//zoom, mask.shape[1]//zoom))
            plt.imsave(targetPath + "embedded/IHC_Ki67_i" + str(imageCounter) + ".png", layerImages(image, mask*255))
            imageCounter += 1
        print("Now there are " + str(imageCounter) + " IHC_Ki69 images")
    print()
    return imageCounter


def PanNukeCrawler(imgCount, imagePath, maskPath, imageType, targetPath, maskIndex=5, zoom=2):
    assert zoom > 0, "zoom must be greater than zero"
    imageCounter = imgCount
    imgData = np.load(imagePath+"images.npy")
    typeData = np.load(imagePath+"types.npy")
    mskData = np.load(maskPath+"masks.npy")
    for i in range(len(typeData)):
        if imageType == typeData[i]:
            image = imgData[i]/255
            mask = mskData[i,:,:,maskIndex].astype(int)
            if maskIndex == 5:
                mask = (mask == 0).astype(float)
            image = resize(image, (image.shape[0]//zoom, image.shape[1]//zoom, image.shape[2]))
            mask = (resize(mask, (mask.shape[0]//zoom, mask.shape[1]//zoom), preserve_range=True, anti_aliasing=False) >0).astype(float)
            #plt.imsave(targetPath + "embedded/PanNuke/HE_i" + str(imageCounter) + ".png",
            plt.imsave(targetPath + "imgZero/HE_i" + str(imageCounter) + ".png",
                       layerImages(image, mask))
            imageCounter +=1
            if imageCounter > 1000:
                break
    print("Now there are " + str(imageCounter) + " PanNuke HE images")
    print()
    return imageCounter


def crawler(path, targetPath, dim, step, zoom=1):
#    HEcrawler(path + "HE_merged/", targetPath + "HE/", dim, step, zoom)
    imgCount = 0
    for indx in range(1,4):
        imgCount = PanNukeCrawler(imgCount, path+"PanNuke/fold" + str(indx)+ "/images/fold" + str(indx)+ "/", path+"PanNuke/fold" + str(indx)+ "/masks/fold" + str(indx)+ "/", "Breast", targetPath+"HE/")
    imgCount = IHCcrawler(path + "IHC_Ki67/", targetPath + "IHC_Ki67/", dim, step)
#    imgCount = hamamatsuCrawler(imgCount, path + "Hamamatsu1/", targetPath + "IHC_Ki67/", dim, step)
    


dimension = 128
scaleDif = 2
crawler(path = "./../data/", targetPath = "./../processedData/", dim=dimension, step=dimension, zoom=scaleDif)
