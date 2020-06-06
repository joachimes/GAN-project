import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


divideData = 0.2


def ihcFormatter(imgIn, mask):
    ihcImgMasked = imgIn * mask[:, :, None]
    return ihcImgMasked


def subsamplerHE(path, imgIn, imgLabel, imgNum, dim, step):
    for i in range(0, len(imgIn) - dim - 1, step):
        for j in range(0, len(imgIn[i]) - dim - 1, step):
            subImg = imgIn[i:i + dim, j:j + dim]
            subImgLabel = imgLabel[i:i + dim, j:j + dim]
            plt.imsave(path + "img/HE_i" + str(imgNum) + ".jpg", subImg)
            plt.imsave(path + "img_mask/HE_i" + str(imgNum) + ".png", subImgLabel)
            imgNum += 1
    return imgNum


def subsamplerIHC(path, imgIn, imgNum, dim, step):
    for i in range(0, len(imgIn) - dim - 1, step):
        for j in range(0, len(imgIn[i]) - dim - 1, step):
            subImg = imgIn[i:i + dim, j:j + dim]
            blackPix = np.count_nonzero(np.all(subImg == [0, 0, 0], axis=2))
            if blackPix == 0:
                plt.imsave(path + "img/IHC_Ki67_i" + str(imgNum) + ".jpg", subImg)
                imgNum += 1
    return imgNum


def HEcrawler(path, targetPath, dim, step, zoom=1):
    assert zoom > 0, "zoom must be greater than zero"
    imageCounter = 0
    for subdir, dirs, files in os.walk(path):
        if len(files) != 2:
            print("This folder " + subdir + " does not contain only 2 images", files)
            continue
        image = plt.imread(subdir+"/"+files[0])
        mask = (plt.imread(subdir+"/"+files[1]) > 1).astype(int)
        image = resize(image, (image.shape[0]//zoom, image.shape[1]//zoom, image.shape[2]))
        mask = resize(mask, (mask.shape[0]//zoom, mask.shape[1]//zoom))
        imageCounter = subsamplerHE(targetPath, image, mask, imageCounter, dim, step)
        print("Now there are " + str(imageCounter) + " HE images")
    print()


def IHCcrawler(path, targetPath, dim, step, zoom=1):
    assert zoom > 0, "zoom must be greater than zero"
    imageCounter = 0
    for subdir, dirs, files in os.walk(path):
        imgAmount = int(len(files) / 3)
        if imgAmount <= 0:
            continue
        for i in range(imgAmount):
            files = sorted(files)
            image = plt.imread(subdir + "/" + files[i + imgAmount])
            segmentedAreaMask = plt.imread(subdir + "/" + files[i + imgAmount * 2])
            imageSegmented = ihcFormatter(image, segmentedAreaMask)
            imageCounter = subsamplerIHC(targetPath, imageSegmented, imageCounter, dim, step)

        print("Now there are " + str(imageCounter) + " IHC_Ki69 images")
    print()


#def crawler(path, targetPath, dim, step, zoom=1):
#    HEcrawler(path + "HE_merged/", targetPath + "HE/", dim, step, zoom)
#    IHCcrawler(path + "IHC_Ki67/", targetPath + "IHC_Ki67/", dim, step)
def PanNukeCrawler(imgCount, imagePath, maskPath, imageType, targetPath, maskIndex=5, zoom=2):
    assert zoom > 0, "zoom must be greater than zero"
    imageCounter = imgCount
    imgData = np.load(imagePath+"images.npy")
    typeData = np.load(imagePath+"types.npy")
    for i in range(len(typeData)):
        if imageType == typeData[i]:
            image = imgData[i]/255
            image = resize(image, (image.shape[0]//zoom, image.shape[1]//zoom, image.shape[2]))
            plt.imsave(targetPath + "img/HE_i" + str(imageCounter) + ".jpg", image)
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
        if imgCount > 1000:
            break
        imgCount = PanNukeCrawler(imgCount, path+"PanNuke/fold" + str(indx)+ "/images/fold" + str(indx)+ "/", path+"PanNuke/fold" + str(indx)+ "/masks/fold" + str(indx)+ "/", "Breast", targetPath+"HE/")
    imgCount = IHCcrawler(path + "IHC_Ki67/", targetPath + "IHC_Ki67/", dim, step)
#    imgCount = hamamatsuCrawler(imgCount, path + "Hamamatsu1/", targetPath + "IHC_Ki67/", dim, step)



dimension = 128
scaleDif = 2
crawler(path="./../data/", targetPath="./../processedData/", dim=dimension, step=dimension, zoom=scaleDif)
