import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root_A, root_B, transforms_=None, unaligned=False, mode="img", exts="jpg", panNuke=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        panNukePath =""
        if panNuke:
            panNukePath = "/PanNuke"

        self.files_A = sorted(glob.glob(os.path.join(root_A, "%s" % (mode+panNukePath)) + "/*." + exts))
        self.files_B = sorted(glob.glob(os.path.join(root_B, "%s" % mode) + "/*." + exts))
        self.files_Am = sorted(glob.glob(os.path.join(root_A, "mask") + "/*.tif"))
        self.files_Bm = sorted(glob.glob(os.path.join(root_B, "mask") + "/*.tif"))
    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_Am = Image.open(self.files_Am[index % len(self.files_Am)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
            image_Bm = Image.open(self.files_Bm[random.randint(0, len(self.files_Bm) - 1)])
        else:
            theVar = index % len(self.files_B)
            image_B = Image.open(self.files_B[theVar])
            image_Bm = Image.open(self.files_Bm[theVar])

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B, "Am": self.transform(image_Am), "Bm": self.transform(image_Bm)}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
