import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import itertools
import datetime
import time

from dataloaderImg import *
from model import *
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image, make_grid

from torch.autograd import Variable
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

### Hyperparameters

num_epochs = 200
batchSize = 32
testSplit = .2
shuffle = True
imgShape = 128
maskChannels = 1
# Embedded
imgChannels = 3 + maskChannels
nCPU = 8 
lr = 2e-4
lr_decay_point = 100
lambda_cyc = 5.0  # Cycle loss weight
lambda_id = 5.0   # identity loss weight
sample_interval = 100
checkpoint_interval = 199
numResBlocks = 6
inputShape = (imgChannels, imgShape, imgShape)

### Load data
transforms_ = [
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
]

dataset = ImageDataset('../processedData/HE',
                       '../processedData/IHC_Ki67',
                       transforms_=transforms_,
                       mode="embedded", exts='png', panNuke=True) # Embedded
randomSeed = 42

if randomSeed is not None:
    np.random.seed(randomSeed)


datasetSize = len(dataset)
print(datasetSize)
testSize = int(testSplit * datasetSize)  # test size length
trainSize = datasetSize - testSize       # train data length

trainDataset, testDataset = random_split(dataset, [trainSize, testSize])

trainLoader = DataLoader(trainDataset, shuffle=shuffle,
                          batch_size=batchSize, num_workers=nCPU)
testLoader = DataLoader(testDataset, shuffle=shuffle,
                         batch_size=batchSize, num_workers=nCPU)

### Utility functions
class imgBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def buffer(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


lr_sched = lambda current_epoch: 1.0 - max(0, current_epoch - lr_decay_point) / float(num_epochs - lr_decay_point)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(testLoader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B)
    # Arrange x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arrange y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s.png" % batches_done, normalize=False)


### Setup model
loss_GAN = nn.MSELoss()
loss_cycle = nn.L1Loss()
loss_identity = nn.L1Loss()
print("Using device:", device)

G_AB = nn.DataParallel(Generator(inputShape, numResBlocks)).to(device)
G_BA = nn.DataParallel(Generator(inputShape, numResBlocks)).to(device)

D_A = nn.DataParallel(Discriminator(inputShape)).to(device)
D_B = nn.DataParallel(Discriminator(inputShape)).to(device)

G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

generatorOptim = torch.optim.Adam(itertools.chain(G_AB.parameters(),
                                                  G_BA.parameters()),
                                  lr, betas=(0.5, 0.999))
d_A_Optim = torch.optim.Adam(D_A.parameters(), lr, betas=(0.5, 0.999))
d_B_Optim = torch.optim.Adam(D_B.parameters(), lr, betas=(0.5, 0.999))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(generatorOptim, lr_lambda=lr_sched)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    d_A_Optim, lr_lambda=lr_sched)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    d_B_Optim, lr_lambda=lr_sched)

fake_A_buffer = imgBuffer()
fake_B_buffer = imgBuffer()


### Training
prev_time = time.time()
for epoch in range(num_epochs):
    print(trainLoader.__len__())
    for i, batch in enumerate(trainLoader):
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        # Adversarial ground truths
        valid_tensor = Variable(Tensor(np.ones((real_A.size(0), *D_A.module.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.module.output_shape))), requires_grad=False)

        ###### Generator
        G_AB.train()
        G_BA.train()

        generatorOptim.zero_grad()

        # Generator loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = loss_GAN(D_B(fake_B), valid_tensor)
        fake_A = G_BA(real_B)
        loss_GAN_BA = loss_GAN(D_A(fake_A), valid_tensor)

        loss_GAN_total = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = loss_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = loss_cycle(recov_B, real_B)

        loss_cycle_res = (loss_cycle_A + loss_cycle_B) / 2

        # total loss
        loss_G = loss_GAN_total + lambda_cyc * loss_cycle_res

        loss_G.backward()
        generatorOptim.step()

        ###### Discriminator A

        d_A_Optim.zero_grad()

        loss_real = loss_GAN(D_A(real_A), valid_tensor)
        fake_A_list = fake_A_buffer.buffer(fake_A)
        loss_fake = loss_GAN(D_A(fake_A.detach()), fake)
        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()
        d_A_Optim.step()

        ###### Discriminator B

        d_B_Optim.zero_grad()

        loss_real = loss_GAN(D_B(real_B), valid_tensor)
        fake_B_list = fake_B_buffer.buffer(fake_B)
        loss_fake = loss_GAN(D_A(fake_B.detach()), fake)
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B.backward()
        d_B_Optim.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        ###### Logging

        batches_done = epoch * len(trainLoader) + i
        batches_left = num_epochs * len(trainLoader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        print(
            "train, [it: {}/{}] [D loss: {:.2f}] [G loss: {:.2f}, adv: {:.2f}, cycle: {:.2f}, identity: {:.2f}] [ETL: {}]"
                .format(
                    epoch,
                    num_epochs,
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN_total,
                    loss_cycle_res.item(),
                    0, # loss_identity_val.item(),
                    time_left
                ))

        # saving sampled images
        if batches_done % sample_interval == 0:
            sample_images(batches_done)

    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        print("Saving model")
        torch.save(G_AB.state_dict(), "saved_models/G_AB_{}".format(epoch))
        torch.save(G_BA.state_dict(), "saved_models/G_BA_{}".format(epoch))
        torch.save(D_A.state_dict(), "saved_models/D_A_{}".format(epoch))
        torch.save(D_B.state_dict(), "saved_models/D_B_{}".format(epoch))
