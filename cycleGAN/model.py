import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_channels, out, normalization=True):
            layers = [nn.Conv2d(in_channels, out, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers += [nn.InstanceNorm2d(out)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            return layers

        self.model = nn.Sequential(*discriminator_block(channels, 64,
                                                        normalization=False),
                                   *discriminator_block(64, 128),
                                   *discriminator_block(128, 256),
                                   *discriminator_block(256, 512),
                                   nn.ZeroPad2d((1, 0, 1, 0)),
                                   nn.Conv2d(512, 1, 4, padding=1, bias=False)
                                   )

    def forward(self, img):
        return self.model(img)


class ResidualBlock(nn.Module):
    def __init__(self, in_feat):
        super(ResidualBlock, self).__init__()

        self.resBlock = nn.Sequential(nn.ReflectionPad2d(1),
                                      nn.Conv2d(in_feat, in_feat, 3),
                                      nn.ReLU(inplace=True),
                                      nn.ReflectionPad2d(1),
                                      nn.Conv2d(in_feat, in_feat, 3),
                                      nn.InstanceNorm2d(in_feat)
                                      )

    def forward(self, x):
        return x + self.resBlock(x)


class Generator(nn.Module):
    def __init__(self, in_shape, num_residual_blocks):
        super(Generator, self).__init__()

        channels = in_shape[0]

        # c7s1-64
        out_feat = 64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_feat, 7),
            nn.InstanceNorm2d(out_feat),
            nn.ReLU(inplace=True)
        ]
        in_feat = out_feat

        # d128,d256
        for _ in range(2):
            out_feat *= 2
            model += [
                nn.Conv2d(in_feat, out_feat, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_feat),
                nn.ReLU(inplace=True)
            ]
            in_feat = out_feat

        # R256
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_feat)]

        # u128, u64
        for _ in range(2):
            out_feat //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_feat, out_feat, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_feat),
                nn.ReLU(inplace=True)
            ]
            in_feat = out_feat

        # c7s1-3
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(out_feat, channels, 7),
                  nn.Tanh()
                  ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
