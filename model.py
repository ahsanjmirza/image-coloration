import torch
import torch.nn as nn

class DRDB(nn.Module):
    def __init__(self, in_size, channels, dilation_r):
        super(DRDB, self).__init__()
        self.conv0 = nn.Conv2d(
            in_channels=in_size, 
            out_channels=channels, 
            kernel_size=3,
            dilation=dilation_r,
            padding='same'
        )
        self.conv1 = nn.Conv2d(
            in_channels=in_size+channels, 
            out_channels=channels,
            kernel_size=3,
            dilation=dilation_r,
            padding='same'
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_size+(2*channels), 
            out_channels=channels,
            kernel_size=3, 
            dilation=dilation_r,
            padding='same'
        )
        self.conv3 = nn.Conv2d(
            in_channels=in_size+(3*channels), 
            out_channels=channels, 
            kernel_size=3,
            dilation=dilation_r,
            padding='same'
        )
        self.conv4 = nn.Conv2d(
            in_channels=in_size+(4*channels), 
            out_channels=channels, 
            kernel_size=3,
            dilation=dilation_r,
            padding='same'
        )
        
        return

    def forward(self, x):
        y = self.conv0(x)
        x = torch.cat((x, y), 1)
        y = self.conv1(x)
        x = torch.cat((x, y), 1)
        y = self.conv2(x)
        x = torch.cat((x, y), 1)
        y = self.conv3(x)
        x = torch.cat((x, y), 1)
        y = self.conv4(x)
        return y


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding='valid'),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.4, inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding='valid'),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding='valid'),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding='valid'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding='valid'),
            nn.Sigmoid()
        )
        return

    def forward(self, x):
        x = self.net(x)
        y = torch.mean(x, dim=(1, 2, 3))
        return y

class Generator(nn.Module):
    def __init__(self, in_size, channels, out_size):
        super(Generator, self).__init__()
        
        self.drdb1 = DRDB(in_size, channels, 1)
        self.drdb2 = DRDB(in_size, channels, 2)
        self.drdb3 = DRDB(in_size, channels, 3)

        self.convC = DRDB(in_size+(3*channels), out_size, 1)
        return

    def forward(self, x):
        x1 = self.drdb1(x)
        x2 = self.drdb2(x)
        x3 = self.drdb3(x)
        concat = torch.cat((x, x1, x2, x3), 1)
        y = self.convC(concat)
        return torch.clip(y, 0.0, 255.0)




