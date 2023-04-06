import torch.nn as nn
import torch
import torch.nn.functional as F

class DeepSeaUp(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1=self.double_conv(x)
        x2 = self.conv1d(x)
        x=x1+x2
        x=self.bn(x)
        x=self.relu(x)
        return x

class DeepSeaSegmentation(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DeepSeaSegmentation, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.res1=ResBlock(n_channels,64)
        self.down1 = nn.MaxPool2d(2)
        self.res2 = ResBlock(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.res3 = ResBlock(128, 256)
        self.up1 = DeepSeaUp(256, 128)
        self.res4 = ResBlock(384, 128)
        self.up2 = DeepSeaUp(128, 64)
        self.res5 = ResBlock(192, 64)
        self.conv3 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)


    def forward(self, x):
        x1=self.res1(x)
        x2=self.down1(x1)
        x2 = self.res2(x2)
        x3 = self.down2(x2)
        x3 = self.res3(x3)
        x4=self.up1(x3,x2)
        x4 = self.res4(x4)
        x5 = self.up2(x4,x1)
        x6 = self.res5(x5)
        logits=self.conv3(x6)
        edges = self.conv4(x6)
        return logits,edges

class DeepSeaTracker(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DeepSeaTracker, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.res1=ResBlock(n_channels,64)
        self.res2 = ResBlock(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.res3 = ResBlock(128, 128)
        self.down2 = nn.MaxPool2d(2)
        self.res4 = ResBlock(128, 256)
        self.up1 = DeepSeaUp(256, 128)
        self.res5 = ResBlock(384, 128)
        self.up2 = DeepSeaUp(128, 64)
        self.res6 = ResBlock(256, 64)
        self.conv3 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)

    def forward(self, img_prev,img_curr):
        img_prev=self.res1(img_prev)
        img_curr = self.res2(img_curr)
        x1=torch.cat((img_prev, img_curr), 1)
        x2=self.down1(x1)
        x2 = self.res3(x2)
        x3 = self.down2(x2)
        x3 = self.res4(x3)
        x4=self.up1(x3,x2)
        x4 = self.res5(x4)
        x5 = self.up2(x4,x1)
        x6=self.res6(x5)
        logits=self.conv3(x6)
        return logits