import torch.nn as nn
import torch
from torch import autograd

class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        #fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class myChannelUnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(myChannelUnet, self).__init__()
        filter = [64,128,256,512,1024]
        self.conv1 = DoubleConv(in_ch, filter[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(filter[0], filter[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(filter[1], filter[2])
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(filter[2], filter[3])
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(filter[3], filter[4])
        self.up6 = nn.ConvTranspose2d(filter[4], filter[3], 2, stride=2)
        self.conv6 = DoubleConv(filter[3]*3,filter[3])
        self.up7 = nn.ConvTranspose2d(filter[3], filter[2], 2, stride=2)
        self.conv7 = DoubleConv(filter[2]*3, filter[2])
        self.up8 = nn.ConvTranspose2d(filter[2], filter[1], 2, stride=2)
        self.conv8 = DoubleConv(filter[1]*3, filter[1])
        self.up9 = nn.ConvTranspose2d(filter[1], filter[0], 2, stride=2)
        self.conv9 = DoubleConv(filter[0]*3, filter[0])
        self.conv10 = nn.Conv2d(filter[0], out_ch, 1)

        self.gau_1 = GAU(filter[4],filter[3])
        self.gau_2 = GAU(filter[3],filter[2])
        self.gau_3 = GAU(filter[2],filter[1])
        self.gau_4 = GAU(filter[1],filter[0])
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        #print(c5.shape)
        up_6 = self.up6(c5)

        gau1 = self.gau_1(c5,c4)
        # print(c4.shape)
        # print(up_6.shape)
        # print(gau1.shape)
        merge6 = torch.cat([c4,up_6, gau1], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        gau2 = self.gau_2(gau1,c3)
        merge7 = torch.cat([c3,up_7, gau2], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        gau3 = self.gau_3(gau2,c2)
        merge8 = torch.cat([c2,up_8, gau3], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        gau4 = self.gau_4(gau3,c1)
        merge9 = torch.cat([c1,up_9, gau4], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out