from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

__all__ = ['Xception', 'Xception_salience', 'Xception_parsing']

class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution (bias discarded) + batch normalization + relu6.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
        g (int): number of blocked connections from input channels
                 to output channels (default: 1).
    """
    def __init__(self, in_c, out_c, k, s=1, p=0, g=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False, groups=g)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu6(self.bn(self.conv(x)))

class SepConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SepConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))

class EntryFLow(nn.Module):
    def __init__(self, nchannels):
        super(EntryFLow, self).__init__()
        self.conv1 = ConvBlock(3, nchannels[0], 3, s=2, p=1)
        self.conv2 = ConvBlock(nchannels[0], nchannels[1], 3, p=1)
        
        self.conv3 = nn.Sequential(
            SepConv(nchannels[1], nchannels[2]),
            nn.ReLU(),
            SepConv(nchannels[2], nchannels[2]),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.conv3s = nn.Sequential(
            nn.Conv2d(nchannels[1], nchannels[2], 1, stride=2, bias=False),
            nn.BatchNorm2d(nchannels[2]),
        )
        
        self.conv4 = nn.Sequential(
            nn.ReLU(),
            SepConv(nchannels[2], nchannels[3]),
            nn.ReLU(),
            SepConv(nchannels[3], nchannels[3]),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.conv4s = nn.Sequential(
            nn.Conv2d(nchannels[2], nchannels[3], 1, stride=2, bias=False),
            nn.BatchNorm2d(nchannels[3])
        )
        
        self.conv5 = nn.Sequential(
            nn.ReLU(),
            SepConv(nchannels[3], nchannels[4]),
            nn.ReLU(),
            SepConv(nchannels[4], nchannels[4]),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.conv5s = nn.Sequential(
            nn.Conv2d(nchannels[3], nchannels[4], 1, stride=2, bias=False),
            nn.BatchNorm2d(nchannels[4]),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        
        x3 = self.conv3(x2)
        x3s = self.conv3s(x2)
        x3 = x3 + x3s
        
        x4 = self.conv4(x3)
        x4s = self.conv4s(x3)
        x4 = x4 + x4s

        x5 = self.conv5(x4)
        x5s = self.conv5s(x4)
        x5 = x5 + x5s

        return x5

class MidFlowBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MidFlowBlock, self).__init__()
        self.conv1 = SepConv(in_channels, out_channels)
        self.conv2 = SepConv(out_channels, out_channels)
        self.conv3 = SepConv(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        return x

class MidFlow(nn.Module):
    def __init__(self, in_channels, out_channels, num_repeats):
        super(MidFlow, self).__init__()
        self.blocks = self._make_layer(in_channels, out_channels, num_repeats)

    def _make_layer(self, in_channels, out_channels, num):
        layers = []
        for i in range(num):
            layers.append(MidFlowBlock(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)

class ExitFlow(nn.Module):
    def __init__(self, in_channels, nchannels):
        super(ExitFlow, self).__init__()
        self.conv1 = SepConv(in_channels, nchannels[0])
        self.conv2 = SepConv(nchannels[0], nchannels[1])
        self.conv2s = nn.Sequential(
            nn.Conv2d(in_channels, nchannels[1], 1, stride=2, bias=False),
            nn.BatchNorm2d(nchannels[1]),
        )

        self.conv3 = SepConv(nchannels[1], nchannels[2])
        self.conv4 = SepConv(nchannels[2], nchannels[3])

    def forward(self, x):
        x1 = self.conv1(F.relu(x))
        x2 = self.conv2(F.relu(x1))
        x2 = F.max_pool2d(x2, 3, stride=2, padding=1)
        x2s = self.conv2s(x)
        x2 = x2 + x2s
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x4 = F.avg_pool2d(x4, x4.size()[2:]).view(x4.size(0), -1)
        return x4

class Xception(nn.Module):
    """Xception

    Reference:
    Chollet. Xception: Deep Learning with Depthwise Separable Convolutions. CVPR 2017.

    Code imported from https://github.com/KaiyangZhou/deep-person-reid
    
    """
    def __init__(self, num_classes, loss={'xent'}, num_mid_flows=8, **kwargs):
        super(Xception, self).__init__()
        self.loss = loss

        self.entryflow = EntryFLow(nchannels=[32, 64, 128, 256, 728])
        self.midflow = MidFlow(728, 728, 8)
        self.exitflow = ExitFlow(728, nchannels=[728, 1024, 1536, 2048])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048
        self.use_salience = False
        self.use_parsing = False

    def forward(self, x):
        x = self.entryflow(x)
        x = self.midflow(x)
        x = self.exitflow(x)
        
        if not self.training:
            return x

        y = self.classifier(x)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, x
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class Xception_salience(nn.Module):
    """Xception

    Reference:
    Chollet. Xception: Deep Learning with Depthwise Separable Convolutions. CVPR 2017.
    """
    def __init__(self, num_classes, loss={'xent'}, num_mid_flows=8, **kwargs):
        super(Xception_salience, self).__init__()
        self.loss = loss

        self.entryflow = EntryFLow(nchannels=[32, 64, 128, 256, 728])
        self.midflow = MidFlow(728, 728, 8)
        self.exitflow = ExitFlow(728, nchannels=[728, 1024, 1536, 2048])
        self.classifier = nn.Linear(2776, num_classes)
        self.feat_dim = 2776
        self.use_salience = True
        self.use_parsing = False

    def forward(self, x, salience_masks):
        x = self.entryflow(x)
        xm = self.midflow(x)
        x = self.exitflow(xm)

        salience_feat = F.upsample(xm, size = (salience_masks.size()[-2], salience_masks.size()[-1]), mode = 'bilinear')
        channel_size = salience_feat.size()[2] * salience_feat.size()[3]
        salience_masks = salience_masks.cuda()
        salience_masks = salience_masks.view(salience_masks.size()[0], channel_size, 1)
        salience_feat = salience_feat.view(salience_feat.size()[0], salience_feat.size()[1], channel_size)
        salience_feat = torch.bmm(salience_feat, salience_masks)#instead of replicating we use matrix product

        #average pooling
        salience_feat = salience_feat.view(salience_feat.size()[:2]) / float(channel_size)
        x = torch.cat((x, salience_feat), dim = 1)
        if not self.training:
            return x

        y = self.classifier(x)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, x
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class Xception_parsing(nn.Module):
    """Xception

    Reference:
    Chollet. Xception: Deep Learning with Depthwise Separable Convolutions. CVPR 2017.
    """
    def __init__(self, num_classes, loss={'xent'}, num_mid_flows=8, **kwargs):
        super(Xception_parsing, self).__init__()
        self.loss = loss

        self.entryflow = EntryFLow(nchannels=[32, 64, 128, 256, 728])
        self.midflow = MidFlow(728, 728, 8)
        self.exitflow = ExitFlow(728, nchannels=[728, 1024, 1536, 2048])
        self.classifier = nn.Linear(5688, num_classes)
        self.feat_dim = 5688
        self.use_salience = False
        self.use_parsing = True

    def forward(self, x, parsing_masks):
        x = self.entryflow(x)
        xm = self.midflow(x)
        x = self.exitflow(xm)

        parsing_feat = F.upsample(xm, size = (parsing_masks.size()[-2], parsing_masks.size()[-1]), mode = 'bilinear')
        channel_size = parsing_feat.size()[2] * parsing_feat.size()[3]
        parsing_masks = parsing_masks.view(parsing_masks.size()[0], parsing_masks.size()[1], channel_size)
        parsing_masks = torch.transpose(parsing_masks, 1, 2)
        parsing_feat = parsing_feat.view(parsing_feat.size()[0], parsing_feat.size()[1], channel_size)
        parsing_feat = torch.bmm(parsing_feat, parsing_masks)
        #average pooling
        parsing_feat = parsing_feat.view(parsing_feat.size()[0], parsing_feat.size()[1] * parsing_feat.size()[2]).cuda() / float(channel_size)

        x = torch.cat((x, parsing_feat), dim = 1)
        if not self.training:
            return x

        y = self.classifier(x)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, x
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))