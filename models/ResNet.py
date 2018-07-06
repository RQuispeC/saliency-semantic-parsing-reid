from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision


__all__ = ['ResNet50', 'ResNet50_salience', 'ResNet50_parsing', 'ResNet50M', 'ResNet50M_salience', 'ResNet50M_parsing']

class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension
        self.use_salience = False
        self.use_parsing = False

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        elif self.loss == {'ring'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50_salience(nn.Module):
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50_salience, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4a = base[6][0]
        self.layers4b = base[6][1]
        self.layers4c = base[6][2]
        self.layers4d = base[6][3]
        self.layers4e = base[6][4]
        self.layers4f = base[6][5]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.classifier = nn.Linear(3072, num_classes)
        self.feat_dim = 3072 # feature dimension
        self.use_salience = True
        self.use_parsing = False

    def forward(self, x, salience_masks):
        '''
        x: batch of input images
        salince_mask: batch of 2D tensor
        parsing_maks: batch of 3D tensor (various parsing masks per image)
        '''
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4a = self.layers4a(x3)
        x4b = self.layers4b(x4a)
        x4c = self.layers4c(x4b)
        x4d = self.layers4d(x4c)
        x4e = self.layers4e(x4d)
        x4f = self.layers4f(x4e)
        x5a = self.layers5a(x4f)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        #upsample feature map to the same size of salience/parsing maps
        salience_feat = F.upsample(x4f, size = (salience_masks.size()[-2], salience_masks.size()[-1]), mode = 'bilinear')

        #reshape tensors such that we can use matrix product as operator and avoid using loops
        channel_size = salience_feat.size()[2] * salience_feat.size()[3]
        salience_masks = salience_masks.cuda()
        salience_masks = salience_masks.view(salience_masks.size()[0], channel_size, 1)
        salience_feat = salience_feat.view(salience_feat.size()[0], salience_feat.size()[1], channel_size)
        salience_feat  = torch.bmm(salience_feat, salience_masks)#instead of replicating we use matrix product
        #average pooling
        salience_feat = salience_feat.view(salience_feat.size()[:2]) / float(channel_size)
        #join features
        combofeat = torch.cat((x5c_feat, salience_feat), dim=1)

        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50_parsing(nn.Module):
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50_parsing, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4a = base[6][0]
        self.layers4b = base[6][1]
        self.layers4c = base[6][2]
        self.layers4d = base[6][3]
        self.layers4e = base[6][4]
        self.layers4f = base[6][5]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.classifier = nn.Linear(7168, num_classes)
        self.feat_dim = 7168 # feature dimension
        self.use_salience = False
        self.use_parsing = True

    def forward(self, x, parsing_masks = None):
        '''
        x: batch of input images
        salince_mask: batch of 2D tensor
        parsing_maks: batch of 3D tensor (various parsing masks per image)
        '''
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4a = self.layers4a(x3)
        x4b = self.layers4b(x4a)
        x4c = self.layers4c(x4b)
        x4d = self.layers4d(x4c)
        x4e = self.layers4e(x4d)
        x4f = self.layers4f(x4e)
        x5a = self.layers5a(x4f)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        #upsample feature map to the same size of salience/parsing maps
        parsing_feat = F.upsample(x4f, size = (parsing_masks.size()[-2], parsing_masks.size()[-1]), mode = 'bilinear')

        channel_size = parsing_feat.size()[2] * parsing_feat.size()[3]
        parsing_masks = parsing_masks.view(parsing_masks.size()[0], parsing_masks.size()[1], channel_size)
        parsing_masks = torch.transpose(parsing_masks, 1, 2)
        parsing_feat = parsing_feat.view(parsing_feat.size()[0], parsing_feat.size()[1], channel_size)
        parsing_feat = torch.bmm(parsing_feat, parsing_masks)
        #average pooling
        parsing_feat = parsing_feat.view(parsing_feat.size()[0], parsing_feat.size()[1] * parsing_feat.size()[2]).cuda() / float(channel_size)
        #join features
        combofeat = torch.cat((x5c_feat, parsing_feat), dim=1)

        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50M(nn.Module):
    """ResNet50 + mid-level features.

    Reference:
    Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for
    Cross-Domain Instance Matching. arXiv:1711.08106.
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50M, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(3072, num_classes)
        self.feat_dim = 3072 # feature dimension
        self.use_salience = False
        self.use_parsing = False

    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))
        
        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        combofeat = torch.cat((x5c_feat, midfeat), dim=1)
        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50M_salience(nn.Module):
    """ResNet50m + mid-level features + weighting of mid-level features and salience maps
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50M_salience, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4a = base[6][0]
        self.layers4b = base[6][1]
        self.layers4c = base[6][2]
        self.layers4d = base[6][3]
        self.layers4e = base[6][4]
        self.layers4f = base[6][5]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(4096, num_classes)
        self.feat_dim = 4096 # feature dimension
        self.use_salience = True
        self.use_parsing = False

    def forward(self, x, salience_masks):
        '''
        x: batch of input images
        salince_mask: batch of 2D tensor
        parsing_maks: batch of 3D tensor (various parsing masks per image)
        '''
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4a = self.layers4a(x3)
        x4b = self.layers4b(x4a)
        x4c = self.layers4c(x4b)
        x4d = self.layers4d(x4c)
        x4e = self.layers4e(x4d)
        x4f = self.layers4f(x4e)
        x5a = self.layers5a(x4f)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))
        #print("4c", x4c.size()) #output [32, 1024, 16, 8]

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        #upsample feature map to the same size of salience/parsing maps
        salience_feat = F.upsample(x4f, size = (salience_masks.size()[-2], salience_masks.size()[-1]), mode = 'bilinear')

        #reshape tensors such that we can use matrix product as operator and avoid using loops
        channel_size = salience_feat.size()[2] * salience_feat.size()[3]
        salience_masks = salience_masks.cuda()
        salience_masks = salience_masks.view(salience_masks.size()[0], channel_size, 1)
        salience_feat = salience_feat.view(salience_feat.size()[0], salience_feat.size()[1], channel_size)
        salience_feat  = torch.bmm(salience_feat, salience_masks)#instead of replicating we use matrix product
        #average pooling
        salience_feat = salience_feat.view(salience_feat.size()[:2]) / float(channel_size)
        #join features
        combofeat = torch.cat((x5c_feat, midfeat, salience_feat), dim=1)

        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50M_parsing(nn.Module):
    """ResNet50 + mid-level features + weighting of mid-level features and semantic parsing maps
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50M_parsing, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4a = base[6][0]
        self.layers4b = base[6][1]
        self.layers4c = base[6][2]
        self.layers4d = base[6][3]
        self.layers4e = base[6][4]
        self.layers4f = base[6][5]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(8192, num_classes)
        self.feat_dim = 8192 # feature dimension
        self.use_salience = False
        self.use_parsing = True

    def forward(self, x, parsing_masks = None):
        '''
        x: batch of input images
        salince_mask: batch of 2D tensor
        parsing_maks: batch of 3D tensor (various parsing masks per image)
        '''
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4a = self.layers4a(x3)
        x4b = self.layers4b(x4a)
        x4c = self.layers4c(x4b)
        x4d = self.layers4d(x4c)
        x4e = self.layers4e(x4d)
        x4f = self.layers4f(x4e)
        x5a = self.layers5a(x4f)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        #upsample feature map to the same size of salience/parsing maps
        parsing_feat = F.upsample(x4f, size = (parsing_masks.size()[-2], parsing_masks.size()[-1]), mode = 'bilinear')

        channel_size = parsing_feat.size()[2] * parsing_feat.size()[3]
        parsing_masks = parsing_masks.view(parsing_masks.size()[0], parsing_masks.size()[1], channel_size)
        parsing_masks = torch.transpose(parsing_masks, 1, 2)
        parsing_feat = parsing_feat.view(parsing_feat.size()[0], parsing_feat.size()[1], channel_size)
        parsing_feat = torch.bmm(parsing_feat, parsing_masks)
        #average pooling
        parsing_feat = parsing_feat.view(parsing_feat.size()[0], parsing_feat.size()[1] * parsing_feat.size()[2]).cuda() / float(channel_size)
        #join features
        combofeat = torch.cat((x5c_feat, midfeat, parsing_feat), dim=1)

        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class multi_ResNet50M_v1(nn.Module):
    """ResNet50 + mid-level features.
    add multiple streams for saliency weighting, etc
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(multi_ResNet50M_v1, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(3200, num_classes)
        self.feat_dim = 3200 # feature dimension
        self.use_salience = True
        self.use_parsing = False

    def forward(self, x, saliency_maps):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        #resize feature map to same size of saliency map. (batch_size, 2048, 8, 4) ==> (batch_size, 2048, 4)
        x5c_saliency = x5c.view(x5c.size(0), x5c.size(1), x5c.size(3), x5c.size(2))
        x5c_saliency = F.avg_pool2d(x5c_saliency, (1, x5c_saliency.size()[3])).view(saliency_maps.size(0), saliency_maps.size(1), saliency_maps.size(2))
        saliency_feat = saliency_maps * x5c_saliency
        saliency_feat = F.avg_pool1d(saliency_feat, saliency_feat.size()[2:]).view(saliency_feat.size(0), saliency_feat.size(1))

        #join features
        combofeat = torch.cat((x5c_feat, midfeat, saliency_feat), dim=1)

        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
class multi_ResNet50M_v2(nn.Module):
    """ResNet50 + mid-level features.
    add multiple streams for saliency weighting, etc
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(multi_ResNet50M_v2, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(3200, num_classes)
        self.feat_dim = 3200 # feature dimension
        self.use_salience = True
        self.use_parsing = False

    def forward(self, x, saliency_maps):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        #resize feature map to same size of saliency map. (batch_size,  ) ==> (batch_size, 2048, 4)
        x5b_saliency = x5c.view(x5b.size(0), x5b.size(1), x5b.size(3), x5b.size(2))
        x5b_saliency = F.avg_pool2d(x5b_saliency, (1, x5b_saliency.size()[3])).view(saliency_maps.size(0), saliency_maps.size(1), saliency_maps.size(2))
        saliency_feat = saliency_maps * x5b_saliency
        saliency_feat = F.avg_pool1d(saliency_feat, saliency_feat.size()[2:]).view(saliency_feat.size(0), saliency_feat.size(1))
        #saliency_feat = saliency_feat.view(saliency_feat.size[0], saliency_feat.size[1] * saliency_feat.size[2])

        #join features
        combofeat = torch.cat((x5c_feat, midfeat, saliency_feat), dim=1)

        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50M_layer4(nn.Module):
    """
    deprecated
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50M_layer4, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4a = base[6][0]
        self.layers4b = base[6][1]
        self.layers4c = base[6][2]
        self.layers4d = base[6][3]
        self.layers4e = base[6][4]
        self.layers4f = base[6][5]
        self.fc_fuse = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension
        self.use_salience = False
        self.use_parsing = False

    def forward(self, x):
        '''
        x: batch of input images
        '''
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4a = self.layers4a(x3)
        x4b = self.layers4b(x4a)
        x4c = self.layers4c(x4b)
        x4d = self.layers4d(x4c)
        x4e = self.layers4e(x4d)
        x4f = self.layers4f(x4e)

        x4d_feat = F.avg_pool2d(x4d, x4d.size()[2:]).view(x4d.size(0), x4d.size(1))
        x4e_feat = F.avg_pool2d(x4e, x4e.size()[2:]).view(x4e.size(0), x4e.size(1))
        x4f_feat = F.avg_pool2d(x4f, x4f.size()[2:]).view(x4f.size(0), x4f.size(1))
        
        midfeat = torch.cat((x4d_feat, x4e_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        combofeat = torch.cat((x4f_feat, midfeat), dim=1)
        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class multi_ResNet50M_v3_layer4(nn.Module):
    """
    deprecated
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(multi_ResNet50M_v3_layer4, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4a = base[6][0]
        self.layers4b = base[6][1]
        self.layers4c = base[6][2]
        self.layers4d = base[6][3]
        self.layers4e = base[6][4]
        self.layers4f = base[6][5]
        self.fc_fuse = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(3072, num_classes)
        self.feat_dim = 3072 # feature dimension
        self.use_salience = True
        self.use_parsing = False

    def forward(self, x, salience_masks):
        '''
        x: batch of input images
        salince_mask: batch of 2D tensor
        '''
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4a = self.layers4a(x3)
        x4b = self.layers4b(x4a)
        x4c = self.layers4c(x4b)
        x4d = self.layers4d(x4c)
        x4e = self.layers4e(x4d)
        x4f = self.layers4f(x4e)

        x4d_feat = F.avg_pool2d(x4d, x4d.size()[2:]).view(x4d.size(0), x4d.size(1))
        x4e_feat = F.avg_pool2d(x4e, x4e.size()[2:]).view(x4e.size(0), x4e.size(1))
        x4f_feat = F.avg_pool2d(x4f, x4f.size()[2:]).view(x4f.size(0), x4f.size(1))
        
        midfeat = torch.cat((x4d_feat, x4e_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        #upsample feature map to the same size of salience/parsing maps
        salience_feat = F.upsample(x4f, size = (salience_masks.size()[-2], salience_masks.size()[-1]), mode = 'bilinear')

        channel_size = salience_feat.size()[2] * salience_feat.size()[3]
        salience_masks = salience_masks.cuda()
        salience_masks = salience_masks.view(salience_masks.size()[0], channel_size, 1)
        salience_feat = salience_feat.view(salience_feat.size()[0], salience_feat.size()[1], channel_size)
        salience_feat  = torch.bmm(salience_feat, salience_masks)#instead of replicating we use matrix product
        #average pooling
        salience_feat = salience_feat.view(salience_feat.size()[:2]) / float(channel_size)
        #join features
        combofeat = torch.cat((x4f_feat, midfeat, salience_feat), dim=1)

        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class multi_ResNet50M_v4_layer4(nn.Module):
    """
    deprecated
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(multi_ResNet50M_v4_layer4, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4a = base[6][0]
        self.layers4b = base[6][1]
        self.layers4c = base[6][2]
        self.layers4d = base[6][3]
        self.layers4e = base[6][4]
        self.layers4f = base[6][5]
        self.fc_fuse = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(7168, num_classes)
        self.feat_dim = 7168 # feature dimension
        self.use_salience = False
        self.use_parsing = True

    def forward(self, x, parsing_masks):
        '''
        x: batch of input images
        parsing_maks: batch of 3D tensor (various parsing masks per image)
        '''
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4a = self.layers4a(x3)
        x4b = self.layers4b(x4a)
        x4c = self.layers4c(x4b)
        x4d = self.layers4d(x4c)
        x4e = self.layers4e(x4d)
        x4f = self.layers4f(x4e)

        x4d_feat = F.avg_pool2d(x4d, x4d.size()[2:]).view(x4d.size(0), x4d.size(1))
        x4e_feat = F.avg_pool2d(x4e, x4e.size()[2:]).view(x4e.size(0), x4e.size(1))
        x4f_feat = F.avg_pool2d(x4f, x4f.size()[2:]).view(x4f.size(0), x4f.size(1))
        
        midfeat = torch.cat((x4d_feat, x4e_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        #upsample feature map to the same size of salience/parsing maps
        parsing_feat = F.upsample(x4f, size = (parsing_masks.size()[-2], parsing_masks.size()[-1]), mode = 'bilinear')

        channel_size = parsing_feat.size()[2] * parsing_feat.size()[3]
        parsing_masks = parsing_masks.view(parsing_masks.size()[0], parsing_masks.size()[1], channel_size)
        parsing_masks = torch.transpose(parsing_masks, 1, 2)
        parsing_feat = parsing_feat.view(parsing_feat.size()[0], parsing_feat.size()[1], channel_size)
        parsing_feat = torch.bmm(parsing_feat, parsing_masks)
        #average pooling
        parsing_feat = parsing_feat.view(parsing_feat.size()[0], parsing_feat.size()[1] * parsing_feat.size()[2]).cuda() / float(channel_size)
        #join features
        combofeat = torch.cat((x4f_feat, midfeat, parsing_feat), dim=1)

        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization + relu.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s, p):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
        
class multi_ResNet50M_v3_conv_up(nn.Module):
    """
    deprecated
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(multi_ResNet50M_v3_conv_up, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4a = base[6][0]
        self.layers4b = base[6][1]
        self.layers4c = base[6][2]
        self.layers4d = base[6][3]
        self.layers4e = base[6][4]
        self.layers4f = base[6][5]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.conv2 = ConvBlock(1024, 1024, k=3, s=1, p=1)
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(4096, num_classes)
        self.feat_dim = 4096 # feature dimension
        self.use_salience = True
        self.use_parsing = False

    def forward(self, x, salience_masks = None):
        '''
        x: batch of input images
        salince_mask: batch of 2D tensor
        '''
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4a = self.layers4a(x3)
        x4b = self.layers4b(x4a)
        x4c = self.layers4c(x4b)
        x4d = self.layers4d(x4c)
        x4e = self.layers4e(x4d)
        x4f = self.layers4f(x4e)
        x5a = self.layers5a(x4f)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))
        #print("4c", x4c.size()) #output [32, 1024, 16, 8]

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        #upsample feature map to the same size of salience/parsing maps
        salience_feat = F.upsample(x4f, scale_factor = 2, mode = 'bilinear')
        salience_feat = self.conv2(salience_feat)
        salience_feat = F.upsample(salience_feat, scale_factor = 2, mode = 'bilinear')
        salience_feat = self.conv2(salience_feat)
        salience_feat = F.upsample(salience_feat, scale_factor = 2, mode = 'bilinear')
        salience_feat = self.conv2(salience_feat)

        #implementation 2
        #reshape tensors such that we can use matrix product as operator and avoid using loops
        channel_size = salience_feat.size()[2] * salience_feat.size()[3]
        salience_masks = salience_masks.cuda()
        salience_masks = salience_masks.view(salience_masks.size()[0], channel_size, 1)
        salience_feat = salience_feat.view(salience_feat.size()[0], salience_feat.size()[1], channel_size)
        salience_feat  = torch.bmm(salience_feat, salience_masks)#instead of replicating we use matrix product
        #average pooling
        salience_feat = salience_feat.view(salience_feat.size()[:2]) / float(channel_size)
        #join features
        combofeat = torch.cat((x5c_feat, midfeat, salience_feat), dim=1)

        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class multi_ResNet50M_v4_conv_up(nn.Module):
    """
    deprecated
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(multi_ResNet50M_v4_conv_up, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4a = base[6][0]
        self.layers4b = base[6][1]
        self.layers4c = base[6][2]
        self.layers4d = base[6][3]
        self.layers4e = base[6][4]
        self.layers4f = base[6][5]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.conv2 = ConvBlock(1024, 1024, k=3, s=1, p=1)
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(8192, num_classes)
        self.feat_dim = 8192 # feature dimension
        self.use_salience = False
        self.use_parsing = True

    def forward(self, x, parsing_masks):
        '''
        x: batch of input images
        salince_mask: batch of 2D tensor
        parsing_maks: batch of 3D tensor (various parsing masks per image)
        '''
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4a = self.layers4a(x3)
        x4b = self.layers4b(x4a)
        x4c = self.layers4c(x4b)
        x4d = self.layers4d(x4c)
        x4e = self.layers4e(x4d)
        x4f = self.layers4f(x4e)
        x5a = self.layers5a(x4f)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))
        #print("4c", x4c.size()) #output [32, 1024, 16, 8]

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        #upsample feature map to the same size of salience/parsing maps
        parsing_feat = F.upsample(x4f, scale_factor = 2, mode = 'bilinear')
        parsing_feat = self.conv2(parsing_feat)
        parsing_feat = F.upsample(parsing_feat, scale_factor = 2, mode = 'bilinear')
        parsing_feat = self.conv2(parsing_feat)
        parsing_feat = F.upsample(parsing_feat, scale_factor = 2, mode = 'bilinear')
        parsing_feat = self.conv2(parsing_feat) 

        #implementation 2: uses matrix multiplication to avoid loops
        channel_size = parsing_feat.size()[2] * parsing_feat.size()[3]
        parsing_masks = parsing_masks.view(parsing_masks.size()[0], parsing_masks.size()[1], channel_size)
        parsing_masks = torch.transpose(parsing_masks, 1, 2)
        parsing_feat = parsing_feat.view(parsing_feat.size()[0], parsing_feat.size()[1], channel_size)
        parsing_feat = torch.bmm(parsing_feat, parsing_masks)
        #average pooling
        parsing_feat = parsing_feat.view(parsing_feat.size()[0], parsing_feat.size()[1] * parsing_feat.size()[2]).cuda() / float(channel_size)
        #join features
        combofeat = torch.cat((x5c_feat, midfeat, parsing_feat), dim=1)

        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50M_multi_layers_v1(nn.Module):
    """
    deprecated
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50M_multi_layers_v1, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(4608, num_classes)
        self.feat_dim = 4608 # feature dimension
        self.use_salience = False
        self.use_parsing = False

    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x3_feat = F.avg_pool2d(x3, x3.size()[2:]).view(x3.size(0), x3.size(1))
        x4_feat = F.avg_pool2d(x4, x4.size()[2:]).view(x4.size(0), x4.size(1))

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))
        
        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        combofeat = torch.cat((x3_feat, x4_feat, x5c_feat, midfeat), dim=1)
        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        elif self.loss == {'xent', 'whtri'}:
            return prelogits, x3_feat, x4_feat, x5c_feat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))