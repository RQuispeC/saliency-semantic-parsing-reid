from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision


__all__ = ['ResNet50', 'ResNet50_salience', 'ResNet50_parsing', 'ResNet50M', 'ResNet50M_salience', 'ResNet50M_parsing', 'ResNet50_full', 'ResNet50M_full', 'ResNet50M_salience_layer', 'ResNet50M_parsing_layer']

class ResNet50(nn.Module):
    """
    Code imported from https://github.com/KaiyangZhou/deep-person-reid
    """
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

class ResNet50_full(nn.Module):
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50_full, self).__init__()
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
        self.use_salience = False
        self.use_parsing = False

    def forward(self, x):
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

        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        #use x4f as feature
        x4f_feat = F.avg_pool2d(x4f, x4f.size()[2:]).view(x4f.size()[:2])

        #join features
        combofeat = torch.cat((x5c_feat, x4f_feat), dim=1)

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

class ResNet50M_full(nn.Module):
    """ResNet50m + mid-level features + weighting of mid-level features and x4f mid features
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50M_full, self).__init__()
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
        self.use_salience = False
        self.use_parsing = False

    def forward(self, x):
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

        #use x4f as feature
        x4f_feat = F.avg_pool2d(x4f, x4f.size()[2:]).view(x4f.size()[:2])

        #join features
        combofeat = torch.cat((x5c_feat, midfeat, x4f_feat), dim=1)

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

##### RESNET50M models for comparison of different layers
class ResNet50M_salience_layer(nn.Module):
    """ResNet50m + mid-level features + weighting of mid-level features and salience maps
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50M_salience_layer, self).__init__()
        self.mid_layer = kwargs['mid_layer']
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = nn.Sequential(base[6][0], base[6][1], base[6][2], base[6][3], base[6][4], base[6][5])
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        if self.mid_layer == 'layer1':
            self.feat_dim = 3136
        elif self.mid_layer == 'layer2':
            self.feat_dim = 3328
        elif self.mid_layer == 'layer3':
            self.feat_dim = 3584
        elif self.mid_layer == 'layer4':
            self.feat_dim = 4096
        else:
            raise KeyError("Unsupported mid layer: {}".format(self.mid_layer))
        self.classifier = nn.Linear(self.feat_dim, num_classes)
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
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        #upsample feature map to the same size of salience/parsing maps
        if self.mid_layer == 'layer1':
            salience_feat = F.upsample(x1, size = (salience_masks.size()[-2], salience_masks.size()[-1]), mode = 'bilinear')
        elif self.mid_layer == 'layer2':
            salience_feat = F.upsample(x2, size = (salience_masks.size()[-2], salience_masks.size()[-1]), mode = 'bilinear')
        elif self.mid_layer == 'layer3':
            salience_feat = F.upsample(x3, size = (salience_masks.size()[-2], salience_masks.size()[-1]), mode = 'bilinear')
        elif self.mid_layer == 'layer4':
            salience_feat = F.upsample(x4, size = (salience_masks.size()[-2], salience_masks.size()[-1]), mode = 'bilinear')
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

class ResNet50M_parsing_layer(nn.Module):
    """ResNet50 + mid-level features + weighting of mid-level features and semantic parsing maps
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50M_parsing_layer, self).__init__()
        self.mid_layer = kwargs['mid_layer']
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = nn.Sequential(base[6][0], base[6][1], base[6][2], base[6][3], base[6][4], base[6][5])
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        if self.mid_layer == 'layer1':
            self.feat_dim = 3392
        elif self.mid_layer == 'layer2':
            self.feat_dim = 4352
        elif self.mid_layer == 'layer3':
            self.feat_dim = 5632
        elif self.mid_layer == 'layer4':
            self.feat_dim = 8192
        else:
            raise KeyError("Unsupported mid layer: {}".format(self.mid_layer))
        self.classifier = nn.Linear(self.feat_dim, num_classes)
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
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        #upsample feature map to the same size of salience/parsing maps
        if self.mid_layer == 'layer1':
            parsing_feat = F.upsample(x1, size = (parsing_masks.size()[-2], parsing_masks.size()[-1]), mode = 'bilinear')
        elif self.mid_layer == 'layer2':
            parsing_feat = F.upsample(x2, size = (parsing_masks.size()[-2], parsing_masks.size()[-1]), mode = 'bilinear')
        elif self.mid_layer == 'layer3':
            parsing_feat = F.upsample(x3, size = (parsing_masks.size()[-2], parsing_masks.size()[-1]), mode = 'bilinear')
        elif self.mid_layer == 'layer4':
            parsing_feat = F.upsample(x4, size = (parsing_masks.size()[-2], parsing_masks.size()[-1]), mode = 'bilinear')

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
