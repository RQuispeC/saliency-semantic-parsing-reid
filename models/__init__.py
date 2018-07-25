from __future__ import absolute_import

from .ResNet import *
from .DenseNet import *
from .Xception import *
from .Inception import *
from .ResNeXt import *

__factory = {
    'resnet50': ResNet50,
    'resnet50-salience': ResNet50_salience,
    'resnet50-parsing': ResNet50_parsing,
    'densenet121': DenseNet121,
    'densenet121-salience': DenseNet121_salience,
    'densenet121-parsing': DenseNet121_parsing,
    'resnet50m': ResNet50M,
    'resnet50m-salience': ResNet50M_salience,
    'resnet50m-parsing': ResNet50M_parsing,
    'xception': Xception,
    'xception-salience': Xception_salience,
    'xception-parsing': Xception_parsing,
    'inceptionv4': InceptionV4ReID,
    'inceptionv4-salience': InceptionV4ReID_salience,
    'inceptionv4-parsing': InceptionV4ReID_parsing,
    'resnext101': ResNeXt101_32x4d,
    'resnext101-salience': ResNeXt101_32x4d_salience, 
    'resnext101-parsing': ResNeXt101_32x4d_parsing,

    'resnet50-full': ResNet50_full,
    'densenet121-full': DenseNet121_full,
    'resnet50m-full': ResNet50M_full,
    'xception-full': Xception_full,
    'inceptionv4-full': InceptionV4ReID_full,
    
}

__use_salience = {
    'resnet50': False,
    'resnet50-salience': True,
    'resnet50-parsing': False,
    'densenet121': False,
    'densenet121-salience': True,
    'densenet121-parsing': False,
    'resnet50m': False,
    'resnet50m-salience': True,
    'resnet50m-parsing': False,
    'xception': False,
    'xception-salience': True,
    'xception-parsing': False,
    'inceptionv4': False,
    'inceptionv4-salience': True,
    'inceptionv4-parsing': False,
    'resnext101': False,
    'resnext101-salience': True, 
    'resnext101-parsing': False,

    'resnet50-full': False,
    'densenet121-full': False,
    'resnet50m-full': False,
    'xception-full': False,
    'inceptionv4-full': False,
}

__use_parsing = {
    'resnet50': False,
    'resnet50-salience': False,
    'resnet50-parsing': True,
    'densenet121': False,
    'densenet121-salience': False,
    'densenet121-parsing': True,
    'resnet50m': False,
    'resnet50m-salience': False,
    'resnet50m-parsing': True,
    'xception': False,
    'xception-salience': False,
    'xception-parsing': True,
    'inceptionv4': False,
    'inceptionv4-salience': False,
    'inceptionv4-parsing': True,
    'resnext101': False,
    'resnext101-salience': False,
    'resnext101-parsing': True,

    'resnet50-full': False,
    'densenet121-full': False,
    'resnet50m-full': False,
    'xception-full': False,
    'inceptionv4-full': False,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)

def use_salience(name):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __use_salience[name]

def use_parsing(name):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __use_parsing[name]