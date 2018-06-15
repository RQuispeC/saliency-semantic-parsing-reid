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
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)