from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import reduce
import torch.utils.model_zoo as model_zoo

import os

__all__ = ['ResNeXt101_32x4d', 'ResNeXt101_32x4d_salience', 'ResNeXt101_32x4d_parsing']

"""
Code imported from https://github.com/Cadene/pretrained-models.pytorch
"""

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

# JUMP TO THE END #########################################################################
resnext101_32x4d_features = nn.Sequential( # Sequential,
    nn.Conv2d(3,64,(7, 7),(2, 2),(3, 3),1,1,bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d((3, 3),(2, 2),(1, 1)),
    nn.Sequential( # Sequential,
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(64,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(256),
                ),
                nn.Sequential( # Sequential,
                    nn.Conv2d(64,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(256),
                ),
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(256,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(256),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(256,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(256),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential( # Sequential,
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(256,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256,256,(3, 3),(2, 2),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(512),
                ),
                nn.Sequential( # Sequential,
                    nn.Conv2d(256,512,(1, 1),(2, 2),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(512),
                ),
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential( # Sequential,
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(2, 2),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                nn.Sequential( # Sequential,
                    nn.Conv2d(512,1024,(1, 1),(2, 2),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential( # Sequential,
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024,1024,(3, 3),(2, 2),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(2048),
                ),
                nn.Sequential( # Sequential,
                    nn.Conv2d(1024,2048,(1, 1),(2, 2),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(2048),
                ),
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(2048,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(2048),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(2048,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(2048),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            nn.ReLU(),
        ),
    )
)

#################################################################################
resnext101_64x4d_features = nn.Sequential(#Sequential,
    nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3), 1, 1, bias = False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d((3, 3), (2, 2), (1, 1)),
    nn.Sequential(#Sequential,
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(256),
                ),
                nn.Sequential(#Sequential,
                    nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(256),
                ),
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(256),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(256),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential(#Sequential,
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(512),
                ),
                nn.Sequential(#Sequential,
                    nn.Conv2d(256, 512, (1, 1), (2, 2), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(512),
                ),
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential(#Sequential,
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (2, 2), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                nn.Sequential(#Sequential,
                    nn.Conv2d(512, 1024, (1, 1), (2, 2), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential(#Sequential,
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                        nn.Conv2d(2048, 2048, (3, 3), (2, 2), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(2048),
                ),
                nn.Sequential(#Sequential,
                    nn.Conv2d(1024, 2048, (1, 1), (2, 2), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(2048),
                ),
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                        nn.Conv2d(2048, 2048, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(2048),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap(lambda x: x, #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                        nn.Conv2d(2048, 2048, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(2048),
                ),
                Lambda(lambda x: x), #Identity,
            ),
            LambdaReduce(lambda x, y: x + y), #CAddTable,
            nn.ReLU(),
        ),
    )
)

#################################################################################

pretrained_settings = {
    'resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/resnext101_32x4d-29e315fa.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'resnext101_64x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/resnext101_64x4d-e77a0586.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    }
}

def resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    """Deprecated"""
    model = ResNeXt101_32x4d(num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnext101_32x4d'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model

def resnext101_64x4d(num_classes=1000, pretrained='imagenet'):
    """Deprecated"""
    model = ResNeXt101_64x4d(num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnext101_64x4d'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model

##################### Model Definition #########################

class ResNeXt101_32x4d(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNeXt101_32x4d, self).__init__()
        self.loss = loss
        self.features = resnext101_32x4d_features
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048
        self.init_params()

    def init_params(self):
        """Load ImageNet pretrained weights"""
        settings = pretrained_settings['resnext101_32x4d']['imagenet']
        pretrained_dict = model_zoo.load_url(settings['url'], map_location=None)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, input):
        x = self.features(input)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if not self.training:
            return x

        y = self.classifier(x)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, x
        elif self.loss == {'cent'}:
            return y, x
        elif self.loss == {'ring'}:
            return y, x
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNeXt101_32x4d_salience(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNeXt101_32x4d_salience, self).__init__()
        self.loss = loss
        self.features = resnext101_32x4d_features
        self.classifier = nn.Linear(3072, num_classes)
        self.feat_dim = 3072
        self.init_params()
        base = nn.Sequential(*list(self.features.children()))
        self.x0 = base[0]
        self.x1 = base[1]
        self.x2 = base[2]
        self.x3 = base[3]
        self.x4 = base[4]
        self.x5 = base[5]
        self.x6 = base[6]
        self.x7 = base[7]
        

    def init_params(self):
        """Load ImageNet pretrained weights"""
        settings = pretrained_settings['resnext101_32x4d']['imagenet']
        pretrained_dict = model_zoo.load_url(settings['url'], map_location=None)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, input, salience_masks):
        #x = self.features(input)
        '''
        ('ff', torch.Size([32, 3, 256, 128]))
        ('f0', torch.Size([32, 64, 128, 64]))
        ('f1', torch.Size([32, 64, 128, 64]))
        ('f2', torch.Size([32, 64, 128, 64]))
        ('f3', torch.Size([32, 64, 64, 32]))
        ('f4', torch.Size([32, 256, 64, 32]))
        ('f5', torch.Size([32, 512, 32, 16]))
        ('f6', torch.Size([32, 1024, 16, 8]))
        ('f7', torch.Size([32, 2048, 8, 4]))
        '''
        x = self.x0(input)
        x = self.x1(x)
        x = self.x2(x)
        x = self.x3(x)
        x = self.x4(x)
        x = self.x5(x)
        x6 = self.x6(x)
        x = self.x7(x6)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

#upsample feature map to fit salience_masks
        salience_feat = F.upsample(x6, scale_factor = 8, mode = 'bilinear')
        #combine feature map with salience_masks (128, 64)
        channel_size = salience_feat.size()[2] * salience_feat.size()[3]
        salience_masks = salience_masks.cuda()
        salience_masks = salience_masks.view(salience_masks.size()[0], channel_size, 1)
        salience_feat = salience_feat.view(salience_feat.size()[0], salience_feat.size()[1], channel_size)
        salience_feat  = torch.bmm(salience_feat, salience_masks)#instead of replicating we use matrix product
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
        elif self.loss == {'cent'}:
            return y, x
        elif self.loss == {'ring'}:
            return y, x
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNeXt101_32x4d_parsing(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNeXt101_32x4d_parsing, self).__init__()
        self.loss = loss
        self.features = resnext101_32x4d_features
        self.classifier = nn.Linear(7168, num_classes)
        self.feat_dim = 7168
        self.init_params()
        base = nn.Sequential(*list(self.features.children()))
        self.x0 = base[0]
        self.x1 = base[1]
        self.x2 = base[2]
        self.x3 = base[3]
        self.x4 = base[4]
        self.x5 = base[5]
        self.x6 = base[6]
        self.x7 = base[7]
        

    def init_params(self):
        """Load ImageNet pretrained weights"""
        settings = pretrained_settings['resnext101_32x4d']['imagenet']
        pretrained_dict = model_zoo.load_url(settings['url'], map_location=None)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, input, parsing_masks):
        #x = self.features(input)
        '''
        ('ff', torch.Size([32, 3, 256, 128]))
        ('f0', torch.Size([32, 64, 128, 64]))
        ('f1', torch.Size([32, 64, 128, 64]))
        ('f2', torch.Size([32, 64, 128, 64]))
        ('f3', torch.Size([32, 64, 64, 32]))
        ('f4', torch.Size([32, 256, 64, 32]))
        ('f5', torch.Size([32, 512, 32, 16]))
        ('f6', torch.Size([32, 1024, 16, 8]))
        ('f7', torch.Size([32, 2048, 8, 4]))
        '''
        x = self.x0(input)
        x = self.x1(x)
        x = self.x2(x)
        x = self.x3(x)
        x = self.x4(x)
        x = self.x5(x)
        x6 = self.x6(x)
        x = self.x7(x6)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        #upsample feature map to fit parsing_masks
        parsing_feat = F.upsample(x6, scale_factor = 8, mode = 'bilinear')
        #combine feature map with parsing_masks (128, 64)
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
        elif self.loss == {'cent'}:
            return y, x
        elif self.loss == {'ring'}:
            return y, x
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNeXt101_32x4d_salience_b5(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNeXt101_32x4d_salience_b5, self).__init__()
        self.loss = loss
        self.features = resnext101_32x4d_features
        self.classifier = nn.Linear(2560, num_classes)
        self.feat_dim = 2560
        self.init_params()
        base = nn.Sequential(*list(self.features.children()))
        self.x0 = base[0]
        self.x1 = base[1]
        self.x2 = base[2]
        self.x3 = base[3]
        self.x4 = base[4]
        self.x5 = base[5]
        self.x6 = base[6]
        self.x7 = base[7]
        

    def init_params(self):
        """Load ImageNet pretrained weights"""
        settings = pretrained_settings['resnext101_32x4d']['imagenet']
        pretrained_dict = model_zoo.load_url(settings['url'], map_location=None)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, input, salience_masks):
        #x = self.features(input)
        '''
        ('ff', torch.Size([32, 3, 256, 128]))
        ('f0', torch.Size([32, 64, 128, 64]))
        ('f1', torch.Size([32, 64, 128, 64]))
        ('f2', torch.Size([32, 64, 128, 64]))
        ('f3', torch.Size([32, 64, 64, 32]))
        ('f4', torch.Size([32, 256, 64, 32]))
        ('f5', torch.Size([32, 512, 32, 16]))
        ('f6', torch.Size([32, 1024, 16, 8]))
        ('f7', torch.Size([32, 2048, 8, 4]))
        '''
        x = self.x0(input)
        x = self.x1(x)
        x = self.x2(x)
        x = self.x3(x)
        x = self.x4(x)
        x5 = self.x5(x)
        x = self.x6(x5)
        x = self.x7(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

#upsample feature map to fit salience_masks
        salience_feat = F.upsample(x5, scale_factor = 4, mode = 'bilinear')
        #combine feature map with salience_masks (128, 64)
        channel_size = salience_feat.size()[2] * salience_feat.size()[3]
        salience_masks = salience_masks.cuda()
        salience_masks = salience_masks.view(salience_masks.size()[0], channel_size, 1)
        salience_feat = salience_feat.view(salience_feat.size()[0], salience_feat.size()[1], channel_size)
        salience_feat  = torch.bmm(salience_feat, salience_masks)#instead of replicating we use matrix product
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
        elif self.loss == {'cent'}:
            return y, x
        elif self.loss == {'ring'}:
            return y, x
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNeXt101_64x4d(nn.Module):
    """This model is not used"""
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNeXt101_64x4d, self).__init__()
        self.loss = loss
        self.features = resnext101_64x4d_features
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048
        self.init_params()

    def init_params(self):
        """Load ImageNet pretrained weights"""
        settings = pretrained_settings['resnext101_64x4d']['imagenet']
        pretrained_dict = model_zoo.load_url(settings['url'], map_location=None)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, input):
        x = self.features(input)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if not self.training:
            return x

        y = self.classifier(x)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, x
        elif self.loss == {'cent'}:
            return y, x
        elif self.loss == {'ring'}:
            return y, x
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
