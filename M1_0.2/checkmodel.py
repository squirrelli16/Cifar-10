 
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 15:31:57 2022

@author: tuann
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torchsummary import summary


from BT2 import *

model = build_Net(num_classes=10, cifar=True)

model = model.cuda()
print ("model")
print (model)

# get the number of model parameters
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))
#print(model)
#model.cuda()
#summary(model, (3, 224, 224))
summary(model, (3, 32, 32))