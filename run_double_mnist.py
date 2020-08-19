#!/bin/python
#-----------------------------------------------------------------------------
# File Name : test_double_mnist.py
# Author: Emre Neftci
#
# Creation Date : Wed 06 May 2020 01:18:58 PM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from torchmeta.datasets.helpers import doublemnist
from torchmeta.utils.data import BatchMetaDataLoader

import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy
import torch.nn.functional as F
from tqdm import tqdm


n_ways = 5
inner_epochs = 5
outerstepsize0 = .1

class Naive(nn.Module):
	"""
	Define your network here.
	"""
	def __init__(self, n_way=5, imgsz=64):
		super(Naive, self).__init__()

		self.net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3),
		                         nn.AvgPool2d(kernel_size=2),
		                         nn.BatchNorm2d(64),
		                         nn.ReLU(inplace=True),

		                         nn.Conv2d(64, 64, kernel_size=3),
		                         nn.AvgPool2d(kernel_size=2),
		                         nn.BatchNorm2d(64),
		                         nn.ReLU(inplace=True),

		                         nn.Conv2d(64, 64, kernel_size=3),
		                         nn.BatchNorm2d(64),
		                         nn.ReLU(inplace=True),

		                         nn.Conv2d(64, 64, kernel_size=3),
		                         nn.BatchNorm2d(64),
		                         nn.ReLU(inplace=True)

		                         )

		# dummy forward to get feature size
		dummy_img = torch.randn(2, 3, imgsz, imgsz)
		repsz = self.net(dummy_img).size()
		_, c, h, w = repsz
		self.fc_dim = c * h * w

		self.fc = nn.Sequential(nn.Linear(self.fc_dim, 64),
		                        nn.ReLU(inplace=True),
		                        nn.Linear(64, n_way))


		print(self)
		print('Naive repnet sz:', repsz)

	def forward(self, x):
		x = self.net(x)
		x = x.view(-1, self.fc_dim)
		pred = self.fc(x)
		return pred

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 14 * 14, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_ways)
        self.flatten = nn.Flatten(start_dim=1, end_dim = -1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.tanh(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.tanh(self.conv2(x)), 2)
        x_ = self.flatten(x)
        x = F.tanh(self.fc1(x_))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x



def totorch(x):
    return x.clone().cuda()


def train_on_batch(model, data, targets, loss_fn, opt_fn):
    model.train()
    x = totorch(data)
    y = totorch(targets)
    model.zero_grad()
    opt_fn.zero_grad()
    ypred = model(x)
    loss_ = loss_fn(ypred,y)
    loss_.backward()
    opt_fn.step()
    return loss_

def train_on_batch_dl(model, dl, loss_fn, opt_fn):
    model.train()
    for data, targets in iter(dl):
        x = totorch(data)
        y = totorch(targets)
        model.zero_grad()
        opt_fn.zero_grad()
        ypred = model(x)
        loss_ = loss_fn(ypred,y)
        loss_.backward()
        opt_fn.step()
    return loss_
    
def predict(model, data):
    model.eval()
    x = totorch(data)
    return model(x).argmax(axis=1)

task_batch_size = 5
niterations = 100000
n_ways = 5
inner_epochs = 5
meta_eval_epochs = 1
eval_epochs = 10
inner_batch_size = 10
outerstepsize0 = .1

train_dataset = doublemnist('data', shots=10, ways=n_ways, shuffle=True, meta_split = "train", test_shots=15, download=True)
test_dataset = doublemnist('data', shots=1, ways=n_ways, shuffle=True, meta_split = "val", test_shots=15, download=True)
train_dataloader = BatchMetaDataLoader(train_dataset, batch_size=task_batch_size, num_workers=4)
test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=1, num_workers=4)

train_dataloader_iter = iter(train_dataloader)
test_dataloader_iter = iter(test_dataloader)

#meta-test evaluation
mbatch = next(test_dataloader_iter)
mdata, mtargets = mbatch["train"]
mdata_, mtargets_ = mbatch["test"]


model = Naive().cuda()
model_step = Naive().cuda()
model_meta = Naive().cuda()

#copy network
model_meta.load_state_dict(deepcopy(model.state_dict()))                           
model_step.load_state_dict(deepcopy(model.state_dict()))                           
loss_fn = torch.nn.CrossEntropyLoss()
opt_fn = torch.optim.Adam(model_step.parameters(), lr=1e-3)
meta_opt_fn = torch.optim.SGD(model_meta.parameters(), lr=1e-2)


acc_ = 0
for iteration in range(niterations):
    batch = next(train_dataloader_iter)
    data, targets = batch["train"]
    data_, targets_ = batch["test"]
    
    #outer loop
    weights_before = deepcopy(model.state_dict())
    weights_after_tasks = [None for i in range(len(data))]
    for task_id in range(len(data)):
        dl = torch.utils.data.DataLoader(list(zip(*[data[task_id],targets[task_id]])), batch_size = inner_batch_size)
        model_step.load_state_dict(model.state_dict())
        for i in range(inner_epochs):
            l = train_on_batch_dl(model_step,
                               dl,
                               loss_fn,
                               opt_fn)
            weights_after_tasks[task_id] = deepcopy(model_step.state_dict())
        acc_inner_train = (predict(model_step, data[task_id]) == targets[task_id].cuda()).float().mean() 
        acc_inner_test = (predict(model_step, data_[task_id]) == targets_[task_id].cuda()).float().mean() 
            
    outerstepsize = outerstepsize0 * (1 - (iteration / niterations)) # linear schedule
    model.load_state_dict({name : 
            weights_before[name] + (sum([weights_after_tasks[i][name]/len(data) for i in range(len(data))]) - weights_before[name])  * outerstepsize for name in weights_before})
            

    
    model_meta.load_state_dict(model.state_dict())

    task_id=0   
    
    for i in range(meta_eval_epochs):
        l = train_on_batch(model_meta, mdata[task_id], mtargets[task_id], loss_fn, meta_opt_fn)

    acc_ = (predict(model_meta, mdata_[task_id]) == mtargets_[task_id].cuda()).float().mean() 
    print(
        iteration,
        acc_inner_train.data.detach().cpu().numpy(), 
        acc_inner_test.data.detach().cpu().numpy(), 
        acc_.data.detach().cpu().numpy())
