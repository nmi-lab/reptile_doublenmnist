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
from create_doublenmnist import *
import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy
import torch.nn.functional as F
from tqdm import tqdm
import time


n_ways = 5
inner_epochs = 5
outerstepsize0 = .1

class Naive(nn.Module):
    """
    Define your network here.
    """
    def __init__(self, n_way=5, imgsz=32):
        super(Naive, self).__init__()
        input_features = 2
        self.net = nn.Sequential(nn.Conv2d(input_features, 64, kernel_size=3),
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
        dummy_img = torch.randn(2, input_features, imgsz, imgsz)
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

def totorch(x):
    return x.clone().cuda()

def train_on_batch(model, data, targets, loss_fn, opt_fn):
    model.train()
    x = totorch(data[:,0])
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
        x = totorch(data[:,0])
        y = totorch(targets)
        model.zero_grad()
        opt_fn.zero_grad()
        ypred = model(x)
        loss_ = loss_fn(ypred,y)
        loss_.backward()
        opt_fn.step()
    return loss_
    
def accuracy(model, dl):
    model.eval()
    acc = []
    for d,t in iter(dl):
        x = totorch(d[:,0])
        y = totorch(t)
        acc += (model(x).argmax(axis=1) == y).float()
    return torch.mean(torch.Tensor(acc))


niterations = 10000
nvals = 5
n_ways = 5
k_shots = 10
k_shots_test = 1
inner_epochs = 5
meta_eval_epochs = 50
inner_batch_size = 5
outerstepsize0 = 1.0
ntasks_per_inner = 5

#train_dataset = doublemnist('data', shots=10, ways=n_ways, shuffle=True, meta_split = "train", test_shots=15, download=True)
#test_dataset = doublemnist('data', shots=1, ways=n_ways, shuffle=True, meta_split = "val", test_shots=15, download=True)
#train_dataloader = BatchMetaDataLoader(train_dataset, batch_size=task_batch_size, num_workers=4)
#test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=1, num_workers=4)
dls = []
for i in range(nvals):
    dls.append( create_doublenmnist(batch_size = inner_batch_size, shots=k_shots_test, ways=n_ways, mtype='val', ds=2))

dlsdt = []
for i in range(nvals):
    dlsdt.append( next(iter(create_doublenmnist(batch_size = inner_batch_size, shots=k_shots_test, ways=n_ways, mtype='val', ds=2)[0])))

#meta-test evaluation
model = Naive().cuda()
model_step = Naive().cuda()
model_meta = Naive().cuda()

#copy network
model_meta.load_state_dict(deepcopy(model.state_dict()))                           
model_step.load_state_dict(deepcopy(model.state_dict()))                           
loss_fn = torch.nn.CrossEntropyLoss()
opt_fn = torch.optim.Adam(model_step.parameters(), lr=1e-3)
meta_opt_fn = torch.optim.SGD(model_meta.parameters(), lr=1e-3)


acc_s = []
acc_ = 0
for iteration in tqdm(range(niterations)):
    #outer loop
    weights_before = deepcopy(model.state_dict())
    weights_after_tasks = [None for i in range(ntasks_per_inner)]
    for task_id in range(ntasks_per_inner):
        train_dl, test_dl = create_doublenmnist(batch_size = inner_batch_size, shots=k_shots, ways=n_ways, mtype='train', ds=2)
        model_step.load_state_dict(model.state_dict())
        idl = iter(train_dl)

        for i in range(inner_epochs):
            data, targets = next(idl)
            l = train_on_batch(model_step,
                               data, targets,
                               loss_fn,
                               opt_fn)
            weights_after_tasks[task_id] = deepcopy(model_step.state_dict())
           
    outerstepsize = outerstepsize0 * (1 - (iteration / niterations)) # linear schedule
    model.load_state_dict({name : weights_before[name] + (sum([weights_after_tasks[i][name]/float(ntasks_per_inner) for i in range(ntasks_per_inner)]) - weights_before[name])  * outerstepsize for name in weights_before})



    
    acc_inner_train = accuracy(model_step, train_dl)
    acc_inner_test = accuracy(model_step, test_dl)
         

    model_meta.load_state_dict(model.state_dict())

    if (iteration%10) == 0:
        acc_ = 0
        for j in range(len(dlsdt)):
            mtrd, mtrt = dlsdt[j]
            for i in range(meta_eval_epochs):
                l = train_on_batch(model_meta, mtrd, mtrt, loss_fn, meta_opt_fn)

            mtr, mte = dls[j]
            acc_ += accuracy(model_meta, mte) 
        acc_ /= len(dls)
        acc_s.append(acc_)
        print(
            iteration,
            acc_inner_train.data.detach().cpu().numpy(), 
            acc_inner_test.data.detach().cpu().numpy(), 
            acc_.data.detach().cpu().numpy())
