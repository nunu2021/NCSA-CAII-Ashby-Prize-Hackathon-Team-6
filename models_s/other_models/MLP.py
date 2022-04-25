from torch.utils.data import TensorDataset
import torch
import numpy as np
import time
from tqdm import tqdm

import torch.nn as nn
from collections import OrderedDict

class SimpleMLP(torch.nn.Module):
    def __init__(self, ninputs=100, nhidden=256, nouts=10, activations='ELU'):
        super().__init__()
        self.layers = OrderedDict([
            ('linear1', nn.Linear(ninputs,nhidden)),
            
            ('act2', nn.ELU()),
            ('dropout1', nn.Dropout()),
            ('linear2', nn.Linear(nhidden,nhidden)),

            ('act2', nn.ELU()),
            ('dropout2', nn.Dropout()),
            ('linear3', nn.Linear(nhidden,nhidden)),

            ('act3', nn.ELU()),
            ('dropout3', nn.Dropout()),
            ('linear4', nn.Linear(nhidden,nouts)),
        ])
        self.net = torch.nn.Sequential(self.layers)
    
    def forward(self, x):
        return self.net(x)
    
def loss_fn(ypred, y):
    return ((ypred-y)**2).mean()

def train_step(dl, model, optim):
    tstart = time.time()
    total_loss = 0
    count = 0
    for X, y in tqdm(dl):
        model.train()

        optim.zero_grad()
        
        ypred = model(X.cuda())
        loss  = loss_fn(ypred, y.cuda())
#         padded = torch.where(X < feat_min_list.unsqueeze(0), feat_min_list.unsqueeze(0), X)
#         Xpad_norm = (padded.log() - feat_mean_list)/ feat_std_list

#         padded = torch.where(y < targ_min_list.unsqueeze(0), targ_min_list.unsqueeze(0), y)
#         ypad_norm = (padded.log() - targ_mean_list)/ targ_std_list

#         ypred = model(Xpad_norm.cuda())

#         loss = loss_fn(ypred, ypad_norm.cuda())
        loss.backward()
        optim.step()
        
        total_loss += loss.item()

    return total_loss/ len(dl), time.time() - tstart

@torch.no_grad()
def val_step(dl, model):
    tstart = time.time()
    total_loss = 0
    for X, y in tqdm(dl):
        model.eval()
        
        ypred = model(X.cuda())
        loss  = loss_fn(ypred, y.cuda())
        
#         padded = torch.where(X < feat_min_list.unsqueeze(0), feat_min_list.unsqueeze(0), X)
#         Xpad_norm = (padded.log() - feat_mean_list)/ feat_std_list

#         padded = torch.where(y < targ_min_list.unsqueeze(0), targ_min_list.unsqueeze(0), y)
#         ypad_norm = (padded.log() - targ_mean_list)/ targ_std_list

#         ypred = model(Xpad_norm.cuda())

#         loss = loss_fn(ypred, ypad_norm.cuda())
        total_loss += loss.item()
    return total_loss/ len(dl) , time.time() - tstart


@torch.no_grad()
def predictions(dl, model):
    targets     = []
    predictions = []
    for X, y in tqdm(dl):
        model.eval()
        ypred = model(X.cuda()).detach().cpu().numpy()
        targets.append(y.numpy())
        predictions.append(ypred)
    targets = np.vstack(targets)
    predictions = np.vstack(predictions)
    return targets, predictions
        

def train_epoch(i, model, optim, seed_start=42, bsz=256, train_test_split=[0.8,0.1,0.1]):
    rng = np.random.default_rng(seed_start+i)
    arr = np.arange(133)
    rng.shuffle(arr)
    grps = np.array_split(arr, 19)
    
    total_train_loss = 0
    total_val_loss = 0
    for g in grps:
        dl_train, dl_val, dl_test = prepare_dataloaders(
            g, 
            train_test_split=train_test_split, 
            bsz = bsz, 
            generator_seed=seed_start
        )
        train_loss, train_time = train_step(dl_train, model, optim)
        val_loss, val_time = val_step(dl_val, model)
#         print(train_loss, val_loss)
        total_train_loss += train_loss
        total_val_loss   += val_loss
    return total_train_loss / len(grps), total_val_loss / len(grps)

        