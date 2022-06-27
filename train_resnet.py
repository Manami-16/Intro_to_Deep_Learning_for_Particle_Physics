import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pyarrow as pa
import pyarrow.parquet as pq
import glob
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import ConcatDataset,DataLoader,sampler
from sklearn.metrics import roc_curve, auc
from utils import *
import os, time
from torchshape import tensorshape
from tqdm import tqdm

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = out_channels//in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.downsample, padding=1)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.downsample)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample > 1:
            residual = self.shortcut(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):

    def __init__(self, in_channels, nblocks, fmaps):
        super(ResNet, self).__init__()
        self.fmaps = fmaps
        self.nblocks = nblocks

        self.conv0 = nn.Conv2d(in_channels, fmaps[0], kernel_size=7, stride=1, padding=1)
        self.layer1 = self.block_layers(self.nblocks, [fmaps[0],fmaps[0]])
        self.layer2 = self.block_layers(1, [fmaps[0],fmaps[1]])
        self.layer3 = self.block_layers(self.nblocks, [fmaps[1],fmaps[1]])
        self.fc = nn.Linear(fmaps[1], 1)
        self.dropout = nn.Dropout(0.3)
        
    def block_layers(self, nblocks, fmaps):
        layers = []
        for _ in range(nblocks):
            layers.append(ResBlock(fmaps[0], fmaps[1]))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout(x)#dropout
        x = self.layer3(x)

        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size()[0], self.fmaps[1])
        x = self.dropout(x)#dropout
        x = self.fc(x)
        
        return x
    
def save_checkpoint(epoch, model, train_loss, train_acc, val_best_auc, total_val_loss, total_val_acc, 
                    total_fpr, total_tpr, prefix=""):
    
    model_out_path = "./saves/" + prefix
    
    state = {"epoch":epoch,
             "model": model.state_dict(),
             "train_loss": train_loss, 
             "train_acc": train_acc, 
             "val_best_auc": val_best_auc, 
             "total_val_loss": total_val_loss, 
             "total_val_acc": total_val_acc, 
             "total_fpr": total_fpr, 
             "total_tpr": total_tpr}
    
    if not os.path.exists("./saves/"):
        os.makedirs("./saves/")

    torch.save(state, model_out_path)
    print("model checkpoint saved @ {}".format(model_out_path))

## Function to validate after training phase
def do_eval(model, val_loader, roc_auc_best, epoch, train_data, device):
    loss_, acc_ = 0., 0.
    y_pred_, y_truth_, pt_ = [], [], []
    
    for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        X, y = data[train_data].to(device), data['m'].to(device) # electron/photon id stored in 'm' key
            
        logits = model(X).to(device)
        y = y.type_as(logits) # needs to be same type as logits for next line
        loss_ += F.binary_cross_entropy_with_logits(logits, y).item()
        pred = logits.ge(0.).byte()
        acc_ += pred.eq(y.byte()).float().mean().item()
        y_pred = torch.sigmoid(logits) # not used as an activation fn here; it's to map the logits to the interval [0, 1].
        # Store batch metrics:
        y_pred_.append(y_pred.tolist())
        y_truth_.append(y.tolist())

    y_pred_ = np.concatenate(y_pred_)
    y_truth_ = np.concatenate(y_truth_)
    s = '%d: Val loss:%.4f, acc:%.4f'%(epoch, loss_/len(val_loader), acc_/len(val_loader))
    print(s)

    print('Epoch: {}, Validation loss: {:.4f}, Validation accuracy: {:.4f}'.format(epoch, loss_/len(val_loader), acc_/len(val_loader)))
    
    fpr, tpr, _ = roc_curve(y_truth_, y_pred_)
    roc_auc = auc(fpr, tpr)
    s = "VAL ROC AUC: %.4f"%(roc_auc)
    print(s)
    
    return roc_auc, loss_/len(val_loader), acc_/len(val_loader), fpr, tpr

    

def train_resnet(device, lr, num_layers, hidden_dim, epochs, train_data='X', save_interval=1):
    
    ######## set up dataset ########
    
    event_data = glob.glob('./data/Single*Pt15To100_pythia8_PU2017_MINIAODSIM.parquet.[1-8]')

    # Training dataset - for optimizing net
    train_set = ConcatDataset([ParquetDatasetLimitedScaled(d) for d in event_data])
    idxs = np.random.permutation(len(train_set))
    train_frac = 0.7  # fraction of training set to the total dataset; the rest will be used for validation
    train_cut = int(train_frac*len(train_set)) 

    train_sampler = sampler.SubsetRandomSampler(idxs[:train_cut])
    train_loader = DataLoader(dataset=train_set, batch_size=32, num_workers=4, drop_last=True,
                              sampler=train_sampler, pin_memory=True)        

    # Validation dataset - for evaluating net performance
    val_set = ConcatDataset([ParquetDatasetLimitedScaled(d) for d in event_data])
    val_sampler = sampler.SubsetRandomSampler(idxs[train_cut:])
    val_loader = DataLoader(dataset=val_set, batch_size=120, num_workers=4, drop_last=True,
                            sampler=val_sampler)
    
    ## initialize a model ##
    print(device)
    model = ResNet(in_channels=1, nblocks=num_layers, fmaps=[16, 32])
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.5)
    print(summary(model, (1, 32, 32)))
    
    ## MAIN ##
    print_step = 5000
    roc_auc_best = 0.5
    print(f'training data is {train_data}')
    print(">> Training <<<<<<<<")

    # training loss and accuracy
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    val_best_auc = []
    total_val_loss = []
    total_val_acc = []
    total_fpr = []
    total_tpr = []

    # training loop
    for e in range(epochs):

        epoch = e+1
        s = '\n>> Epoch %d <<<<<<<<'%(epoch)
        print(s)

        # Run training
        model.train()
        now = time.time()
        running_loss, running_acc = 0, 0

        for data in tqdm(train_loader, total=len(train_loader)):
            X, y = data[train_data].to(device), data['m'].to(device)
            optimizer.zero_grad()
            logits = model(X).to(device)
            y = y.type_as(logits) #same type for next line
            loss = F.binary_cross_entropy_with_logits(logits, y).to(device)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pred = logits.ge(0.).byte()
            acc = pred.eq(y.byte()).float().mean()
            running_acc += acc.item()

        now = time.time() - now
        s = '%d: Train time:%.2fs in %d steps'%(epoch, now, len(train_loader))
        print(s)

        train_loss.append(running_loss/len(train_loader))
        train_acc.append(running_acc/len(train_loader))
        print('Epoch: {}, Train loss: {:.4f}, Train accuracy: {:.4f}'.format(epoch, train_loss[epoch-1], train_acc[epoch-1]))

        # Run Validation
        model.eval()
        roc_auc, val_loss, val_acc, fpr, tpr = do_eval(model, val_loader, roc_auc_best, epoch, train_data, device)
        val_best_auc.append(roc_auc)
        total_val_loss.append(val_loss)
        total_val_acc.append(val_acc)
        total_fpr.append(fpr)
        total_tpr.append(tpr)
        
        lr_scheduler.step()
        
        if epoch % save_interval == 0 and epoch > 0:
            save_epoch = (epoch // save_interval) * save_interval
            prefix = f'resnet_w_{num_layers}_layers_saved_at_{epoch}_epochs_trained_on_{train_data}'
            save_checkpoint(epoch, model, train_loss, train_acc, val_best_auc, total_val_loss, total_val_acc, total_fpr, total_tpr, prefix)
    
    return model
