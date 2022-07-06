'''
1. class for parquet files
2. validation loop 
(Kyungmin Park, CMU)
'''

import pickle
import numpy as np
import ROOT as r
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

names = ['idx','pt','pho_id','pi0_p4','m','Xtz','ieta','iphi','pho_p4','X','dR']

## class for parquet data
class ParquetDataset:
    def __init__(self, filename):
        self.parquet = pq.ParquetFile(filename)
        self.cols = None # read all columns
        #self.cols = ['X_jets.list.item.list.item.list.item','y'] 
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols).to_pydict()

        data['idx'] = np.int64(data['idx'])
        for name in names:
            if name == 'idx':
                continue
            data[name] = np.double(data[name]) # everything else is doubles

        #data['X_jets'] = np.float32(data['X_jets'][0])  not sure why there's a [0] here
        return dict(data)
    def __len__(self):
        return self.parquet.num_row_groups
    
class ParquetDatasetLimited:
    def __init__(self, filename):
        self.y = [0] if "Electron" in filename else [1]
        self.parquet = pq.ParquetFile(filename)
        self.cols = ['X','pt','m']  
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols).to_pydict()

        data['X'] = np.float32(data['X'][0])
        data['pt'] = np.float32(data['pt'])
        data['m'] = np.int64(self.y)

        data['X'][data['X'] < 1.e-3] = 0. # Zero-Suppression
        #data['X'][-1,...] = 25.*data['X_jets'][-1,...] # For HCAL: to match pixel intensity distn of other layers
        #data['X'] = data['X']/100. # To standardize
        
        return dict(data)
    
    def __len__(self):
        return self.parquet.num_row_groups
    
class ParquetDatasetLimitedScaled:
    def __init__(self, filename):
        self.y = [0] if "Electron" in filename else [1]
        self.parquet = pq.ParquetFile(filename)
        self.cols = ['X','pt','m']
        self.scaler = StandardScaler()
        
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols).to_pydict()

        data['X'] = np.float32(data['X'][0])
        
        data['pt'] = np.float32(data['pt']).reshape(1, -1)
        data['pt'] = self.scaler.fit_transform(data['pt'])
        
        data['m'] = np.int64(self.y)

        data['X'][data['X'] < 1.e-3] = 0. # Zero-Suppression
        #data['X'][-1,...] = 25.*data['X_jets'][-1,...] # For HCAL: to match pixel intensity distn of other layers
        data['X'] = data['X']/50. # To standardize
        
        return dict(data)
    
    def __len__(self):
        return self.parquet.num_row_groups
    
    
class ParquetToNumpy:
    def __init__(self,type,nfiles):
        
        filename = 'data/SingleElectronPt15To100_pythia8_PU2017_MINIAODSIM.parquet.' if type=='e' else 'data/SinglePhotonPt15To100_pythia8_PU2017_MINIAODSIM.parquet.'
        
        if type != 'e':
            type = 'g'
            
        try:
            self.datadict = pickle.load( open( "obj/"+type+"_"+str(nfiles)+"_dict.pkl", "rb" ) )
            if self.datadict != None:
                return
        except IOError:
            pass

        self.datadict = {}
        for name in names:
            self.datadict[name] = []

        for i in range(nfiles):
            parquet = ParquetDataset(filename+str(i+1))
            for name in names:
                print(i,name)
                for j in range(len(parquet)):
                    self.datadict[name].append(parquet[j][name][0])
        
        for name in names:
            self.datadict[name] = np.stack(self.datadict[name], axis=0)
                    
        pickle.dump( self.datadict, open( "obj/"+type+"_"+str(nfiles)+"_dict.pkl", "wb" ) )

    def __getitem__(self, name):
        return self.datadict[name]

    def getDict(self):
        return self.datadict
    
    
## Function to validate after training phase
# def do_eval(model, val_loader, roc_auc_best, epoch, train_data, device):
#     loss_, acc_ = 0., 0.
#     y_pred_, y_truth_, pt_ = [], [], []
# #     now = time.time()
#     for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
#         X, y = data[train_data].cuda(), data['m'].cuda() # electron/photon id stored in 'm' key
            
#         logits = model(X).to(device)
#         y = y.type_as(logits) # needs to be same type as logits for next line
#         loss_ += F.binary_cross_entropy_with_logits(logits, y).item()
#         pred = logits.ge(0.).byte()
#         acc_ += pred.eq(y.byte()).float().mean().item()
#         y_pred = torch.sigmoid(logits) # not used as an activation fn here; it's to map the logits to the interval [0, 1].
#         # Store batch metrics:
#         y_pred_.append(y_pred.tolist())
#         y_truth_.append(y.tolist())


# #     now = time.time() - now
#     y_pred_ = np.concatenate(y_pred_)
#     y_truth_ = np.concatenate(y_truth_)
#     s = '\n%d: Val time:%.2fs in %d steps'%(epoch, now, len(val_loader))
#     print(s)
#     s = '%d: Val loss:%.4f, acc:%.4f'%(epoch, loss_/len(val_loader), acc_/len(val_loader))
#     print(s)

# #     val_loss.append(loss_/len(val_loader))
# #     val_acc.append(acc_/len(val_loader))
#     print('Epoch: {}, Validation loss: {:.4f}, Validation accuracy: {:.4f}'.format(epoch, loss_/len(val_loader), acc_/len(val_loader)))
    
#     fpr, tpr, _ = roc_curve(y_truth_, y_pred_)
#     roc_auc = auc(fpr, tpr)
#     s = "VAL ROC AUC: %.4f"%(roc_auc)
#     print(s)

#     if roc_auc > roc_auc_best:
#         roc_auc_best = roc_auc
    
#     return roc_auc_best, loss_/len(val_loader), acc_/len(val_loader), fpr, tpr

    