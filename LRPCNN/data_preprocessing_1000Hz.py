
import torch
import numpy as np
import scipy.io as scio
import matplotlib as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader 


# declare data_class
class train_data_dataset(Dataset):
    '''EEG dataset'''

    # data load, preprocessing
    def __init__(self):
        
        # load kfold_data file
        x_train = scio.loadmat('cv_data/ch118/stft_aw/train/train_data_1', squeeze_me=True)
        y_train = scio.loadmat('cv_data/ch118/stft_aw/train/train_label_1', squeeze_me=True)


        # extract signal only
        x_train = x_train['seperate_train_data']
        y_train = y_train['seperate_train_label']
        
        
        # make tensor
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)

        # x_train.shape = [252,18,200, 200]
        # y_train.shape = [252,1]

        # .unsqueeze-> 3d version
        #self.train_data = x_train.unsqueeze(1)
        
        self.train_data = x_train
        self.train_label = y_train

        
    def __getitem__(self, index):
        _x = self.train_data[index]
        _y = self.train_label[index]
        return _x, _y

    def __len__(self):
        return len(self.train_data)



class EEG_val_dataset(Dataset):
    '''EEG dataset'''

    # data load, preprocessing
    def __init__(self):       
        # load kfold_data file
        
        
        x_val = scio.loadmat('cv_data/ch118/stft_aw/val/val_data_1', squeeze_me=True)
        y_val = scio.loadmat('cv_data/ch118/stft_aw/val/val_label_1', squeeze_me=True)

        # extract signal only        
        x_val = x_val['val_data']
        y_val = y_val['val_label']

        # make tensor
        x_val = torch.from_numpy(x_val).double()
        y_val = torch.from_numpy(y_val).double()


        # x_test.shape = [28,18,258,24]
        # y_test.shape = [28,1]

        # .unsqueeze-> 3d version
        #self.test_data = x_test.unsqueeze(1)
        self.val_data = x_val
        self.val_label= y_val
   
    def __getitem__(self, index):
        _x = self.val_data[index]
        _y = self.val_label[index]
        return _x, _y

    def __len__(self):
        return len(self.val_data)

   


class test_data_dataset(Dataset):
    '''EEG dataset'''

    # data load, preprocessing
    def __init__(self):       
        # load kfold_data file
        
        
        x_test = scio.loadmat('cv_data/ch118/stft_aw/test/test_data_1', squeeze_me=True)
        y_test = scio.loadmat('cv_data/ch118/stft_aw/test/test_label_1', squeeze_me=True)

        # extract signal only
        #x_test = x_test['freq_reduced_xtest']
       
        x_test = x_test['test_data']
        y_test = y_test['test_label']

        # make tensor
        x_test = torch.from_numpy(x_test).double()
        y_test = torch.from_numpy(y_test).double()


        # x_test.shape = [28,18,258,24]
        # y_test.shape = [28,1]

        # .unsqueeze-> 3d version
        #self.test_data = x_test.unsqueeze(1)
        self.test_data = x_test
        self.test_label= y_test
   
    def __getitem__(self, index):
        _x = self.test_data[index]
        _y = self.test_label[index]
        return _x, _y

    def __len__(self):
        return len(self.test_data)

   

