import torch
import numpy as np
import scipy.io as scio
from scipy.sparse import csr_matrix
import matplotlib as plt

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import StratifiedKFold

# 2GB 넘는 파일읽어오기위한
import hdf5storage as hdf


def kfold_data():
    
    # .mat file load
    eeg_data = hdf.loadmat('cv_data/ch118/stft_aw/original/STFT_aw_train', squeeze_me=True)
    eeg_label = hdf.loadmat('cv_data/ch118/stft_aw/original/STFT_aw_label', squeeze_me=True)

    # extract signal only
    eeg_data = eeg_data['spectro_train']
    eeg_label = eeg_label['spectro_label']

    
    # make tensor
    eeg_data = torch.from_numpy(eeg_data).double()
    eeg_label = torch.from_numpy(eeg_label).double()
    
    print("x",eeg_data.shape)
    print("y",eeg_label.shape)

    eeg_data = eeg_data.numpy()
    eeg_label = eeg_label.numpy()

    # using stratifiedkfold
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    skf_val = StratifiedKFold(n_splits=9, shuffle=True)

    temp_idx =1
    cmp_idx = 0
    
    # seperate train/test = 9 : 1 
    for data_idx, test_idx in skf.split(eeg_data, eeg_label):
        print("data:",data_idx, "test",test_idx)

        train_data, test_data = eeg_data[data_idx], eeg_data[test_idx]
        train_label, test_label = eeg_label[data_idx], eeg_label[test_idx]

        scio.savemat('cv_data/ch118/stft_aw/test/test_data_'+str(temp_idx)+'.mat',{'test_data':test_data})
        scio.savemat('cv_data/ch118/stft_aw/test/test_label_'+str(temp_idx)+'.mat',{'test_label':test_label})

        cmp_idx = 0
        # seperate train/val = 8 : 1 
        for train_idx, val_idx in skf_val.split(train_data, train_label):

            if cmp_idx == 0:
                print("train_data", train_idx, "val_data", val_idx)

                seperate_train_data, val_data = train_data[train_idx], train_data[val_idx]
                seperate_train_label, val_label = train_label[train_idx], train_label[val_idx]

        
                scio.savemat('cv_data/ch118/stft_aw/train/train_data_'+str(temp_idx)+'.mat' ,{'seperate_train_data':seperate_train_data})
                scio.savemat('cv_data/ch118/stft_aw/train/train_label_'+str(temp_idx)+'.mat' ,{'seperate_train_label':seperate_train_label})
                
                
                scio.savemat('cv_data/ch118/stft_aw/val/val_data_'+str(temp_idx)+'.mat' ,{'val_data':val_data})
                scio.savemat('cv_data/ch118/stft_aw/val/val_label_'+str(temp_idx)+'.mat' ,{'val_label':val_label})

                
                cmp_idx += 1
                temp_idx +=1



kfold_data()
