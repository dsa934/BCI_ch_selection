
import torch
import numpy as np
import scipy.io as scio
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import StratifiedKFold


def kfold_data():
    
    # .mat file load
    eeg_data = scio.loadmat('C:/Users/dsa93/Desktop/job_preparing/github_code/python_ver/ccs_(R)csp/ccs_rcsp/data_csp/sort_eeg', squeeze_me=True)
    eeg_label = scio.loadmat('C:/Users/dsa93/Desktop/job_preparing/github_code/python_ver/ccs_(R)csp/ccs_rcsp/data_csp/sort_eeg_label', squeeze_me=True)

    print("check")
    # extract signal only
    eeg_data = eeg_data['sort_eeg']
    eeg_label = eeg_label['sort_eeg_label']

    
    # make tensor
    eeg_data = torch.from_numpy(eeg_data).double()
    eeg_label = torch.from_numpy(eeg_label).double()
    
    print("x",eeg_data.shape)
    print("y",eeg_label.shape)


    eeg_data = eeg_data.numpy()
    eeg_label = eeg_label.numpy()

    # using stratifiedkfoldX
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    
    temp_idx = 1
    cmp_idx = 0
    
    # seperate train/test = 9 : 1 
    for data_idx, test_idx in skf.split(eeg_data, eeg_label):
        print("data:",data_idx, "test",test_idx)

        train_data, test_data = eeg_data[data_idx], eeg_data[test_idx]
        train_label, test_label = eeg_label[data_idx], eeg_label[test_idx]

        scio.savemat('C:/Users/dsa93/Desktop/job_preparing/github_code/python_ver/ccs_(R)csp/ccs_rcsp/data_csp/train/train_'+str(temp_idx)+'.mat',{'train_data':train_data})
        scio.savemat('C:/Users/dsa93/Desktop/job_preparing/github_code/python_ver/ccs_(R)csp/ccs_rcsp/data_csp/train/train_label_'+str(temp_idx)+'.mat',{'train_label':train_label})

        scio.savemat('C:/Users/dsa93/Desktop/job_preparing/github_code/python_ver/ccs_(R)csp/ccs_rcsp/data_csp/test/test_'+str(temp_idx)+'.mat',{'test_data':test_data})
        scio.savemat('C:/Users/dsa93/Desktop/job_preparing/github_code/python_ver/ccs_(R)csp/ccs_rcsp/data_csp/test/test_label_'+str(temp_idx)+'.mat',{'test_label':test_label})

        temp_idx +=1;


kfold_data()
