import scipy.io as scio
import torch
import jw_functions
import numpy as np

## load data (Cross validation ver.)
eeg_train = scio.loadmat("C:/Users/dsa93/Desktop/job_preparing/github_code/python_ver/ccs_(R)csp/ccs_rcsp/data_csp/train/train_2", squeeze_me=True)
eeg_train_label = scio.loadmat("C:/Users/dsa93/Desktop/job_preparing/github_code/python_ver/ccs_(R)csp/ccs_rcsp/data_csp/train/train_label_2", squeeze_me=True)
test_data = scio.loadmat("C:/Users/dsa93/Desktop/job_preparing/github_code/python_ver/ccs_(R)csp/ccs_rcsp/data_csp/test/test_2", squeeze_me=True)
test_data_label = scio.loadmat("C:/Users/dsa93/Desktop/job_preparing/github_code/python_ver/ccs_(R)csp/ccs_rcsp/data_csp/test/test_label_2", squeeze_me=True)

# dict -> numpy
# [252, 118, 301] , [28, 118,301]
eeg_train = eeg_train['train_data']
eeg_train_label = eeg_train_label['train_label']
test_data = test_data['test_data']
test_data_label = test_data_label['test_label']

# tensor화
eeg_train = torch.from_numpy(eeg_train)
eeg_train_label = torch.from_numpy(eeg_train_label)
test_data = torch.from_numpy(test_data)
test_data_label = torch.from_numpy(test_data_label)

# signal preprocessing (CSP ver.)
#bp_eeg_train, bp_test_data = jw_functions.data_preprocessing(eeg_train, test_data)

# signal preprocessing (RCSP ver.)
bp_eeg_train, bp_test_data, alpha, beta, Ns = jw_functions.regular_data_preprocessing(eeg_train, test_data)


# CSP filter
#csp_filter = jw_functions.jw_csp(bp_eeg_train)
#csp_test_filter = jw_functions.jw_csp(bp_test_data)

# Regular CSP filter
csp_filter = jw_functions.jw_rcsp(bp_eeg_train,alpha,beta,Ns)
csp_test_filter = jw_functions.jw_rcsp(bp_test_data,alpha,beta,Ns)


# CSP filtered EEG data & make csp feature
csp_train_feature = jw_functions.csp_feature(csp_filter, bp_eeg_train, 'train')
csp_test_feature = jw_functions.csp_feature(csp_test_filter, bp_test_data, 'test')

# ck save
eeg_train_label = eeg_train_label.numpy()
test_data_label = test_data_label.numpy()

scio.savemat('C:/Users/dsa93/Desktop/compare_paper_other_algorithm/CCS-RCSP/code_matlab/BCI_III_IVa_dataset/data_csp/al/csp_train_feature'+'.mat',{'csp_train_feature':csp_train_feature})
scio.savemat('C:/Users/dsa93/Desktop/compare_paper_other_algorithm/CCS-RCSP/code_matlab/BCI_III_IVa_dataset/data_csp/al/eeg_train_label'+'.mat',{'eeg_train_label':eeg_train_label})

scio.savemat('C:/Users/dsa93/Desktop/compare_paper_other_algorithm/CCS-RCSP/code_matlab/BCI_III_IVa_dataset/data_csp/al/csp_test_feature'+'.mat',{'csp_test_feature':csp_test_feature})
scio.savemat('C:/Users/dsa93/Desktop/compare_paper_other_algorithm/CCS-RCSP/code_matlab/BCI_III_IVa_dataset/data_csp/al/eeg_test_label'+'.mat',{'test_data_label':test_data_label})


## train with SVM

from sklearn.svm import SVC

svm_model = SVC(kernel='rbf', C=1, gamma=0.1)
svm_model.fit(csp_train_feature, eeg_train_label)
y_pred = svm_model.predict(csp_test_feature)

print("prediction :", y_pred)
print("label : ", test_data_label)

ck_cnt = 0
for idx in range(len(y_pred)):

    if y_pred[idx] == test_data_label[idx]:

        ck_cnt +=1 

print("accuracy : ", ck_cnt/len(y_pred) *100,"%")

## result
