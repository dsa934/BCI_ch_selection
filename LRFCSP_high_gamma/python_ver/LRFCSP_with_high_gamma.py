
'''
 " Motor Imagery Classification using local region CSP features with high-gamma band " 

  by jinwoo Lee

'''

import scipy.io as scio
import numpy as np


## load data
train_data = scio.loadmat("C:/Users/dsa93/Desktop/github/LRFCSP_high_gamma/python_ver/data_set_IVa_al.mat")
test_data = scio.loadmat("C:/Users/dsa93/Desktop/github/LRFCSP_high_gamma/python_ver/true_labels_al.mat")


print(train_data.keys())
print(np.shape(list(train_data['cnt'])))
print(type(test_data))
# data init( Ref. BCI compettion III-IVa)



