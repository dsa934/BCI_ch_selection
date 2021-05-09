'''

Restoration of papers related to the BCI channels.

"Optimal channel selection using covariance matrix and cross-combining"

by jinwoo Lee


'''

import scipy.io as scio
import jw_functions as jwf
import numpy as np

## load data
# train / test data shape : [252,118,501] / [28,118,501]
train_data = scio.loadmat("C:/Users/dsa93/Desktop/compare_paper_other_algorithm/park_optimal_channelsel_conference/BCI_com_iii_Iva/data_100Hz/al/train/train_1")
train_label = scio.loadmat("C:/Users/dsa93/Desktop/compare_paper_other_algorithm/park_optimal_channelsel_conference/BCI_com_iii_Iva/data_100Hz/al/train/train_label_1")
test_data = scio.loadmat("C:/Users/dsa93/Desktop/compare_paper_other_algorithm/park_optimal_channelsel_conference/BCI_com_iii_Iva/data_100Hz/al/test/test_1")
test_label = scio.loadmat("C:/Users/dsa93/Desktop/compare_paper_other_algorithm/park_optimal_channelsel_conference/BCI_com_iii_Iva/data_100Hz/al/test/test_label_1")

train_data = train_data['train_data']
train_label = train_label['train_label']
test_data = test_data['test_data']
test_label = test_label['test_label']

# channel selection based on covariance matrix
target_ch, sort_region, trial_sub_region = jwf.covariance_channel_selection(train_data)

# get cross-combining region features
cc_train_1, cc_test_1 = jwf.cross_combining_feature(train_data, test_data, target_ch, sort_region, trial_sub_region, 4, 8 )   # 4-8Hz
cc_train_2, cc_test_2 = jwf.cross_combining_feature(train_data, test_data, target_ch, sort_region, trial_sub_region, 8, 12 )  # 8-12Hz
cc_train_3, cc_test_3 = jwf.cross_combining_feature(train_data, test_data, target_ch, sort_region, trial_sub_region, 12, 16 ) # 12-16Hz
cc_train_4, cc_test_4 = jwf.cross_combining_feature(train_data, test_data, target_ch, sort_region, trial_sub_region, 16, 20 ) # 16-20Hz
cc_train_5, cc_test_5 = jwf.cross_combining_feature(train_data, test_data, target_ch, sort_region, trial_sub_region, 20, 24 ) # 20-24Hz
cc_train_6, cc_test_6 = jwf.cross_combining_feature(train_data, test_data, target_ch, sort_region, trial_sub_region, 24, 28 ) # 24-28Hz
cc_train_7, cc_test_7 = jwf.cross_combining_feature(train_data, test_data, target_ch, sort_region, trial_sub_region, 28, 32 ) # 28-32Hz

# gather cc features
region_train_data = np.concatenate((cc_train_1,cc_train_2,cc_train_3,cc_train_4,cc_train_5,cc_train_6,cc_train_7), axis=2)
region_test_data = np.concatenate((cc_test_1,cc_test_2,cc_test_3,cc_test_4,cc_test_5,cc_test_6,cc_test_7), axis=2)

# select FBCSP features by MIBIF ( mutual information based individual feature algorithm )
sel_train, sel_test = jwf.mutual_information(region_train_data, train_label, region_test_data)

# training
from sklearn.svm import SVC

num_region = len(sel_train)
accy = np.zeros((num_region))
# shuffle test data for fair
tmp = np.random.permutation(28)

for region_idx in range(num_region):

    svm_model = SVC(kernel='rbf', C=1, gamma=10)
    svm_model.fit(sel_train[region_idx,:,:], train_label[0])
    y_pred = svm_model.predict(sel_test[region_idx,tmp,:])

    for ran_idx in range(len(tmp)):

        test_label[0][ran_idx] =  test_label[0][tmp[ran_idx]]

    print("==============")
    print("prediction : ",y_pred)
    print("true_label : ", test_label[0])

    ck_cnt = 0
    for idx in range(len(y_pred)):

        if y_pred[idx] == test_label[0][idx]:
            ck_cnt +=1

    print("Accuracy: ", ck_cnt/len(y_pred)*100, "%")

    accy[region_idx] = ck_cnt/len(y_pred)*100

print(np.max(accy))