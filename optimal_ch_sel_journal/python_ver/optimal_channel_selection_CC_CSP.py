'''

Restoration of papers related to the BCI channels.

"Optimal Channel Selection Using Correlation Coefficient for CSP Based EEG Classification"

by jinwoo Lee

'''
import scipy.io as scio
import jw_functions as jwf
import numpy as np

# load data
train_data = scio.loadmat("C:/Users/dsa93/Desktop/compare_paper_other_algorithm/park_optimal_cs_journal/data100Hz/ay/train/train_10")
train_label = scio.loadmat("C:/Users/dsa93/Desktop/compare_paper_other_algorithm/park_optimal_cs_journal/data100Hz/ay/train/train_label_10")
test_data = scio.loadmat("C:/Users/dsa93/Desktop/compare_paper_other_algorithm/park_optimal_cs_journal/data100Hz/ay/test/test_10")
test_label = scio.loadmat("C:/Users/dsa93/Desktop/compare_paper_other_algorithm/park_optimal_cs_journal/data100Hz/ay/test/test_label_10")

# data shape = [252,118,501], [28,118,50]
train_data = train_data['train_data']
train_label = train_label['train_label']
test_data = test_data['test_data']
test_label = test_label['test_label']

# configurate distinctive channels based on correlation coefficient using t-statistic
rh_eeg, rf_eeg, support_channel_group = jwf.config_distinctive_channels(train_data)

# get fisher socre & FBCSP features
gfbcsp_train1, gfbcsp_test1 = jwf.fisher_fbcsp_feature(rh_eeg, rf_eeg, test_data, support_channel_group, 4, 8)  # 4-8Hz
gfbcsp_train2, gfbcsp_test2 = jwf.fisher_fbcsp_feature(rh_eeg, rf_eeg, test_data, support_channel_group, 8, 12)  # 8-12Hz
gfbcsp_train3, gfbcsp_test3 = jwf.fisher_fbcsp_feature(rh_eeg, rf_eeg, test_data, support_channel_group, 12, 16)  # 12-16Hz
gfbcsp_train4, gfbcsp_test4 = jwf.fisher_fbcsp_feature(rh_eeg, rf_eeg, test_data, support_channel_group, 16, 20)  # 16-20Hz
gfbcsp_train5, gfbcsp_test5 = jwf.fisher_fbcsp_feature(rh_eeg, rf_eeg, test_data, support_channel_group, 20, 24)  # 20-24Hz
gfbcsp_train6, gfbcsp_test6 = jwf.fisher_fbcsp_feature(rh_eeg, rf_eeg, test_data, support_channel_group, 24, 28)  # 24-28Hz
gfbcsp_train7, gfbcsp_test7 = jwf.fisher_fbcsp_feature(rh_eeg, rf_eeg, test_data, support_channel_group, 28, 32)  # 28-32Hz
gfbcsp_train8, gfbcsp_test8 = jwf.fisher_fbcsp_feature(rh_eeg, rf_eeg, test_data, support_channel_group, 32, 36)  # 32-36Hz

# gathering FBCSP features
group_fbcsp_train = np.concatenate((gfbcsp_train1,gfbcsp_train2,gfbcsp_train3,gfbcsp_train4,gfbcsp_train5,gfbcsp_train6,gfbcsp_train7,gfbcsp_train8), axis=2)
group_fbcsp_test = np.concatenate((gfbcsp_test1,gfbcsp_test2,gfbcsp_test3,gfbcsp_test4,gfbcsp_test5,gfbcsp_test6,gfbcsp_test7,gfbcsp_test8), axis=2)

# Mutual information
train_data_set, test_data_set = jwf.mutual_information(group_fbcsp_train, group_fbcsp_test, train_label)

# training
from sklearn.svm import SVC

svm_model = SVC(kernel='rbf', C=1, gamma=10)
svm_model.fit(train_data_set,train_label[0])
y_pred = svm_model.predict(test_data_set)

tmp = np.random.permutation(28)
test_data_set = test_data_set[tmp,:]

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