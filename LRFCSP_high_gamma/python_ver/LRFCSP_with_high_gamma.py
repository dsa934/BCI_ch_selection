
'''
 " Motor Imagery Classification using local region CSP features with high-gamma band " 

  by jinwoo Lee

'''

import scipy.io as scio
import numpy as np
import jw_functions as jwf

## load data
train_data = scio.loadmat("C:/Users/dsa93/Desktop/job_preparing/github_code/python_ver/LRFCSP_with_high_gamma/LRFCSP_with_high_gamma/data_set_IVa_al.mat")
true_label = scio.loadmat("C:/Users/dsa93/Desktop/job_preparing/github_code/python_ver/LRFCSP_with_high_gamma/LRFCSP_with_high_gamma/true_labels_al.mat")
true_label = true_label['true_y']


## data preprocessing
rh_data, rf_data, test_data, rh_len, rf_len, test_len = jwf.data_preprocessing(train_data)

'''
 - 필자가 해당 논문을 작성할때 최초 코딩은 matlab으로 진행하였으며,
 - 실험에 참고한 데이터셋(BCI competition III-IVa) 홈페이지에는 데이터에 대하여 
 - cnt = 0.1 * double(cnt)처리에 대한 안내가 나와있으나 python 에서는 double 형 데이터타입을 지원하지 않기 떄문에
 - 이를 무시하고 python version으로 코딩을 진행하였으나, 결과가 동일하게 나오지 않아 모든 코드를 일일히 체크한 결과
 - 결과적으로 double 타입의 데이터형 전환에 유무가 결과에 영향을 미치는것을 확인하였다.
 - 아래는 matlab version 코드에서 최초 전처리 부분 진행 후 데이터만 가져온 코드이다.

rh_data = scio.loadmat("C:/Users/dsa93/Desktop/job_preparing/github_code/python_ver/LRFCSP_with_high_gamma/LRFCSP_with_high_gamma/bp_rh.mat")
rf_data = scio.loadmat("C:/Users/dsa93/Desktop/job_preparing/github_code/python_ver/LRFCSP_with_high_gamma/LRFCSP_with_high_gamma/bp_rf.mat")
test_data = scio.loadmat("C:/Users/dsa93/Desktop/job_preparing/github_code/python_ver/LRFCSP_with_high_gamma/LRFCSP_with_high_gamma/bp_test.mat")

rh_data = rh_data['bp_rh']
rf_data = rf_data['bp_rf'
test_data = test_data['bp_test']
'''

# set test_data's true labels
true_label = true_label[0][280-test_len:]

## Configurate local region CSP feature (LRCSP) 
# region 1 ~ 18 
rh_lrcsp1, rf_lrcsp1, test_lrcsp1,difference1 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 0,1,3,5)                 # ch_list : 1, 2, 4, 6
rh_lrcsp2, rf_lrcsp2, test_lrcsp2,difference2 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 1,0,3,2)                 # ch_list : 2, 1, 4, 3
rh_lrcsp3, rf_lrcsp3, test_lrcsp3,difference3 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 1,4,3,8,2)               # ch_list : 2, 5, 4, 9, 3
rh_lrcsp4, rf_lrcsp4, test_lrcsp4,difference4 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 0,1,2,4,5,6,3)           # ch_list : 1, 2, 3, 5, 6, 7, 4
rh_lrcsp5, rf_lrcsp5, test_lrcsp5,difference5 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 4,2,3,6,8)               # ch_list : 5, 3, 4, 7, 9
#6
rh_lrcsp6, rf_lrcsp6, test_lrcsp6,difference6 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 5,0,3,6,7)               # ch_list : 6, 1, 4, 7, 8
rh_lrcsp7, rf_lrcsp7, test_lrcsp7,difference7 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 6,3,4,5,7,8,9)           # ch_list : 7, 4, 5, 6, 8, 9, 10
rh_lrcsp8, rf_lrcsp8, test_lrcsp8,difference8 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 7,5,6,9)                 # ch_list : 8, 6, 7, 10
rh_lrcsp9, rf_lrcsp9, test_lrcsp9,difference9 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 8,4,10,9)                # ch_list : 9, 5, 11, 10
rh_lrcsp10, rf_lrcsp10, test_lrcsp10,difference10 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 9,8,6,15,7,17)           # ch_list : 10, 9, 7, 16, 8, 18
#11
rh_lrcsp11, rf_lrcsp11, test_lrcsp11,difference11 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 10,8,11,13,15)           # ch_list : 11, 9, 12, 14, 16
rh_lrcsp12, rf_lrcsp12, test_lrcsp12,difference12 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 10,8,12,13,11)           # ch_list : 11, 9, 13, 14, 12
rh_lrcsp13, rf_lrcsp13, test_lrcsp13,difference13 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 14,11,12,13)             # ch_list : 15, 12, 13, 14
rh_lrcsp14, rf_lrcsp14, test_lrcsp14,difference14 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 10,11,12,13,14,15,16)    # ch_list : 11, 12, 13, 14, 15, 16, 17
rh_lrcsp15, rf_lrcsp15, test_lrcsp15,difference15 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 14,12,13,16)             # ch_list : 15, 13, 14, 17
#16
rh_lrcsp16, rf_lrcsp16, test_lrcsp16,difference16 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 15,13,16,17,10,8,9)      # ch_list : 16, 14, 17, 18, 11, 9, 10
rh_lrcsp17, rf_lrcsp17, test_lrcsp17,difference17 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 13,14,16,17,15)          # ch_list : 14, 15, 17, 18, 16
rh_lrcsp18, rf_lrcsp18, test_lrcsp18,difference18 = jwf.calculate_lrcsp_feature(rh_data, rf_data, test_data, rh_len, rf_len, test_len, 17,9,16,15)              # ch_list : 18, 10, 17, 16


## sorting eigenvalue score
diff_set = difference1,difference2,difference3,difference4,difference5,difference6,difference7,difference8,difference9,difference10,difference11,difference12,difference13,difference14,difference15,difference16,difference17,difference18
diff_idx = np.argsort(diff_set)

# LRCSP features 
# [112, 36 ] , [56, 36]
RH_feature = np.concatenate((rh_lrcsp1,rh_lrcsp2,rh_lrcsp3,rh_lrcsp4,rh_lrcsp5,rh_lrcsp6,rh_lrcsp7,rh_lrcsp8,rh_lrcsp9,rh_lrcsp10,rh_lrcsp11,rh_lrcsp12,rh_lrcsp13,rh_lrcsp14,rh_lrcsp15,rh_lrcsp16,rh_lrcsp17,rh_lrcsp18),axis=1)
RF_feature = np.concatenate((rf_lrcsp1,rf_lrcsp2,rf_lrcsp3,rf_lrcsp4,rf_lrcsp5,rf_lrcsp6,rf_lrcsp7,rf_lrcsp8,rf_lrcsp9,rf_lrcsp10,rf_lrcsp11,rf_lrcsp12,rf_lrcsp13,rf_lrcsp14,rf_lrcsp15,rf_lrcsp16,rf_lrcsp17,rf_lrcsp18),axis=1)
test_feature = np.concatenate((test_lrcsp1,test_lrcsp2,test_lrcsp3,test_lrcsp4,test_lrcsp5,test_lrcsp6,test_lrcsp7,test_lrcsp8,test_lrcsp9,test_lrcsp10,test_lrcsp11,test_lrcsp12,test_lrcsp13,test_lrcsp14,test_lrcsp15,test_lrcsp16,test_lrcsp17,test_lrcsp18), axis=1)

# training
from sklearn.svm import SVC

accy = np.zeros((18,1))
for eig_idx in range(len(diff_idx)):

    feat_rh = np.zeros((rh_len,2))
    feat_rf = np.zeros((rf_len,2))
    feat_test = np.zeros((test_len,2))

    for difference_idx in range(eig_idx, len(diff_idx)):
        # python idx가 0부터 시작하기 떄문에 발생하는 예외처리
        # => first channel이 0번 채널로 인식되서 for문내에서 데이터 shape이 변해버리는 이슈가 있었음 (112,1) -> (112,0)
        # 따라서 idx 관계식도 변화함 (2n : 2n+2 n>=2) 
        if diff_idx[difference_idx] == 0:
            
            tmp_rh = RH_feature[:, 0 : 2]
            tmp_rf = RF_feature[:, 0 : 2]
            tmp_test = test_feature[:, 0 : 2]

        else:

            tmp_rh = RH_feature[:, 2*diff_idx[difference_idx] : 2*diff_idx[difference_idx]+2]
            tmp_rf = RF_feature[:, 2*diff_idx[difference_idx] : 2*diff_idx[difference_idx]+2]
            tmp_test = test_feature[:, 2*diff_idx[difference_idx] : 2*diff_idx[difference_idx]+2]


        feat_rh = np.concatenate((feat_rh,tmp_rh), axis=1)
        feat_rf = np.concatenate((feat_rf,tmp_rf), axis=1)
        feat_test = np.concatenate((feat_test,tmp_test), axis=1)

    # 제일 앞부분 garbage area 제거 
    feat_rh = feat_rh[:,2:]
    feat_rf = feat_rf[:,2:]
    feat_test = feat_test[:,2:]


    # whole train data = [224, n_feature]
    # whole true label = [224,1]
    whole_train_data = np.concatenate((feat_rh, feat_rf),axis=0)
    whole_train_label = np.concatenate((np.ones((rh_len,1)), np.ones((rf_len,1))+1), axis=0)

    svm_model = SVC(kernel='rbf', C=1, gamma=10)
    svm_model.fit(whole_train_data,whole_train_label)
    y_pred = svm_model.predict(feat_test)

    print("==============")
    print("prediction : ",y_pred)
    print("true_label : ", true_label)

    ck_cnt = 0
    for idx in range(len(y_pred)):

        if y_pred[idx] == true_label[idx]:

            ck_cnt +=1 

    print("Accuracy : ", ck_cnt/len(y_pred) * 100, "%")
    accy[eig_idx] = ck_cnt/len(y_pred) * 100
   
print("total accuracy", accy)
