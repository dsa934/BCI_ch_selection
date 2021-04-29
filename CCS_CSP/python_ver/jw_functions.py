
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

## CSP version data preprocessing
def data_preprocessing(eeg_train, test_data):

    # init
    train_data = eeg_train
    test_data = test_data

    # set params
    num_train_trial = len(train_data[:,:,:])
    num_test_trial = len(test_data[:,:,:])
    num_ch = len(train_data[1,:,:])
    num_time = len(train_data[1,1,:])
    Ns = 13


    ## make CSP feature from CV data
    mean_cor_list=np.zeros((num_train_trial,Ns))
    zsnorm_data = np.zeros((num_train_trial, num_ch, num_time))
    zsnorm_test = np.zeros((num_test_trial, num_ch, num_time))

    # zscore normalization(train)
    for trial_idx in range (num_train_trial):

        zsnorm_data[trial_idx,:,:] =  ( train_data[trial_idx,:,:].squeeze() - train_data[trial_idx,:,:].squeeze().mean(axis=0) ) / train_data[trial_idx,:,:].squeeze().std(axis=0)  
        # 118 x 118 
        corr_train = np.corrcoef(zsnorm_data[trial_idx,:,:])
    
        # 118 x 1
        corr_list = np.mean(corr_train, axis=1)

        # idx list
        corr_idx = [idx for idx, value in sorted(enumerate(corr_list), reverse=True, key=lambda x:x[1])[:Ns] ]

        # 각 trial별 corr value가 높은 13개의 채널집합
        mean_cor_list[trial_idx,:] = corr_idx

    # zscore normalization(test)
    for trial_idx in range(num_test_trial):

        zsnorm_test[trial_idx,:,:] = ( test_data[trial_idx,:,:].squeeze() - test_data[trial_idx, :,:].squeeze().mean(axis=0)) / test_data[trial_idx,:,:].squeeze().std(axis=0) 

    # Select Ns channels from whole channels
    whole_ch_list = np.reshape(mean_cor_list, (1, Ns*num_train_trial))
    uniq_ch_list = np.unique(whole_ch_list)
    ch_list = np.zeros( (len(uniq_ch_list),2 ))

    for idx in range(len(uniq_ch_list)):

        ch_list[idx][0] = uniq_ch_list[idx]
   
        for trial_idx in range(len(whole_ch_list.T)):
        
            if ch_list[idx][0] == whole_ch_list[0][trial_idx]:

                ch_list[idx][1] += 1

    # index 살려서 정렬
    col_name = ['ch_list', 'ch_cnt']
    pd_ch_list = pd.DataFrame(ch_list, columns=col_name)
    final_ch_list = pd_ch_list.sort_values(by = 'ch_cnt', axis=0, ascending = False)[:Ns].values


    #sel_ch = sorted(final_ch_list.T[0])
    sel_train = np.zeros((num_train_trial,Ns,num_time))
    sel_test = np.zeros((num_test_trial,Ns,num_time))
    sel_ch = final_ch_list.T[0]

      
    for ch_idx in range(len(sel_ch)):

        sel_train[:,ch_idx,:] = zsnorm_data[:,int(sel_ch[ch_idx]),:]
        sel_test[:,ch_idx,:] = zsnorm_test[:,int(sel_ch[ch_idx]),:]


    ## bandpass filtering 

    # butter = (order, [low, high], btype='band low high')
    [max_b, min_b] = butter(3, [8/50, 30/50], btype="band")

    bp_train_data = np.zeros((num_train_trial, Ns, num_time))
    bp_test_data= np.zeros((num_test_trial, Ns, num_time))


    for ch_idx in range(Ns):

        for trial_idx in range(num_train_trial):

            bp_train_data[trial_idx,ch_idx,:] = filtfilt(max_b, min_b, sel_train[trial_idx,ch_idx,:])

        for test_idx in range(num_test_trial):
        
            bp_test_data[test_idx,ch_idx,:] = filtfilt(max_b, min_b, sel_test[test_idx, ch_idx,:])

    return bp_train_data, bp_test_data


## Regurlar CSP version data preprocessing
def regular_data_preprocessing(eeg_train, test_data):

    # init
    train_data = eeg_train
    test_data = test_data

    # set params
    num_train_trial = len(train_data[:,:,:])
    num_test_trial = len(test_data[:,:,:])
    num_ch = len(train_data[1,:,:])
    num_time = len(train_data[1,1,:])
    Ns = 14
    alpha = 0.01
    beta = 0.01

    ## make CSP feature from CV data
    mean_cor_list=np.zeros((num_train_trial,Ns))
    zsnorm_data = np.zeros((num_train_trial, num_ch, num_time))
    zsnorm_test = np.zeros((num_test_trial, num_ch, num_time))

    # zscore normalization(train)
    for trial_idx in range (num_train_trial):

        zsnorm_data[trial_idx,:,:] =  ( train_data[trial_idx,:,:].squeeze() - train_data[trial_idx,:,:].squeeze().mean(axis=0) ) / train_data[trial_idx,:,:].squeeze().std(axis=0)  
        # 118 x 118 
        corr_train = np.corrcoef(zsnorm_data[trial_idx,:,:])
    
        # 118 x 1
        corr_list = np.mean(corr_train, axis=1)

        # idx list
        corr_idx = [idx for idx, value in sorted(enumerate(corr_list), reverse=True, key=lambda x:x[1])[:Ns] ]

        # 각 trial별 corr value가 높은 13개의 채널집합
        mean_cor_list[trial_idx,:] = corr_idx

    # zscore normalization(test)
    for trial_idx in range(num_test_trial):

        zsnorm_test[trial_idx,:,:] = ( test_data[trial_idx,:,:].squeeze() - test_data[trial_idx, :,:].squeeze().mean(axis=0)) / test_data[trial_idx,:,:].squeeze().std(axis=0) 

    # Select Ns channels from whole channels
    whole_ch_list = np.reshape(mean_cor_list, (1, Ns*num_train_trial))
    uniq_ch_list = np.unique(whole_ch_list)
    ch_list = np.zeros( (len(uniq_ch_list),2 ))

    for idx in range(len(uniq_ch_list)):

        ch_list[idx][0] = uniq_ch_list[idx]
   
        for trial_idx in range(len(whole_ch_list.T)):
        
            if ch_list[idx][0] == whole_ch_list[0][trial_idx]:

                ch_list[idx][1] += 1

    # index 살려서 정렬
    col_name = ['ch_list', 'ch_cnt']
    pd_ch_list = pd.DataFrame(ch_list, columns=col_name)
    final_ch_list = pd_ch_list.sort_values(by = 'ch_cnt', axis=0, ascending = False)[:Ns].values


    #sel_ch = sorted(final_ch_list.T[0])
    sel_train = np.zeros((num_train_trial,Ns,num_time))
    sel_test = np.zeros((num_test_trial,Ns,num_time))
    sel_ch = final_ch_list.T[0]
    
      
    for ch_idx in range(len(sel_ch)):

        sel_train[:,ch_idx,:] = zsnorm_data[:,int(sel_ch[ch_idx]),:]
        sel_test[:,ch_idx,:] = zsnorm_test[:,int(sel_ch[ch_idx]),:]


    ## bandpass filtering 

    # butter = (order, [low, high], btype='band low high')
    [max_b, min_b] = butter(3, [8/50, 30/50], btype="band")

    bp_train_data = np.zeros((num_train_trial, Ns, num_time))
    bp_test_data= np.zeros((num_test_trial, Ns, num_time))


    for ch_idx in range(Ns):

        for trial_idx in range(num_train_trial):

            bp_train_data[trial_idx,ch_idx,:] = filtfilt(max_b, min_b, sel_train[trial_idx,ch_idx,:])

        for test_idx in range(num_test_trial):
        
            bp_test_data[test_idx,ch_idx,:] = filtfilt(max_b, min_b, sel_test[test_idx, ch_idx,:])

    return bp_train_data, bp_test_data, alpha, beta, Ns



def jw_csp(input_data):

    # set args
    num_trial = len(input_data)
    num_ch = len(input_data[1,:,:])
    num_time = len(input_data[1,1,:])
        
    # seperate label
    RH_data = input_data[:int(num_trial/2), :,:]
    RF_data = input_data[int(num_trial/2):num_trial,:,:]

    # get normalized cov matrix
    # (X'X) / tr(X'X)
    RH_cov = np.zeros((int(num_trial/2),num_ch, num_ch))
    RF_cov = np.zeros((int(num_trial/2),num_ch, num_ch))

    for trial_idx in range(int(num_trial/2)):
        RH_cov[trial_idx,:,:] = np.matmul(RH_data[trial_idx,:,:], RH_data[trial_idx,:,:].T) / np.trace(np.matmul(RH_data[trial_idx,:,:], RH_data[trial_idx,:,:].T))
        RF_cov[trial_idx,:,:] = np.matmul(RF_data[trial_idx,:,:], RF_data[trial_idx,:,:].T) / np.trace(np.matmul(RF_data[trial_idx,:,:], RF_data[trial_idx,:,:].T))


    mean_RH = np.mean(RH_cov,0)
    mean_RF = np.mean(RF_cov,0)
    cov_sum = mean_RH + mean_RF 
    
    # eigen value decomposition
    eigen_val, eigen_vec = np.linalg.eig(cov_sum)

    # 1-dim val vec -> 2-dim matrix (by diag)
    eigen_val = np.diag(eigen_val)

    # 백색화변환행렬
    P =  np.matmul(np.sqrt(np.linalg.inv(eigen_val)),eigen_vec.T )
    s_RH = np.matmul(np.matmul(P, mean_RH),P.T)
    s_RF = np.matmul(np.matmul(P,mean_RF),P.T)

    RH_eigen_val, RH_eigen_vec = np.linalg.eig(s_RH)
    RF_eigen_val, RF_eigen_vec = np.linalg.eig(s_RF)

    RH_eigen_val = np.diag(RH_eigen_val)
    RF_eigen_val = np.diag(RF_eigen_val)

    max_feature = np.matmul(P.T, RH_eigen_vec)
    min_feature = max_feature.T

    BB = np.matmul(np.matmul(max_feature.T,mean_RH ),max_feature)
    BBB = np.matmul(np.matmul(max_feature.T,mean_RF ),max_feature)
    
    RH_max_val = np.argmax(np.diag(BB))
    RF_max_val = np.argmax(np.diag(BBB))

    # csp_filter = 2,13
    csp_filter = np.zeros((2,13))
    csp_filter[0] = min_feature[RH_max_val]
    csp_filter[1] = min_feature[RF_max_val]

    return csp_filter


def jw_rcsp(input_data, alpha, beta, Ns):

    # set args
    num_trial = len(input_data)
    num_ch = len(input_data[1,:,:])
    num_time = len(input_data[1,1,:])
        
    # seperate label
    RH_data = input_data[:int(num_trial/2), :,:]
    RF_data = input_data[int(num_trial/2):num_trial,:,:]

    # get normalized cov matrix
    # (X'X) / tr(X'X)
    # 기존의 csp와 다르게 Regularized CSP를 사용
    # 다른 subject값을 이용한 공분산을 만들어야 하지만 해당 논문에선 pair-wise한 값으로 대체
    RH_cov = np.zeros((int(num_trial/2),num_ch, num_ch))
    RF_cov = np.zeros((int(num_trial/2),num_ch, num_ch))

    RH_hat_cov = np.zeros((int(num_trial/2),num_ch, num_ch))
    FH_hat_cov = np.zeros((int(num_trial/2),num_ch, num_ch))


    for trial_idx in range(int(num_trial/2)):
        RH_cov[trial_idx,:,:] = np.matmul(RH_data[trial_idx,:,:], RH_data[trial_idx,:,:].T) / np.trace(np.matmul(RH_data[trial_idx,:,:], RH_data[trial_idx,:,:].T))
        RF_cov[trial_idx,:,:] = np.matmul(RF_data[trial_idx,:,:], RF_data[trial_idx,:,:].T) / np.trace(np.matmul(RF_data[trial_idx,:,:], RF_data[trial_idx,:,:].T))

        RH_hat_cov = np.cov(RH_data[trial_idx,:,:])
        RF_hat_cov = np.cov(RF_data[trial_idx,:,:])


    mean_RH = np.mean(RH_cov,0)
    mean_RF = np.mean(RF_cov,0)

    mean_hat_RH = np.mean(RH_hat_cov,0)
    mean_hat_RF = np.mean(RF_hat_cov,0)


    # 정규화 평군공분산 행렬
    P_RH = ((1-alpha) * mean_RH + alpha * mean_hat_RH) / num_trial
    P_RF = ((1-alpha) * mean_RF + alpha * mean_hat_RF) / num_trial

    Iden = np.eye(Ns,Ns)
    Q_RH = (1- beta) * P_RH + (beta/num_trial) * np.trace(P_RH) * Iden
    Q_RF = (1- beta) * P_RF + (beta/num_trial) * np.trace(P_RF) * Iden
        
    avg_cov = Q_RH + Q_RF
    eigen_val, eigen_vec = np.linalg.eig(avg_cov)
    eigen_val = np.diag(eigen_val)

    # 정규화 평균공분산의 백색화 변환행렬
    P =  np.matmul(np.sqrt(np.linalg.inv(eigen_val)),eigen_vec.T )

    s_RH = np.matmul(np.matmul(P, Q_RH),P.T)
    s_RF = np.matmul(np.matmul(P,Q_RF),P.T)


    RH_eigen_val, RH_eigen_vec = np.linalg.eig(s_RH)
    RF_eigen_val, RF_eigen_vec = np.linalg.eig(s_RF)

    RH_eigen_val = np.diag(RH_eigen_val)
    RF_eigen_val = np.diag(RF_eigen_val)

    max_feature = np.matmul(P.T, RH_eigen_vec)
    min_feature = max_feature.T

    BB = np.matmul(np.matmul(max_feature.T,mean_RH ),max_feature)
    BBB = np.matmul(np.matmul(max_feature.T,mean_RF ),max_feature)

    #BB = max_feature.T * mean_RH * max_feature
    #BBB = max_feature.T * mean_RF * max_feature
    
    RH_max_val = np.argmax(np.diag(BB))
    RF_max_val = np.argmax(np.diag(BBB))

    # csp_filter = 2,13
    csp_filter = np.zeros((2,Ns))
    csp_filter[0] = min_feature[RH_max_val]
    csp_filter[1] = min_feature[RF_max_val]

    return csp_filter


def csp_feature(csp_filter, input_data, ck_flag):

    #set args
    num_trial = len(input_data)
    num_time = len(input_data[1,1,:])


    if ck_flag =='train' :
        print("configurate train data ...")
        max_var_eeg = np.zeros((num_trial,1))
        min_var_eeg = np.zeros((num_trial,1))

        for train_idx in range(num_trial):

            bp_csp_train = np.matmul( csp_filter, input_data[train_idx,:,:])

            max_var_eeg[train_idx] = np.log10( np.var(bp_csp_train[0,:]) / (np.var(bp_csp_train[0,:]) + np.var(bp_csp_train[1,:])) )
            min_var_eeg[train_idx] = np.log10( np.var(bp_csp_train[1,:]) / (np.var(bp_csp_train[0,:]) + np.var(bp_csp_train[1,:])) )


        csp_feature_train = np.zeros((num_trial,2))
        csp_feature_train[:,0] = max_var_eeg.squeeze()
        csp_feature_train[:,1] = min_var_eeg.squeeze()
        print("complete.")

        return csp_feature_train

    else :
        print("configurate test data...")
        max_test_var_eeg = np.zeros((num_trial,1))
        min_test_var_eeg = np.zeros((num_trial,1))

        for test_idx in range(num_trial):

            bp_csp_test = np.matmul( csp_filter, input_data[test_idx,:,:])

            max_test_var_eeg[test_idx] = np.log10( np.var(bp_csp_test[0,:]) / (np.var(bp_csp_test[0,:]) + np.var(bp_csp_test[1,:])) )
            min_test_var_eeg[test_idx] = np.log10( np.var(bp_csp_test[1,:]) / (np.var(bp_csp_test[0,:]) + np.var(bp_csp_test[1,:])) )


        csp_feature_test = np.zeros((num_trial,2))
        csp_feature_test[:,0] = max_test_var_eeg.squeeze()
        csp_feature_test[:,1] = min_test_var_eeg.squeeze()
        print("complete.")

        return csp_feature_test

