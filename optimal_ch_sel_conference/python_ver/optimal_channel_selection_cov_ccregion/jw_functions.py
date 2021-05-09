
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.metrics import mutual_info_score


def covariance_channel_selection(train_data):
    
    #set init args
    num_train = len(train_data)
    num_ch = len(train_data[0])
    num_time = len(train_data[0][0])

    
    # bandpass filtering
    bp_train = np.zeros((num_train, num_ch, int((num_time-1)/2)))
    [max_boundary, min_boundary] = butter(4, [4/50, 32/50], btype='band')
    
    for ch_idx in range(num_ch):

        for trial_idx in range(num_train):

            tmp = filtfilt(max_boundary, min_boundary, train_data[trial_idx,ch_idx,:])
            bp_train[trial_idx, ch_idx,:] = tmp[50:300]


    # seperate class 
    train_rh = bp_train[:int(num_train/2),:,:]
    train_rf = bp_train[int(num_train/2):,:,:]

    # covariance matrix (XX') /tr(XX')
    cov_rh = np.zeros((int(num_train/2), num_ch, num_ch))
    cov_rf = np.zeros((int(num_train/2), num_ch, num_ch))
    
    for trial_idx in range(int(num_train/2)):

        cov_rh[trial_idx,:,:] = np.matmul(train_rh[trial_idx,:,:] ,train_rh[trial_idx,:,:].T) / np.trace(np.matmul(train_rh[trial_idx,:,:] ,train_rh[trial_idx,:,:].T))
        cov_rf[trial_idx,:,:] = np.matmul(train_rf[trial_idx,:,:] ,train_rf[trial_idx,:,:].T) / np.trace(np.matmul(train_rf[trial_idx,:,:] ,train_rf[trial_idx,:,:].T))
  
    mean_rh = np.mean(cov_rh,0)
    mean_rf = np.mean(cov_rf,0)

    # calculate channel score matrix V
    V = np.abs(mean_rh - mean_rf)

    # find channel list(H) based score matrix V
    H = list()
    max_value = 0
    for ch_idx in range(num_ch):

        if len(np.unique(H)) <= 20:

            max_value = np.max(V)
            row, col = np.where(V == max_value)
            H.extend(row)
            V[row, col]=0

    # sub_ch_region shape : [2,21] row_0 : channel_number, row_1: channel_count
    sub_ch_region = np.unique(H,return_index = True)
    idx_sort = sub_ch_region[1].argsort()[::-1]
    
    sort_region = sub_ch_region[0]
    sort_region = sort_region[idx_sort]

    # target channel : 가장 중복 회수가 높은 채널
    target_ch = sort_region[0]
    
    # num_sub_region : 8 ( ref papers)
    num_sub_region = 8
    trial_sub_region = len(sub_ch_region[0]) - num_sub_region 

    return target_ch, sort_region, trial_sub_region


def cross_combining_feature(train_data, test_data, target_ch, sort_region, trial_sub_region, min_fs, max_fs ):

    #set init args
    num_train = len(train_data)
    num_test = len(test_data)
    num_ch = len(train_data[0])
    num_time = len(train_data[0][0])
    num_region_inner_ch = 8

    # bp filtering
    bp_train = np.zeros((num_train, num_ch, int((num_time-1)/2)))
    bp_test = np.zeros((num_test, num_ch, int((num_time-1)/2)))
    [max_boundary, min_boundary] = butter(4, [min_fs/50, max_fs/50], btype='band')

    for ch_idx in range(num_ch):

        for train_idx in range(num_train):

            tmp = filtfilt(max_boundary,min_boundary, train_data[train_idx,ch_idx,:])
            bp_train[train_idx, ch_idx, :] = tmp[50:300]

        for test_idx in range(num_test):

            tmp = filtfilt(max_boundary,min_boundary, test_data[test_idx,ch_idx,:])
            bp_test[test_idx, ch_idx, :] = tmp[50:300]

    # seperate class
    train_rh = bp_train[:int(num_train/2),:,:]
    train_rf = bp_train[int(num_train/2):,:,:]

    # cross combined channels
    cross_region_rh = np.zeros((trial_sub_region, int(num_train/2), num_region_inner_ch, int((num_time-1)/2) ))
    cross_region_rf = np.zeros((trial_sub_region, int(num_train/2), num_region_inner_ch, int((num_time-1)/2) ))
    cross_region_test = np.zeros((trial_sub_region, num_test, num_region_inner_ch, int((num_time-1)/2) ))

    for region_idx in range(trial_sub_region):

        # case : right hand
        cross_region_rh[region_idx,:,0,:] = train_rh[:,target_ch,:]
        cross_region_rh[region_idx,:,1 : num_region_inner_ch,:] = train_rh[:,sort_region[region_idx : region_idx + num_region_inner_ch-1],:]

        # case : right foot
        cross_region_rf[region_idx,:,0,:] = train_rf[:,target_ch,:]
        cross_region_rf[region_idx,:,1 : num_region_inner_ch,:] = train_rf[:,sort_region[region_idx : region_idx + num_region_inner_ch-1],:]

        # case : test data
        cross_region_test[region_idx,:,0,:] = bp_test[:,target_ch,:]
        cross_region_test[region_idx,:,1 : num_region_inner_ch,:] = bp_test[:,sort_region[region_idx : region_idx + num_region_inner_ch-1],:]


    # calculate cross-combining csp filter
    # filter shape : [ 13, 2, 8]
    region_csp_filter = cross_cspfilter(cross_region_rh, cross_region_rf)
    
    # cross-combining csp features
    region_csp_feature_rh = np.zeros((trial_sub_region, int(num_train/2),2))
    region_csp_feature_rf = np.zeros((trial_sub_region, int(num_train/2),2))
    region_csp_feature_test = np.zeros((trial_sub_region, num_test, 2))

    for region_idx in range(trial_sub_region):

        # case : train data
        for train_idx in range(int(num_train/2)):

            temp_rh = np.matmul(region_csp_filter, cross_region_rh[region_idx, train_idx, :,:])
            temp_rf = np.matmul(region_csp_filter, cross_region_rf[region_idx, train_idx, :,:])

            rh_max = np.log10( np.var(temp_rh[0]))
            rh_min = np.log10( np.var(temp_rh[1]))

            rf_max = np.log10( np.var(temp_rf[0]))
            rf_min = np.log10( np.var(temp_rf[1]))

            region_csp_feature_rh[region_idx, train_idx,0] = rh_max
            region_csp_feature_rh[region_idx, train_idx,1] = rh_min

            region_csp_feature_rf[region_idx, train_idx,0] = rf_max
            region_csp_feature_rf[region_idx, train_idx,1] = rf_min

        # case : test data
        for test_idx in range(num_test):

            temp_test = np.matmul(region_csp_filter, cross_region_test[region_idx, test_idx, :,:])

            test_max = np.log10( np.var(temp_test[0]))
            test_min = np.log10( np.var(temp_test[1]))

            region_csp_feature_test[region_idx, test_idx,0] = test_max
            region_csp_feature_test[region_idx, test_idx,1] = test_min

    # summation groups
    final_train_data = np.concatenate((region_csp_feature_rh, region_csp_feature_rf), axis=1)

    return final_train_data, region_csp_feature_test
    

def cross_cspfilter(cross_region_rh, cross_region_rf):
    
    # set init args
    # input data shape : [13, 126, 8, 250]
    num_region = len(cross_region_rh)
    num_trial = len(cross_region_rh[0])
    num_ch = len(cross_region_rh[0][0])
    num_time = len(cross_region_rh[0][0][0])
    

    # calculate region cross csp
    cov_rh = np.zeros((num_trial, num_ch, num_ch))
    cov_rf = np.zeros((num_trial, num_ch, num_ch ))

    region_csp_filter = np.zeros((num_region, 2, num_ch))

    for region_idx in range(num_region):

        for trial_idx in range(num_trial):
            
            cov_rh[trial_idx, :,: ] = np.matmul(cross_region_rh[region_idx, trial_idx,:,:], cross_region_rh[region_idx, trial_idx,:,:].T) / np.trace(np.matmul(cross_region_rh[region_idx, trial_idx,:,:], cross_region_rh[region_idx, trial_idx,:,:].T))
            cov_rf[trial_idx, :,: ] = np.matmul(cross_region_rf[region_idx, trial_idx,:,:], cross_region_rf[region_idx, trial_idx,:,:].T) / np.trace(np.matmul(cross_region_rf[region_idx, trial_idx,:,:], cross_region_rf[region_idx, trial_idx,:,:].T))

        mean_rh = np.mean(cov_rh, axis=0)
        mean_rf = np.mean(cov_rf, axis=0)
        cov_sum = mean_rh + mean_rf
            
        # eigen value decomposition
        eigen_val, eigen_vec = np.linalg.eig(cov_sum)

        # 1-dim val vec -> 2-dim matrix (by diag)
        eigen_val = np.diag(eigen_val)

        # 백색화변환행렬
        P =  np.matmul(np.sqrt(np.linalg.inv(eigen_val)),eigen_vec.T )
        s_RH = np.matmul(np.matmul(P, mean_rh),P.T)
        s_RF = np.matmul(np.matmul(P,mean_rf),P.T)

        # Nan value exception for eigenvalue decompostion
        if np.isnan(s_RH).any() or np.isnan(s_RF).any():
            s_RH[np.isnan(s_RH)]=0
            s_RF[np.isnan(s_RF)]=0

        RH_eigen_val, RH_eigen_vec = np.linalg.eig(s_RH)
        RF_eigen_val, RF_eigen_vec = np.linalg.eig(s_RF)

        max_feature = np.matmul(P.T, RH_eigen_vec)
        min_feature = max_feature.T

        BB = np.matmul(np.matmul(max_feature.T,mean_rh ),max_feature)
        BBB = np.matmul(np.matmul(max_feature.T,mean_rf ),max_feature)
    
        RH_max_val = np.argmax(np.diag(BB))
        RF_max_val = np.argmax(np.diag(BBB))

        # csp_filter [2,num_ch]
        region_csp_filter[region_idx,0,:] = min_feature[RH_max_val]
        region_csp_filter[region_idx,1,:] = min_feature[RF_max_val]

    return region_csp_filter


def mutual_information(region_train_data, train_label, region_test_data):

    # set init args
    num_region = len(region_train_data)
    num_trial = len(region_train_data[0])
    num_test = len(region_test_data[0])
    num_fb = len(region_train_data[0][0])
    train_label = train_label[0]

    # nan exceptions
    if np.isnan(region_train_data).any() :
        region_train_data[np.isnan(region_train_data)]=0

    # calculate mutual information
    mi_fb = np.zeros((num_region, num_fb))
    idx = np.zeros((num_region, num_fb))
    for region_idx in range(num_region):

        for fb_idx in range(num_fb):

            mi_fb[region_idx, fb_idx] = mutual_info_score(train_label, region_train_data[region_idx, :, fb_idx].squeeze())
    
        idx[region_idx,:num_fb] = np.argsort((mi_fb[region_idx,:]), axis=0)[::-1]

    # select features
    sel_train = np.zeros((num_region, num_trial ,4))
    sel_test = np.zeros((num_region, num_test ,4))

    for region_idx in range(num_region):

        sel_train[region_idx, :, 0] = region_train_data[region_idx, :, int(idx[region_idx,0])]
        sel_train[region_idx, :, 1] = region_train_data[region_idx, :, int(idx[region_idx,1])]
        sel_train[region_idx, :, 2] = region_train_data[region_idx, :, int(idx[region_idx,0])]
        sel_train[region_idx, :, 3] = region_train_data[region_idx, :, int(idx[region_idx,1])]

        sel_test[region_idx, :, 0] = region_test_data[region_idx, :, int(idx[region_idx,0])]
        sel_test[region_idx, :, 1] = region_test_data[region_idx, :, int(idx[region_idx,1])]
        sel_test[region_idx, :, 2] = region_test_data[region_idx, :, int(idx[region_idx,0])]
        sel_test[region_idx, :, 3] = region_test_data[region_idx, :, int(idx[region_idx,1])]

    return sel_train, sel_test