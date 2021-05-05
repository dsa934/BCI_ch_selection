
import numpy as np
from scipy.signal import butter, filtfilt


def data_preprocessing(input_data):
    
    # input_data's keys : [ header, version, globals, cnt, nfo, mrk ]
    # cnt : signal's timesamples
    train_data = np.array(input_data['cnt'],dtype=np.float64)
    train_data = 0.1 * train_data

    # pos : each signal's visual cue position
    temp_cue = np.array(input_data['mrk']['pos'])
    cue = temp_cue.item()

    # whoel data trials : 280 ( train : test = 224 : 56 )
    whole_trial =280
    ntrain = 224
    ntest = whole_trial - ntrain
    nch = len(train_data[0])
    ntimesample = 5000;
    eeg_train = np.zeros((ntrain, nch, ntimesample ))
    eeg_test = np.zeros((ntest, nch, ntimesample))

    # split train_data & test data based visual cue  [trials, channels, timesamples]
    # train_data : [224,118,5000]  
    for train_idx in range(ntrain):

        temp_signal = train_data[cue[0][train_idx]:cue[0][train_idx]+ntimesample, :]
        eeg_train[train_idx] = temp_signal.T
    
    # test_data : [56, 118, 5000 ] 
    for test_idx in range(ntest):
        
        temp_signal = train_data[cue[0][test_idx+ntrain-1]:cue[0][test_idx+ntrain-1]+ntimesample, :]
        eeg_test[test_idx] = temp_signal.T
    
    # seperate class (right hand, right foot)
    class_set = np.array(input_data['mrk']['y']) 
    class_set = class_set[0][0][0]
    RH = list()
    RF = list()

    for train_idx in range(ntrain):

        if class_set[train_idx] == 1:
            RH.append(train_idx)
            
        else:
            RF.append(train_idx)
    
    rh_eeg = eeg_train[RH]
    rf_eeg = eeg_train[RF]

    # channel list based on referenced papers
    ch_list = [49,42,43,51,52,59,60,88,53,90,54,46,47,55,57,63,64,92]
    n_sel_ch = 18

    # data shape : [ntrials, n_sel_ch, ntimesamples]
    sel_rh_eeg = rh_eeg[:,ch_list,:]
    sel_rf_eeg = rf_eeg[:,ch_list,:]
    sel_test_eeg = eeg_test[:,ch_list,:]

    ## bandapss filtering with mu, beta and high-gamma bands
    [mb_max, mb_min] = butter(4, [9/500, 30/500], btype="band")
    [hg_max, hg_min] = butter(4, [140/500, 160/500], btype="band")

    bp_rh = np.zeros((len(RH),n_sel_ch,2500))
    mu_beta_rh = np.zeros((len(RH),ntimesample))
    hg_rh = np.zeros((len(RH),ntimesample))

    bp_rf = np.zeros((len(RF),n_sel_ch,2500))
    mu_beta_rf = np.zeros((len(RF),ntimesample))
    hg_rf = np.zeros((len(RF),ntimesample))

    bp_test = np.zeros((ntest,n_sel_ch,2500))
    mu_beta_test = np.zeros((ntest,ntimesample))
    hg_test = np.zeros((ntest,ntimesample))

    for ch_idx in range(n_sel_ch):

        for right_idx in range(len(RH)):

            mu_beta_rh[right_idx,:] = filtfilt(mb_max, mb_min, sel_rh_eeg[right_idx, ch_idx,:])
            hg_rh[right_idx,:] = filtfilt(hg_max, hg_min, sel_rh_eeg[right_idx, ch_idx,:])
        bp_rh[:,ch_idx,:] = mu_beta_rh[:,501:3001] + hg_rh[:,501:3001]


        for foot_idx in range(len(RF)):

            mu_beta_rf[foot_idx,:] = filtfilt(mb_max, mb_min, sel_rf_eeg[foot_idx, ch_idx,:])
            hg_rf[foot_idx,:] = filtfilt(hg_max, hg_min, sel_rf_eeg[foot_idx, ch_idx,:])
        bp_rf[:,ch_idx,:] = mu_beta_rf[:,501:3001] + hg_rf[:,501:3001]


        for test_idx in range(ntest):

            mu_beta_test[test_idx,:] = filtfilt(mb_max, mb_min, sel_test_eeg[test_idx, ch_idx,:])
            hg_test[test_idx,:] = filtfilt(hg_max, hg_min, sel_test_eeg[test_idx, ch_idx,:])
        bp_test[:,ch_idx,:] = mu_beta_test[:,501:3001] + hg_test[:,501:3001]

    return bp_rh, bp_rf, bp_test, len(RH), len(RF), ntest


def calculate_lrcsp_feature(rh_data, rf_data, test_data, len_rh, len_rf, len_test, n_region1, n_region2, n_region3, n_region4, n_region5=100, n_region6=100, n_region7=100):
    
    ## case 1: 4 channels
    if n_region5 == 100:
        rh_cov = np.zeros((len_rh, 4,4))
        rf_cov = np.zeros((len_rf, 4,4))
        test_cov = np.zeros((len_test, 4,4))

        # calculate each class's covariance matrix
        for rh_idx in range(len_rh):
            rh_region =  rh_data[rh_idx,n_region1,:], rh_data[rh_idx,n_region2,:],rh_data[rh_idx,n_region3,:],rh_data[rh_idx,n_region4,:] 
            rh_region = np.array(rh_region)
            rh_cov[rh_idx] = np.matmul(rh_region, rh_region.T) / np.trace(np.matmul(rh_region, rh_region.T))
    
        for rf_idx in range(len_rf):
            rf_region =  rf_data[rf_idx,n_region1,:], rf_data[rf_idx,n_region2,:],rf_data[rf_idx,n_region3,:],rf_data[rf_idx,n_region4,:] 
            rf_region = np.array(rf_region)
            rf_cov[rf_idx] = np.matmul(rf_region, rf_region.T) / np.trace(np.matmul(rf_region, rf_region.T))
 

        for test_idx in range(len_test):
            test_region =  test_data[test_idx,n_region1,:], test_data[test_idx,n_region2,:],test_data[test_idx,n_region3,:],test_data[test_idx,n_region4,:] 
            test_region = np.array(test_region)
            test_cov[test_idx] = np.matmul(test_region, test_region.T) / np.trace(np.matmul(test_region, test_region.T))

        # calculate CSP filter
        csp_filter, difference = jw_csp(rh_cov, rf_cov, 4)

        # set params
        rh_max_feature=list()
        rh_min_feature=list()
        rf_max_feature=list()
        rf_min_feature=list()
        test_max_feature=list()
        test_min_feature=list()

        ## calculate LRCSP features
        # right hand
        for rh_idx in range(len_rh):
            rh_region =  rh_data[rh_idx,n_region1,:], rh_data[rh_idx,n_region2,:],rh_data[rh_idx,n_region3,:],rh_data[rh_idx,n_region4,:]
            rh_feature = np.matmul(csp_filter, rh_region)

            rh_max_feature.append( np.var(rh_feature[0]) / (np.var(rh_feature[0]) + np.var(rh_feature[1])) )
            rh_min_feature.append( np.var(rh_feature[1]) / (np.var(rh_feature[0]) + np.var(rh_feature[1])) )

        feature_rh = rh_max_feature, rh_min_feature
        feature_rh = np.array(feature_rh).T

        # right foot
        for rf_idx in range(len_rf):
            rf_region =  rf_data[rf_idx,n_region1,:], rf_data[rf_idx,n_region2,:],rf_data[rf_idx,n_region3,:],rf_data[rf_idx,n_region4,:]
            rf_feature = np.matmul(csp_filter, rf_region)

            rf_max_feature.append( np.var(rf_feature[0]) / (np.var(rf_feature[0]) + np.var(rf_feature[1])) )
            rf_min_feature.append( np.var(rf_feature[1]) / (np.var(rf_feature[0]) + np.var(rf_feature[1])) )

        feature_rf = rf_max_feature, rf_min_feature
        feature_rf = np.array(feature_rf).T

        # test 
        for test_idx in range(len_test):
            test_region =  test_data[test_idx,n_region1,:], test_data[test_idx,n_region2,:],test_data[test_idx,n_region3,:],test_data[test_idx,n_region4,:]
            test_feature = np.matmul(csp_filter, test_region)

            test_max_feature.append( np.var(test_feature[0]) / (np.var(test_feature[0]) + np.var(test_feature[1])) )
            test_min_feature.append( np.var(test_feature[1]) / (np.var(test_feature[0]) + np.var(test_feature[1])) )

        feature_test = test_max_feature, test_min_feature
        feature_test = np.array(feature_test).T

    ## case 2 : 5 channels 
    elif n_region6 ==100:
        rh_cov = np.zeros((len_rh, 5,5))
        rf_cov = np.zeros((len_rf, 5,5))
        test_cov = np.zeros((len_test, 5,5))

        # calculate each class's covariance matrix
        for rh_idx in range(len_rh):
            rh_region =  rh_data[rh_idx,n_region1,:], rh_data[rh_idx,n_region2,:],rh_data[rh_idx,n_region3,:],rh_data[rh_idx,n_region4,:] ,rh_data[rh_idx,n_region5,:]
            rh_region = np.array(rh_region)
            rh_cov[rh_idx] = np.matmul(rh_region, rh_region.T) / np.trace(np.matmul(rh_region, rh_region.T))
    
        for rf_idx in range(len_rf):
            rf_region =  rf_data[rf_idx,n_region1,:], rf_data[rf_idx,n_region2,:],rf_data[rf_idx,n_region3,:],rf_data[rf_idx,n_region4,:] ,rf_data[rf_idx,n_region5,:]
            rf_region = np.array(rf_region)
            rf_cov[rf_idx] = np.matmul(rf_region, rf_region.T) / np.trace(np.matmul(rf_region, rf_region.T))
 

        for test_idx in range(len_test):
            test_region =  test_data[test_idx,n_region1,:], test_data[test_idx,n_region2,:],test_data[test_idx,n_region3,:],test_data[test_idx,n_region4,:] ,test_data[test_idx,n_region5,:]
            test_region = np.array(test_region)
            test_cov[test_idx] = np.matmul(test_region, test_region.T) / np.trace(np.matmul(test_region, test_region.T))
    

        # calculate CSP filter
        csp_filter, difference = jw_csp(rh_cov, rf_cov,5)

        # set params
        rh_max_feature=list()
        rh_min_feature=list()
        rf_max_feature=list()
        rf_min_feature=list()
        test_max_feature=list()
        test_min_feature=list()


        ## calculate LRCSP features
        # right hand
        for rh_idx in range(len_rh):
            rh_region =  rh_data[rh_idx,n_region1,:], rh_data[rh_idx,n_region2,:],rh_data[rh_idx,n_region3,:],rh_data[rh_idx,n_region4,:],rh_data[rh_idx,n_region5,:]
            rh_feature = np.matmul(csp_filter, rh_region)

            rh_max_feature.append( np.var(rh_feature[0]) / (np.var(rh_feature[0]) + np.var(rh_feature[1])) )
            rh_min_feature.append( np.var(rh_feature[1]) / (np.var(rh_feature[0]) + np.var(rh_feature[1])) )

        feature_rh = rh_max_feature, rh_min_feature
        feature_rh = np.array(feature_rh).T

        # right foot
        for rf_idx in range(len_rf):
            rf_region =  rf_data[rf_idx,n_region1,:], rf_data[rf_idx,n_region2,:],rf_data[rf_idx,n_region3,:],rf_data[rf_idx,n_region4,:],rf_data[rf_idx,n_region5,:]
            rf_feature = np.matmul(csp_filter, rf_region)

            rf_max_feature.append( np.var(rf_feature[0]) / (np.var(rf_feature[0]) + np.var(rf_feature[1])) )
            rf_min_feature.append( np.var(rf_feature[1]) / (np.var(rf_feature[0]) + np.var(rf_feature[1])) )

        feature_rf = rf_max_feature, rf_min_feature
        feature_rf = np.array(feature_rf).T

        # test 
        for test_idx in range(len_test):
            test_region =  test_data[test_idx,n_region1,:], test_data[test_idx,n_region2,:],test_data[test_idx,n_region3,:],test_data[test_idx,n_region4,:],test_data[test_idx,n_region5,:]
            test_feature = np.matmul(csp_filter, test_region)

            test_max_feature.append( np.var(test_feature[0]) / (np.var(test_feature[0]) + np.var(test_feature[1])) )
            test_min_feature.append( np.var(test_feature[1]) / (np.var(test_feature[0]) + np.var(test_feature[1])) )

        feature_test = test_max_feature, test_min_feature
        feature_test = np.array(feature_test).T

    # case 3: 6 channels
    elif n_region7 ==100:
        rh_cov = np.zeros((len_rh, 6,6))
        rf_cov = np.zeros((len_rf, 6,6))
        test_cov = np.zeros((len_test, 6,6))

        # calculate each class's covariance matrix
        for rh_idx in range(len_rh):
            rh_region =  rh_data[rh_idx,n_region1,:], rh_data[rh_idx,n_region2,:],rh_data[rh_idx,n_region3,:],rh_data[rh_idx,n_region4,:],rh_data[rh_idx,n_region5,:] ,rh_data[rh_idx,n_region6,:]  
            rh_region = np.array(rh_region)
            rh_cov[rh_idx] = np.matmul(rh_region, rh_region.T) / np.trace(np.matmul(rh_region, rh_region.T))
    
        for rf_idx in range(len_rf):
            rf_region =  rf_data[rf_idx,n_region1,:], rf_data[rf_idx,n_region2,:],rf_data[rf_idx,n_region3,:],rf_data[rf_idx,n_region4,:],rf_data[rf_idx,n_region5,:] ,rf_data[rf_idx,n_region6,:]  
            rf_region = np.array(rf_region)
            rf_cov[rf_idx] = np.matmul(rf_region, rf_region.T) / np.trace(np.matmul(rf_region, rf_region.T))
 

        for test_idx in range(len_test):
            test_region =  test_data[test_idx,n_region1,:], test_data[test_idx,n_region2,:],test_data[test_idx,n_region3,:],test_data[test_idx,n_region4,:],test_data[test_idx,n_region5,:],test_data[test_idx,n_region6,:] 
            test_region = np.array(test_region)
            test_cov[test_idx] = np.matmul(test_region, test_region.T) / np.trace(np.matmul(test_region, test_region.T))
    

        # calculate CSP filter
        csp_filter, difference = jw_csp(rh_cov, rf_cov,6)

        # set params
        rh_max_feature=list()
        rh_min_feature=list()
        rf_max_feature=list()
        rf_min_feature=list()
        test_max_feature=list()
        test_min_feature=list()


        ## calculate LRCSP features
        # right hand
        for rh_idx in range(len_rh):
            rh_region =  rh_data[rh_idx,n_region1,:], rh_data[rh_idx,n_region2,:],rh_data[rh_idx,n_region3,:],rh_data[rh_idx,n_region4,:],rh_data[rh_idx,n_region5,:],rh_data[rh_idx,n_region6,:]
            rh_feature = np.matmul(csp_filter, rh_region)

            rh_max_feature.append( np.var(rh_feature[0]) / (np.var(rh_feature[0]) + np.var(rh_feature[1])) )
            rh_min_feature.append( np.var(rh_feature[1]) / (np.var(rh_feature[0]) + np.var(rh_feature[1])) )

        feature_rh = rh_max_feature, rh_min_feature
        feature_rh = np.array(feature_rh).T

        # right foot
        for rf_idx in range(len_rf):
            rf_region =  rf_data[rf_idx,n_region1,:], rf_data[rf_idx,n_region2,:],rf_data[rf_idx,n_region3,:],rf_data[rf_idx,n_region4,:],rf_data[rf_idx,n_region5,:],rf_data[rf_idx,n_region6,:]
            rf_feature = np.matmul(csp_filter, rf_region)

            rf_max_feature.append( np.var(rf_feature[0]) / (np.var(rf_feature[0]) + np.var(rf_feature[1])) )
            rf_min_feature.append( np.var(rf_feature[1]) / (np.var(rf_feature[0]) + np.var(rf_feature[1])) )

        feature_rf = rf_max_feature, rf_min_feature
        feature_rf = np.array(feature_rf).T

        # test 
        for test_idx in range(len_test):
            test_region =  test_data[test_idx,n_region1,:], test_data[test_idx,n_region2,:],test_data[test_idx,n_region3,:],test_data[test_idx,n_region4,:],test_data[test_idx,n_region5,:],test_data[test_idx,n_region6,:]
            test_feature = np.matmul(csp_filter, test_region)

            test_max_feature.append( np.var(test_feature[0]) / (np.var(test_feature[0]) + np.var(test_feature[1])) )
            test_min_feature.append( np.var(test_feature[1]) / (np.var(test_feature[0]) + np.var(test_feature[1])) )

        feature_test = test_max_feature, test_min_feature
        feature_test = np.array(feature_test).T


    # case 4: 7 channels
    else:
        rh_cov = np.zeros((len_rh, 7,7))
        rf_cov = np.zeros((len_rf, 7,7))
        test_cov = np.zeros((len_test, 7,7))

        # calculate each class's covariance matrix
        for rh_idx in range(len_rh):
            rh_region =  rh_data[rh_idx,n_region1,:], rh_data[rh_idx,n_region2,:],rh_data[rh_idx,n_region3,:],rh_data[rh_idx,n_region4,:],rh_data[rh_idx,n_region5,:] ,rh_data[rh_idx,n_region6,:] ,rh_data[rh_idx,n_region7,:]  
            rh_region = np.array(rh_region)
            rh_cov[rh_idx] = np.matmul(rh_region, rh_region.T) / np.trace(np.matmul(rh_region, rh_region.T))
    
        for rf_idx in range(len_rf):
            rf_region =  rf_data[rf_idx,n_region1,:], rf_data[rf_idx,n_region2,:],rf_data[rf_idx,n_region3,:],rf_data[rf_idx,n_region4,:],rf_data[rf_idx,n_region5,:], rf_data[rf_idx,n_region6,:], rf_data[rf_idx,n_region7,:] 
            rf_region = np.array(rf_region)
            rf_cov[rf_idx] = np.matmul(rf_region, rf_region.T) / np.trace(np.matmul(rf_region, rf_region.T))
 

        for test_idx in range(len_test):
            test_region =  test_data[test_idx,n_region1,:], test_data[test_idx,n_region2,:],test_data[test_idx,n_region3,:],test_data[test_idx,n_region4,:],test_data[test_idx,n_region5,:],test_data[test_idx,n_region6,:],test_data[test_idx,n_region7,:] 
            test_region = np.array(test_region)
            test_cov[test_idx] = np.matmul(test_region, test_region.T) / np.trace(np.matmul(test_region, test_region.T))
    

        # calculate CSP filter
        csp_filter, difference = jw_csp(rh_cov, rf_cov,7)

        # set params
        rh_max_feature=list()
        rh_min_feature=list()
        rf_max_feature=list()
        rf_min_feature=list()
        test_max_feature=list()
        test_min_feature=list()


        ## calculate LRCSP features
        # right hand
        for rh_idx in range(len_rh):
            rh_region =  rh_data[rh_idx,n_region1,:], rh_data[rh_idx,n_region2,:],rh_data[rh_idx,n_region3,:],rh_data[rh_idx,n_region4,:],rh_data[rh_idx,n_region5,:],rh_data[rh_idx,n_region6,:],rh_data[rh_idx,n_region7,:]
            rh_feature = np.matmul(csp_filter, rh_region)

            rh_max_feature.append( np.var(rh_feature[0]) / (np.var(rh_feature[0]) + np.var(rh_feature[1])) )
            rh_min_feature.append( np.var(rh_feature[1]) / (np.var(rh_feature[0]) + np.var(rh_feature[1])) )

        feature_rh = rh_max_feature, rh_min_feature
        feature_rh = np.array(feature_rh).T

        # right foot
        for rf_idx in range(len_rf):
            rf_region =  rf_data[rf_idx,n_region1,:], rf_data[rf_idx,n_region2,:],rf_data[rf_idx,n_region3,:],rf_data[rf_idx,n_region4,:],rf_data[rf_idx,n_region5,:],rf_data[rf_idx,n_region6,:],rf_data[rf_idx,n_region7,:]
            rf_feature = np.matmul(csp_filter, rf_region)

            rf_max_feature.append( np.var(rf_feature[0]) / (np.var(rf_feature[0]) + np.var(rf_feature[1])) )
            rf_min_feature.append( np.var(rf_feature[1]) / (np.var(rf_feature[0]) + np.var(rf_feature[1])) )

        feature_rf = rf_max_feature, rf_min_feature
        feature_rf = np.array(feature_rf).T

        # test 
        for test_idx in range(len_test):
            test_region =  test_data[test_idx,n_region1,:], test_data[test_idx,n_region2,:],test_data[test_idx,n_region3,:],test_data[test_idx,n_region4,:],test_data[test_idx,n_region5,:],test_data[test_idx,n_region6,:],test_data[test_idx,n_region7,:]
            test_feature = np.matmul(csp_filter, test_region)

            test_max_feature.append( np.var(test_feature[0]) / (np.var(test_feature[0]) + np.var(test_feature[1])) )
            test_min_feature.append( np.var(test_feature[1]) / (np.var(test_feature[0]) + np.var(test_feature[1])) )

        feature_test = test_max_feature, test_min_feature
        feature_test = np.array(feature_test).T


        

    return feature_rh, feature_rf, feature_test, difference



def jw_csp(rh_cov, rf_cov, n_ch):
    
    mean_rh = np.mean(rh_cov,0)
    mean_rf = np.mean(rf_cov,0)
    
    cov_sum = mean_rh + mean_rf

    # eigen value decomposition
    eigen_val, eigen_vec = np.linalg.eig(cov_sum)

    # 1-dim val vec -> 2-dim matrix (by diag)
    eigen_val = np.diag(eigen_val)

    # 백색화변환행렬
    P =  np.matmul(np.sqrt(np.linalg.inv(eigen_val)),eigen_vec.T )
    s_RH = np.matmul(np.matmul(P, mean_rh),P.T)
    s_RF = np.matmul(np.matmul(P,mean_rf),P.T)

    RH_eigen_val, RH_eigen_vec = np.linalg.eig(s_RH)
    RF_eigen_val, RF_eigen_vec = np.linalg.eig(s_RF)

    dif1 = np.max(RH_eigen_val)
    dif2 = np.min(RH_eigen_val)

    difference = dif1 - dif2

    max_feature = np.matmul(P.T, RH_eigen_vec)
    min_feature = max_feature.T

    BB = np.matmul(np.matmul(max_feature.T,mean_rh ),max_feature)
    BBB = np.matmul(np.matmul(max_feature.T,mean_rf ),max_feature)
    
    RH_max_val = np.argmax(np.diag(BB))
    RF_max_val = np.argmax(np.diag(BBB))

    # csp_filter = 2,n_ch
    csp_filter = np.zeros((2,n_ch))
    csp_filter[0] = min_feature[RH_max_val]
    csp_filter[1] = min_feature[RF_max_val]

    return csp_filter, difference
