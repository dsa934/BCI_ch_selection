
import numpy as np
import scipy.stats as statistic
from scipy.signal import butter, filtfilt
from sklearn.metrics import mutual_info_score


def config_distinctive_channels(train_data):

    # set init args
    num_train = len(train_data)
    num_ch = len(train_data[1,:,:])
    num_timesample = len(train_data[1,1,:])

    # seperate RH,RF class
    rh_eeg = train_data[:int(len(train_data)/2),:,:]
    rf_eeg = train_data[int(len(train_data)/2):,:,:]

    # bandpass filtering
    [max_boundary, min_boundary] = butter(4, [4/50, 36/50], btype='band')

    bp_right = np.zeros((num_train,num_ch,int((num_timesample-1)/2)))
    bp_foot = np.zeros((num_train,num_ch,int((num_timesample-1)/2)))
    
    for ch_idx in range(num_ch):

        # right hand case
        for right_idx in range (int(num_train/2)):
            tmp = filtfilt(max_boundary, min_boundary, rh_eeg[right_idx, ch_idx, :])            
            bp_right[right_idx, ch_idx, :] = tmp[50:300]

        # right foot case
        for foot_idx in range(int(num_train/2)):
            tmp = filtfilt(max_boundary, min_boundary, rf_eeg[foot_idx, ch_idx, :])
            bp_foot[foot_idx, ch_idx, :] = tmp[50:300]


    # distinctive channels using t-statistic (based on CC)
    corr_rh = np.zeros((int(num_train/2), num_ch, num_ch))
    corr_rf = np.zeros((int(num_train/2), num_ch, num_ch))

    for right_idx in range(int(num_train/2)):

        corr_rh[right_idx,:,:] = np.corrcoef(bp_right[right_idx,:,:])

    for foot_idx in range(int(num_train/2)):

        corr_rf[foot_idx,:,:] = np.corrcoef(bp_foot[foot_idx,:,:])


    # calcuate t-statistic & p_value  ( ref paper's equation (6) )
    p_thr = 0.05
    p_value = np.zeros((num_ch,num_ch))
    
    # t-statistic, p_value = ttest_idx(data1,data2)
    for row_idx in range(num_ch):

        for col_idx in range(num_ch):

            _, p_value[row_idx, col_idx] = statistic.ttest_ind(corr_rh[:, row_idx, col_idx], corr_rf[:, row_idx, col_idx])

    
    # calculate MI score for each channels 
    ck_mi_score = np.zeros((num_ch,num_ch))

    for base_idx in range(num_ch):

        for cmp_idx in range(num_ch):

            if p_value[base_idx, cmp_idx] < p_thr:

                ck_mi_score[base_idx, cmp_idx] = p_value[base_idx,cmp_idx]

    # mi_score : 각 채널별 조건( over p_value)를 만족하는 채널의 수
    mi_score = np.zeros((num_ch))

    for base_idx in range(num_ch):

        mi_score[base_idx] = len(np.array(np.nonzero(ck_mi_score[base_idx][:])).T)

    # H : distinctive channles ( higher than the avg mi_score, ref paper )  
    H = list()
    for base_idx in range(num_ch):

        if mi_score[base_idx] > np.mean(mi_score):

            H.append(base_idx)

    # support channel group of distinctive channel group H
    mean_rh = np.mean(corr_rh,axis=0)
    mean_rf = np.mean(corr_rf,axis=0)

    # p_thr = 0.9
    semi_p_thr = 0.9
    support_channel_group = list()
    
    for h_idx in range(len(H)):

        temp = list()

        for cmp_idx in range(len(H)):

            if mean_rh[H[h_idx], H[cmp_idx]] > semi_p_thr and mean_rf[H[h_idx], H[cmp_idx]] > semi_p_thr:
                
                temp.append(H[cmp_idx])

        support_channel_group.append(temp)

    
    return rh_eeg, rf_eeg, support_channel_group



def fisher_fbcsp_feature(rh_eeg, rf_eeg, test_eeg, support_channel_group, minfs, maxfs):

    # set init args
    # support_channel_group :(42,) , rh_eeg : (126,118,501)
    num_group = len(support_channel_group)
    num_trial = len(rh_eeg)
    num_ch = len(rh_eeg[0])
    num_time = int((len(rh_eeg[0][0])-1)/2)
    num_test_trial = len(test_eeg)
    
    [max_boundary, min_boundary] = butter(4, [minfs/50, maxfs/50], btype='band') 

    fb_rh = np.zeros((num_trial, num_ch, num_time))
    fb_rf = np.zeros((num_trial, num_ch, num_time))
    fb_test = np.zeros((num_test_trial, num_ch, num_time))
    

    # bandpass filtering [minfs - maxfs]
    for trial_idx in range(num_trial):

        for ch_idx in range(num_ch):

            temp_rh= filtfilt(max_boundary, min_boundary, rh_eeg[trial_idx,ch_idx,:])
            temp_rf= filtfilt(max_boundary, min_boundary, rf_eeg[trial_idx,ch_idx,:])

            fb_rh[trial_idx,ch_idx, :] = temp_rh[50:300]
            fb_rf[trial_idx,ch_idx, :] = temp_rf[50:300]


    for test_idx in range(num_test_trial):

        for ch_idx in range(num_ch):

            temp_test= filtfilt(max_boundary, min_boundary, test_eeg[test_idx,ch_idx,:])
            
            fb_test[test_idx,ch_idx,:] = temp_test[50:300]



    gfbcsp_train, gfbcsp_test = support_csp_filter(fb_rh, fb_rf, fb_test, support_channel_group, num_group, num_trial, num_ch, num_test_trial)

    return gfbcsp_train, gfbcsp_test



def support_csp_filter(fb_rh, fb_rf, fb_test, support_channel_group, num_group, num_trial, num_ch, num_test_trial):
    
    #set init args
    num_time = len(fb_rh[0][0])
    gfbcsp_feature = np.zeros((num_group, num_trial*2,2))
    gfbcsp_test_feature = np.zeros((num_group, num_test_trial,2))
   
    # calculate group fbcsp features
    for group_idx in range(num_group):

        # get each group csp filter
        ch_list = support_channel_group[group_idx]
        zrh_feature = np.zeros((num_trial,2))
        zrf_feature = np.zeros((num_trial,2))
        ztest_feature = np.zeros((num_test_trial,2))

        gcsp_filter = group_csp_filter(fb_rh, fb_rf, ch_list, num_group, num_trial, num_ch, num_test_trial, num_time)

        for trial_idx in range(num_trial):

            # z_rh / _rf : [2 x num_time]
            z_rh = np.matmul(gcsp_filter,fb_rh[trial_idx,ch_list,:])
            z_rf = np.matmul(gcsp_filter,fb_rf[trial_idx,ch_list,:])
                        
            var_max_z_rh = np.log10(np.var(z_rh[0,:]) / ( np.var(z_rh[0,:]) + np.var(z_rh[1,:]) ) )
            var_min_z_rh = np.log10(np.var(z_rh[1,:]) / ( np.var(z_rh[0,:]) + np.var(z_rh[1,:]) ) )

            var_max_z_rf = np.log10(np.var(z_rf[0,:]) / ( np.var(z_rf[0,:]) + np.var(z_rf[1,:]) ) )
            var_min_z_rf = np.log10(np.var(z_rf[1,:]) / ( np.var(z_rf[0,:]) + np.var(z_rf[1,:]) ) )
            

            zrh_feature[trial_idx,0] = var_max_z_rh
            zrh_feature[trial_idx,1] = var_min_z_rh

            zrf_feature[trial_idx,0] = var_max_z_rf
            zrf_feature[trial_idx,1] = var_min_z_rf


        gfbcsp_feature[group_idx] = np.concatenate((zrh_feature, zrf_feature), axis=0)

        for test_idx in range(num_test_trial):

            z_test = np.matmul(gcsp_filter,fb_test[test_idx,ch_list,:])
            
            var_max_z_test = np.log10(np.var(z_test[0,:]) / ( np.var(z_test[0,:]) + np.var(z_test[1,:]) ) )
            var_min_z_test = np.log10(np.var(z_test[1,:]) / ( np.var(z_test[0,:]) + np.var(z_test[1,:]) ) )
            

            ztest_feature[test_idx,0] = var_max_z_test
            ztest_feature[test_idx,1] = var_min_z_test


        gfbcsp_test_feature[group_idx] = ztest_feature

    return gfbcsp_feature, gfbcsp_test_feature
            

def group_csp_filter(fb_rh, fb_rf, ch_list, num_group, num_trial, num_ch, num_test_trial, num_time):

    # calcualte each group csp filter  (42 groups)
    # fb_rh : [ 125, 118, 250]
    # group 마다 다른 ch 정보를 가지고 다니면서 처리하면 크기가 달라도 처리가능 
    rh_cov = np.zeros((num_trial, len(ch_list),len(ch_list)))
    rf_cov = np.zeros((num_trial, len(ch_list),len(ch_list)))
    
    for trial_idx in range(num_trial):

        rh_cov[trial_idx,:,:] = np.matmul(fb_rh[trial_idx,ch_list,:],fb_rh[trial_idx,ch_list,:].T) / np.trace(np.matmul(fb_rh[trial_idx,ch_list,:],fb_rh[trial_idx,ch_list,:].T))
        rf_cov[trial_idx,:,:] = np.matmul(fb_rf[trial_idx,ch_list,:],fb_rf[trial_idx,ch_list,:].T) / np.trace(np.matmul(fb_rf[trial_idx,ch_list,:],fb_rf[trial_idx,ch_list,:].T))


    mean_rh = np.mean(rh_cov, axis=0)
    mean_rf = np.mean(rf_cov, axis=0)
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

    max_feature = np.matmul(P.T, RH_eigen_vec)
    min_feature = max_feature.T

    BB = np.matmul(np.matmul(max_feature.T,mean_rh ),max_feature)
    BBB = np.matmul(np.matmul(max_feature.T,mean_rf ),max_feature)
    
    RH_max_val = np.argmax(np.diag(BB))
    RF_max_val = np.argmax(np.diag(BBB))

    # csp_filter [2,len(ch_list)]
    csp_filter = np.zeros((2, len(ch_list)))
    csp_filter[0] = min_feature[RH_max_val]
    csp_filter[1] = min_feature[RF_max_val]

    return csp_filter

def mutual_information(train_data, test_data, train_label):

    # set init args
    # train_data shape : [ num_group(42), trial(252), 16(2x8) ]
    # train_label shape : [1,252]
    num_fb = 8
    num_group = len(train_data)
    num_trial = len(train_data[0])
    num_test_trial = len(test_data[0])
    max_values = [0,2,4,6,8,10,12,14]
    train_label = train_label[0]
    mi_fb = np.zeros((num_group, num_fb))
    idx = np.zeros((num_group,num_fb))

    for group_idx in range(num_group):

        for fb_idx in range(num_fb):

            mi_fb[group_idx, fb_idx] = mutual_info_score(train_label,train_data[group_idx,:,max_values[fb_idx]].squeeze())
            
        idx[group_idx,:num_fb] = np.argsort((mi_fb[group_idx,:]), axis=0)[::-1]

    # select high mi score features
    sel_group_csp_feature = np.zeros((num_group, num_trial, int(4)))
    sel_group_csp_test_feature = np.zeros((num_group, num_test_trial, int(4)))

    for group_idx in range(num_group):

        sel_group_csp_feature[group_idx,:,0] = train_data[group_idx,:,max_values[int(idx[group_idx,0])]]
        sel_group_csp_feature[group_idx,:,1] = train_data[group_idx,:,max_values[int(idx[group_idx,0])]+1]
        sel_group_csp_feature[group_idx,:,2] = train_data[group_idx,:,max_values[int(idx[group_idx,1])]]
        sel_group_csp_feature[group_idx,:,3] = train_data[group_idx,:,max_values[int(idx[group_idx,1])]+1]

        sel_group_csp_test_feature[group_idx,:,0] = test_data[group_idx,:,max_values[int(idx[group_idx,0])]]
        sel_group_csp_test_feature[group_idx,:,1] = test_data[group_idx,:,max_values[int(idx[group_idx,0])]+1]
        sel_group_csp_test_feature[group_idx,:,2] = test_data[group_idx,:,max_values[int(idx[group_idx,1])]]
        sel_group_csp_test_feature[group_idx,:,3] = test_data[group_idx,:,max_values[int(idx[group_idx,1])]+1]

    # select group by fisher score
    deno_rh = np.zeros((int(num_trial/2)))
    deno_rf = np.zeros((int(num_trial/2)))
    z_fisher = np.zeros((num_group))
    z_idx = np.zeros((num_group))
    for group_idx in range(num_group):

        grh_csp_feature = sel_group_csp_feature[group_idx,:int(num_trial/2),:]
        grf_csp_feature = sel_group_csp_feature[group_idx, int(num_trial/2):,:]

        numerator = np.linalg.norm( (np.mean(grh_csp_feature, axis=0)) - np.mean(grf_csp_feature,axis=0))

        for trial_idx in range(int(num_trial/2)):

            deno_rh[trial_idx] = np.linalg.norm(grh_csp_feature[trial_idx,:] - np.mean(grh_csp_feature,axis=0))
            deno_rf[trial_idx] = np.linalg.norm(grf_csp_feature[trial_idx,:] - np.mean(grf_csp_feature,axis=0))

        denominator = (np.mean(deno_rh) + np.mean(deno_rf)) * 0.5
        z_fisher[group_idx] = numerator / denominator

    z_idx = np.argsort((z_fisher))[::-1]
    
    # selected group number, fb_feature number (according group number)
    sel_group = z_idx[1]

    
    # final_train_data
    train_data_set = sel_group_csp_feature[sel_group,:,:]

    # final_test_data
    test_data_set = sel_group_csp_test_feature[sel_group,:,:]

    return train_data_set, test_data_set


    




