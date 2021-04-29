%{

- cross validation 사용 (BCI 데이터는 trial이 적기 때문)
- mi : MIToolbox-3.0.0의 함수를 사용하여 mutual information을 계산 (line 671)

%}

%% load band-passed data (applied cross validation)
clear all
load('C:\Users\dsa93\Desktop\compare_paper_other_algorithm\park_optimal_cs_journal\data100Hz\ay\train\train_10');
load('C:\Users\dsa93\Desktop\compare_paper_other_algorithm\park_optimal_cs_journal\data100Hz\ay\train\train_label_10');
load('C:\Users\dsa93\Desktop\compare_paper_other_algorithm\park_optimal_cs_journal\data100Hz\ay\test\test_10');
load('C:\Users\dsa93\Desktop\compare_paper_other_algorithm\park_optimal_cs_journal\data100Hz\ay\test\test_label_10');

original_train_data = train_data;
original_train_label = train_label;
original_test_data = test_data;
original_test_label = test_label;

clear test_data test_label train_data train_label

%% init params
data_training=length(original_train_label);
rh=1;
rf=1;

% right hand/foot seperate
for k=1:data_training
    if original_train_label(k)==0
        ll(rh)=k;
        rh=rh+1;
    else
        rr(rf)=k;
        rf=rf+1;
    end
end
l=length(ll);
r=length(rr);

tr=280-data_training;

for k=1:l
    left_eeg(:,:,k)=original_train_data(ll(k),:,:);
end
for k=1:r
    right_eeg(:,:,k)=original_train_data(rr(k),:,:);
end

for k=1:tr
    test_eeg(:,:,k)=original_test_data(k,:,:);
end

% bandpass filtering 4- 36Hz
[bbb,aaa]=butter(4,[4/50 36/50]);
for ch_idx =1:118
    % left trials
    for k=1:l
        ect1(k,:)=filtfilt(bbb,aaa,left_eeg(ch_idx,:,k));
    end
    bp_RH(:,:,ch_idx)=ect1(:,51:300);
        
    % right trials    
    for k=1:r
        ect2(k,:)=filtfilt(bbb,aaa,right_eeg(ch_idx,:,k));
    end
    bp_RF(:,:,ch_idx)=ect2(:,51:300);
    
end
    
%% distinctive channels based on correlation coefficient using t-statistic
% corr_RH = [126 x 118 x 118] bp_RH = [ 126 x 250 x 118] ( trials, time , ch)
for rh_idx = 1: l
    corr_RH(rh_idx,:,:) = corrcoef(squeeze(bp_RH(rh_idx,:,:)));
end

for rf_idx = 1:r
    corr_RF(rf_idx,:,:) = corrcoef(squeeze(bp_RF(rf_idx,:,:)));
end


% find t-statistic & p_value simultaneously
% paper equation (6) ref.
% 등분산 을 가정해야만 자유도가 (l + r -2)
% P^(k,p) 는 k,p 채널에 대한 p_value니까 ttest2에 넣기위해서는 trial들을 vector화
% trial x 1 x1 ( trial x k x p )
P_thr = 0.05;

for row_idx = 1:118
    
    for col_idx = 1:118
        
        [h, p_value(row_idx,col_idx)] = ttest2(corr_RH(:,row_idx,col_idx), corr_RF(:,row_idx,col_idx), 'Vartype','equal');
    end
end

% calculate MI_score each channels s(k) =(k, 1 , ... , 118)  , s(k+1) = (k+1, 1, ..., 118)

before_mi_score= zeros(118,118);

for base_idx = 1: 118
    
    for comp_idx = 1:118
        
        if p_value(base_idx, comp_idx) < P_thr
            before_mi_score(base_idx, comp_idx) = p_value(base_idx, comp_idx);
        end
    end
end

% mi_score = 조건을 만족하는 채널수 
for base_idx = 1:118
    
    mi_score(base_idx) = length(nonzeros(before_mi_score(base_idx,:))');
end

% H : distinctive channles ( higher than the avg mi_score ) 
ck_p_idx = 1;
for base_idx = 1:118
    
    if  mi_score(base_idx) > mean(mi_score) 
        H(ck_p_idx) = base_idx;
        ck_p_idx = ck_p_idx +1;
    end
end

%% supporting channel group of distintive channel D^(h)
% mean correlation coefficient
mean_RH = squeeze(mean(corr_RH,1));
mean_RF = squeeze(mean(corr_RF,1));

% p_thr = 0.9
semi_P_thr = 0.9;

for h_idx = 1 : length(H)
    ck_p_D=1;
    for cmp_idx = 1: length(H)
        
        if mean_RH( H(h_idx), H(cmp_idx)) > semi_P_thr && mean_RF( H(h_idx), H(cmp_idx)) > semi_P_thr 
            support_ch_D(h_idx, ck_p_D) = H(cmp_idx);
            ck_p_D = ck_p_D+1;
        end
    end
end


%% fisher score & FBCSP feature ( group seperate -> FBCSP -> ...)
% original signal
% left/ right eeg = [118, 501, 112] 
% support group eeg signal , num of support group = length(suppot_ch_D);

num_support = length(support_ch_D);
num_ch_support = length(support_ch_D(1,:));
num_time = length(left_eeg(1,:,1));

num_RH_trial = length(left_eeg(1,1,:));
num_RF_trial = length(right_eeg(1,1,:));
num_test_trial = length(test_eeg(1,1,:));

num_trial_RH = length(left_eeg(1,1,:));
num_trial_RF = length(right_eeg(1,1,:));


% seperate groups
% 그룹마다 유효채널수가 다르기때문에 그수를 기억해야함 (rem_sup_ch)
for idx = 1: num_support
    rem_sup_ch(idx,1) = length(nonzeros(support_ch_D(idx,:)));
end

% suport_RH/RF =[63, 15, 501,112] ( 그룹수, 그룹내 최고채널수, time, trial)
support_RH = zeros(num_support, num_ch_support, num_time, num_RH_trial);
support_RF = zeros(num_support, num_ch_support, num_time, num_RF_trial);
support_test =  zeros(num_support, num_ch_support, num_time, num_test_trial);

for sup_idx = 1: num_support
     support_ch = nonzeros(support_ch_D(sup_idx,:));
     support_RH(sup_idx, 1:length(support_ch),:,:) = left_eeg(support_ch, :, :);
end

for sup_idx = 1: num_support
     support_ch = nonzeros(support_ch_D(sup_idx,:));
     support_RF(sup_idx, 1:length(support_ch),:,:) = right_eeg(support_ch, :, :);
end

for sup_idx = 1: num_support
     support_ch = nonzeros(support_ch_D(sup_idx,:));
     support_test(sup_idx, 1:length(support_ch),:,:) = test_eeg(support_ch, :, :);
end


%% 8 FBCSP features (4-36Hz)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4-8 Hz
[fb1_max,fb1_min]=butter(4,[4/50 8/50]);

% suport_RH/RF =[63, 15, 501,112]
% fb1_RH / RF = [63, 15, 112, 250]  -> [63,112,250,15]
for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for rh_trial_idx = 1: num_RH_trial
            
            ect1(rh_trial_idx,:)=filtfilt(fb1_max,fb1_min,squeeze(support_RH(group_idx,ch_idx,:,rh_trial_idx)));
            
        end
        fb1_temp_RH(:,:,ch_idx)=ect1(:,51:300);
    end
    fb1_RH(group_idx,:,:,:)=fb1_temp_RH;
    
end

for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for rf_trial_idx = 1: num_RF_trial
            
            ect2(rf_trial_idx,:)=filtfilt(fb1_max,fb1_min,squeeze(support_RF(group_idx,ch_idx,:,rf_trial_idx)));
            
        end
        fb1_temp_RF(:,:,ch_idx)=ect2(:,51:300);
    end
    fb1_RF(group_idx,:,:,:)=fb1_temp_RF;
    
end

for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for test_trial_idx = 1: num_test_trial
            
            ect3(test_trial_idx,:)=filtfilt(fb1_max,fb1_min,squeeze(support_test(group_idx,ch_idx,:,test_trial_idx)));
            
        end
        fb1_temp_test(:,:,ch_idx)=ect3(:,51:300);
    end
    fb1_test(group_idx,:,:,:)=fb1_temp_test;
    
end

% fb1_group_csp_feature = [ 그룹 수, trial, 2]  
% fb1_test_group_csp_feature = [ 그룹 수, trial,2]
[fb1_group_csp_feature, fb1_test_group_csp_feature] = support_csp_filter(fb1_RH, fb1_RF, fb1_test, rem_sup_ch);
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 8-12 Hz
[fb2_max,fb2_min]=butter(4,[8/50 12/50]);

% suport_RH/RF =[63, 15, 501,112]
% fb2_RH / RF = [63, 15, 112, 250]  -> [63,112,250,15]
for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for rh_trial_idx = 1: num_RH_trial
            
            ect1(rh_trial_idx,:)=filtfilt(fb2_max,fb2_min,squeeze(support_RH(group_idx,ch_idx,:,rh_trial_idx)));
            
        end
        fb2_temp_RH(:,:,ch_idx)=ect1(:,51:300);
    end
    fb2_RH(group_idx,:,:,:)=fb2_temp_RH;
    
end

for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for rf_trial_idx = 1: num_RF_trial
            
            ect2(rf_trial_idx,:)=filtfilt(fb2_max,fb2_min,squeeze(support_RF(group_idx,ch_idx,:,rf_trial_idx)));
            
        end
        fb2_temp_RF(:,:,ch_idx)=ect2(:,51:300);
    end
    fb2_RF(group_idx,:,:,:)=fb2_temp_RF;
    
end

for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for test_trial_idx = 1: num_test_trial
            
            ect3(test_trial_idx,:)=filtfilt(fb2_max,fb2_min,squeeze(support_test(group_idx,ch_idx,:,test_trial_idx)));
            
        end
        fb2_temp_test(:,:,ch_idx)=ect3(:,51:300);
    end
    fb2_test(group_idx,:,:,:)=fb2_temp_test;
    
end

% fb2_group_csp_feature = [ 그룹 수, trial, 2]  
% fb2_test_group_csp_feature = [ 그룹 수, trial,2]
[fb2_group_csp_feature, fb2_test_group_csp_feature] = support_csp_filter(fb2_RH, fb2_RF, fb2_test, rem_sup_ch);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 12-16 Hz
[fb3_max,fb3_min]=butter(4,[12/50 16/50]);

% suport_RH/RF =[63, 15, 501,112]
% fb3_RH / RF = [63, 15, 112, 250]  -> [63,112,250,15]
for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for rh_trial_idx = 1: num_RH_trial
            
            ect1(rh_trial_idx,:)=filtfilt(fb3_max,fb3_min,squeeze(support_RH(group_idx,ch_idx,:,rh_trial_idx)));
            
        end
        fb3_temp_RH(:,:,ch_idx)=ect1(:,51:300);
    end
    fb3_RH(group_idx,:,:,:)=fb3_temp_RH;
    
end

for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for rf_trial_idx = 1: num_RF_trial
            
            ect2(rf_trial_idx,:)=filtfilt(fb3_max,fb3_min,squeeze(support_RF(group_idx,ch_idx,:,rf_trial_idx)));
            
        end
        fb3_temp_RF(:,:,ch_idx)=ect2(:,51:300);
    end
    fb3_RF(group_idx,:,:,:)=fb3_temp_RF;
    
end

for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for test_trial_idx = 1: num_test_trial
            
            ect3(test_trial_idx,:)=filtfilt(fb3_max,fb3_min,squeeze(support_test(group_idx,ch_idx,:,test_trial_idx)));
            
        end
        fb3_temp_test(:,:,ch_idx)=ect3(:,51:300);
    end
    fb3_test(group_idx,:,:,:)=fb3_temp_test;
    
end

% fb3_group_csp_feature = [ 그룹 수, trial, 2]  
% fb3_test_group_csp_feature = [ 그룹 수, trial,2]
[fb3_group_csp_feature, fb3_test_group_csp_feature] = support_csp_filter(fb3_RH, fb3_RF, fb3_test, rem_sup_ch);
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 16-20 Hz
[fb4_max,fb4_min]=butter(4,[16/50 20/50]);

% suport_RH/RF =[63, 15, 501,112]
% fb4_RH / RF = [63, 15, 112, 250]  -> [63,112,250,15]
for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for rh_trial_idx = 1: num_RH_trial
            
            ect1(rh_trial_idx,:)=filtfilt(fb4_max,fb4_min,squeeze(support_RH(group_idx,ch_idx,:,rh_trial_idx)));
            
        end
        fb4_temp_RH(:,:,ch_idx)=ect1(:,51:300);
    end
    fb4_RH(group_idx,:,:,:)=fb4_temp_RH;
    
end

for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for rf_trial_idx = 1: num_RF_trial
            
            ect2(rf_trial_idx,:)=filtfilt(fb4_max,fb4_min,squeeze(support_RF(group_idx,ch_idx,:,rf_trial_idx)));
            
        end
        fb4_temp_RF(:,:,ch_idx)=ect2(:,51:300);
    end
    fb4_RF(group_idx,:,:,:)=fb4_temp_RF;
    
end

for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for test_trial_idx = 1: num_test_trial
            
            ect3(test_trial_idx,:)=filtfilt(fb4_max,fb4_min,squeeze(support_test(group_idx,ch_idx,:,test_trial_idx)));
            
        end
        fb4_temp_test(:,:,ch_idx)=ect3(:,51:300);
    end
    fb4_test(group_idx,:,:,:)=fb4_temp_test;
    
end

% fb4_group_csp_feature = [ 그룹 수, trial, 2]  
% fb4_test_group_csp_feature = [ 그룹 수, trial,2]
[fb4_group_csp_feature, fb4_test_group_csp_feature] = support_csp_filter(fb4_RH, fb4_RF, fb4_test, rem_sup_ch);
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 20-24 Hz
[fb5_max,fb5_min]=butter(4,[20/50 24/50]);

% suport_RH/RF =[63, 15, 501,112]
% fb5_RH / RF = [63, 15, 112, 250]  -> [63,112,250,15]
for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for rh_trial_idx = 1: num_RH_trial
            
            ect1(rh_trial_idx,:)=filtfilt(fb5_max,fb5_min,squeeze(support_RH(group_idx,ch_idx,:,rh_trial_idx)));
            
        end
        fb5_temp_RH(:,:,ch_idx)=ect1(:,51:300);
    end
    fb5_RH(group_idx,:,:,:)=fb5_temp_RH;
    
end

for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for rf_trial_idx = 1: num_RF_trial
            
            ect2(rf_trial_idx,:)=filtfilt(fb5_max,fb5_min,squeeze(support_RF(group_idx,ch_idx,:,rf_trial_idx)));
            
        end
        fb5_temp_RF(:,:,ch_idx)=ect2(:,51:300);
    end
    fb5_RF(group_idx,:,:,:)=fb5_temp_RF;
    
end

for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for test_trial_idx = 1: num_test_trial
            
            ect3(test_trial_idx,:)=filtfilt(fb5_max,fb5_min,squeeze(support_test(group_idx,ch_idx,:,test_trial_idx)));
            
        end
        fb5_temp_test(:,:,ch_idx)=ect3(:,51:300);
    end
    fb5_test(group_idx,:,:,:)=fb5_temp_test;
    
end

% fb5_group_csp_feature = [ 그룹 수, trial, 2]  
% fb5_test_group_csp_feature = [ 그룹 수, trial,2]
[fb5_group_csp_feature, fb5_test_group_csp_feature] = support_csp_filter(fb5_RH, fb5_RF, fb5_test, rem_sup_ch);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 24-28 Hz
[fb6_max,fb6_min]=butter(4,[24/50 28/50]);

% suport_RH/RF =[63, 15, 501,112]
% fb6_RH / RF = [63, 15, 112, 250]  -> [63,112,250,15]
for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for rh_trial_idx = 1: num_RH_trial
            
            ect1(rh_trial_idx,:)=filtfilt(fb6_max,fb6_min,squeeze(support_RH(group_idx,ch_idx,:,rh_trial_idx)));
            
        end
        fb6_temp_RH(:,:,ch_idx)=ect1(:,51:300);
    end
    fb6_RH(group_idx,:,:,:)=fb6_temp_RH;
    
end

for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for rf_trial_idx = 1: num_RF_trial
            
            ect2(rf_trial_idx,:)=filtfilt(fb6_max,fb6_min,squeeze(support_RF(group_idx,ch_idx,:,rf_trial_idx)));
            
        end
        fb6_temp_RF(:,:,ch_idx)=ect2(:,51:300);
    end
    fb6_RF(group_idx,:,:,:)=fb6_temp_RF;
    
end

for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for test_trial_idx = 1: num_test_trial
            
            ect3(test_trial_idx,:)=filtfilt(fb6_max,fb6_min,squeeze(support_test(group_idx,ch_idx,:,test_trial_idx)));
            
        end
        fb6_temp_test(:,:,ch_idx)=ect3(:,51:300);
    end
    fb6_test(group_idx,:,:,:)=fb6_temp_test;
    
end

% fb6_group_csp_feature = [ 그룹 수, trial, 2]  
% fb6_test_group_csp_feature = [ 그룹 수, trial,2]
[fb6_group_csp_feature, fb6_test_group_csp_feature] = support_csp_filter(fb6_RH, fb6_RF, fb6_test, rem_sup_ch);
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 28-32 Hz
[fb7_max,fb7_min]=butter(4,[28/50 32/50]);

% suport_RH/RF =[63, 15, 501,112]
% fb7_RH / RF = [63, 15, 112, 250]  -> [63,112,250,15]
for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for rh_trial_idx = 1: num_RH_trial
            
            ect1(rh_trial_idx,:)=filtfilt(fb7_max,fb7_min,squeeze(support_RH(group_idx,ch_idx,:,rh_trial_idx)));
            
        end
        fb7_temp_RH(:,:,ch_idx)=ect1(:,51:300);
    end
    fb7_RH(group_idx,:,:,:)=fb7_temp_RH;
    
end

for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for rf_trial_idx = 1: num_RF_trial
            
            ect2(rf_trial_idx,:)=filtfilt(fb7_max,fb7_min,squeeze(support_RF(group_idx,ch_idx,:,rf_trial_idx)));
            
        end
        fb7_temp_RF(:,:,ch_idx)=ect2(:,51:300);
    end
    fb7_RF(group_idx,:,:,:)=fb7_temp_RF;
    
end

for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for test_trial_idx = 1: num_test_trial
            
            ect3(test_trial_idx,:)=filtfilt(fb7_max,fb7_min,squeeze(support_test(group_idx,ch_idx,:,test_trial_idx)));
            
        end
        fb7_temp_test(:,:,ch_idx)=ect3(:,51:300);
    end
    fb7_test(group_idx,:,:,:)=fb7_temp_test;
    
end

% fb7_group_csp_feature = [ 그룹 수, trial, 2]  
% fb7_test_group_csp_feature = [ 그룹 수, trial,2]
[fb7_group_csp_feature, fb7_test_group_csp_feature] = support_csp_filter(fb7_RH, fb7_RF, fb7_test, rem_sup_ch);
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 32-36 Hz
[fb8_max,fb8_min]=butter(4,[32/50 36/50]);

% suport_RH/RF =[63, 15, 501,112]
% fb8_RH / RF = [63, 15, 112, 250]  -> [63,112,250,15]
for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for rh_trial_idx = 1: num_RH_trial
            
            ect1(rh_trial_idx,:)=filtfilt(fb8_max,fb8_min,squeeze(support_RH(group_idx,ch_idx,:,rh_trial_idx)));
            
        end
        fb8_temp_RH(:,:,ch_idx)=ect1(:,51:300);
    end
    fb8_RH(group_idx,:,:,:)=fb8_temp_RH;
    
end

for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for rf_trial_idx = 1: num_RF_trial
            
            ect2(rf_trial_idx,:)=filtfilt(fb8_max,fb8_min,squeeze(support_RF(group_idx,ch_idx,:,rf_trial_idx)));
            
        end
        fb8_temp_RF(:,:,ch_idx)=ect2(:,51:300);
    end
    fb8_RF(group_idx,:,:,:)=fb8_temp_RF;
    
end

for group_idx = 1 : num_support
    
    for ch_idx = 1: num_ch_support
        
        for test_trial_idx = 1: num_test_trial
            
            ect3(test_trial_idx,:)=filtfilt(fb8_max,fb8_min,squeeze(support_test(group_idx,ch_idx,:,test_trial_idx)));
            
        end
        fb8_temp_test(:,:,ch_idx)=ect3(:,51:300);
    end
    fb8_test(group_idx,:,:,:)=fb8_temp_test;
    
end

% fb8_group_csp_feature = [ 그룹 수, trial, 2]  
% fb8_test_group_csp_feature = [ 그룹 수, trial,2]
[fb8_group_csp_feature, fb8_test_group_csp_feature] = support_csp_filter(fb8_RH, fb8_RF, fb8_test, rem_sup_ch);
    

%% gathering each FBCSP
% group_feautre = [ 그룹수 ,trial,  2 x (fb수)] = [그룹수, 252, 2 x 8 ]
% group_test_feature = [ 그룹수, trial, 2 x(fb수)] = [그룹수, 28, 2x8]

for group_idx = 1 : num_support
    
    group_csp_feature(group_idx, :, :) = [ squeeze(fb1_group_csp_feature(group_idx, :, :)),  squeeze(fb2_group_csp_feature(group_idx, :, :)), squeeze(fb3_group_csp_feature(group_idx, :, :)), squeeze(fb4_group_csp_feature(group_idx, :, :)), ...
        squeeze(fb5_group_csp_feature(group_idx, :, :)), squeeze(fb6_group_csp_feature(group_idx, :, :)), squeeze(fb7_group_csp_feature(group_idx, :, :)), squeeze(fb8_group_csp_feature(group_idx, :, :))];
end

for group_idx = 1 : num_support
    
    group_test_csp_feature(group_idx, :, :) = [ squeeze(fb1_test_group_csp_feature(group_idx, :, :)),  squeeze(fb2_test_group_csp_feature(group_idx, :, :)), squeeze(fb3_test_group_csp_feature(group_idx, :, :)), squeeze(fb4_test_group_csp_feature(group_idx, :, :)), ...
        squeeze(fb5_test_group_csp_feature(group_idx, :, :)), squeeze(fb6_test_group_csp_feature(group_idx, :, :)), squeeze(fb7_test_group_csp_feature(group_idx, :, :)), squeeze(fb8_test_group_csp_feature(group_idx, :, :))];
end


%% Mutual information for select best two filter banks

%max min 포함
num_fb = 8;

% filterbank 8개라서 무조건 16개의 특성이 나옴
% 이 16개 특성중 어떤 필터뱅크가 best인지 알려면 
% 각 fb의 max에 해당하는 홀수값만 설정하면됨 
max_values = [ 1,3,5,7,9,11,13,15];

for group_idx = 1: num_support
    
   for fb_idx = 1: num_fb
        mi_fb(group_idx, fb_idx, :) = mi(squeeze(group_csp_feature(group_idx, :, max_values(fb_idx):max_values(fb_idx)+1)),original_train_label');
   end 
  [val(group_idx, 1:num_fb), idx(group_idx,1:num_fb) ] = sort(mi_fb(group_idx,:), 'descend');
end


% selected two discriminative filter
for group_idx = 1:num_support
    
    temp_idx(group_idx,:) = idx(group_idx,1:2);
end


% selected high Mi score features
% 각 그룹별 MI에 의한 best filterbank를 선정 
% selected_group_csp_feature = [ 63 , 224, 4]
for group_idx = 1:num_support
          
    selected_group_csp_feature(group_idx, :, 1) = group_csp_feature(group_idx, : , max_values(temp_idx(group_idx,1)) );  
    selected_group_csp_feature(group_idx, :, 2) = group_csp_feature(group_idx, : , max_values(temp_idx(group_idx,1))+1 );  
    selected_group_csp_feature(group_idx, :, 3) = group_csp_feature(group_idx, : , max_values(temp_idx(group_idx,2)) );  
    selected_group_csp_feature(group_idx, :, 4) = group_csp_feature(group_idx, : , max_values(temp_idx(group_idx,1))+1 );  
    
end

% test data를 위해 각 그룹별 best filterbank 기억
for group_idx = 1: num_support
   rem_fb(group_idx,1) = max_values(temp_idx(group_idx,1)); 
   rem_fb(group_idx,2) = max_values(temp_idx(group_idx,2)); 
end

%fisher score에 의해서 어떤 그룹이 가장 좋은지를 선택
for group_idx = 1: num_support
    
    %gRH_csp_feature = [ 112,4 ] 
    gRH_csp_feature = squeeze(selected_group_csp_feature(group_idx, 1:num_RH_trial, :));
    gRF_csp_feature = squeeze(selected_group_csp_feature(group_idx, num_RH_trial+1:end, :));
    
    numerator = norm(mean(gRH_csp_feature,1) - mean(gRF_csp_feature,1),2);
    
    for fisher_rh_idx = 1 : num_RH_trial
        %deno_RH = [ 1x 112]
        deno_RH(fisher_rh_idx) = norm(gRH_csp_feature(fisher_rh_idx,:) - mean(gRH_csp_feature,1), 2);
    end
    
    for fisher_rf_idx = 1 : num_RF_trial
        deno_RF(fisher_rh_idx) = norm(gRF_csp_feature(fisher_rh_idx,:) - mean(gRF_csp_feature,1), 2);
    end
        
    
    
    denominator = (mean(deno_RH,2) + mean(deno_RF,2)) * 0.5; 
    
    Z_fisher(group_idx) =  numerator /  denominator ;
    
end

[sort_Z_fisher_val,sort_Z_fisher_idx]= sort(Z_fisher, 'descend','MissingPlacement','last');

final_selected_support_group = sort_Z_fisher_idx(1);
final_selected_support_fb = rem_fb(sort_Z_fisher_idx(1),:);


%% train

train_data = squeeze(selected_group_csp_feature(final_selected_support_group,:,:));
SVMStruct = fitcsvm(train_data, original_train_label');    

% make test data
test_data(:,:,1) = group_test_csp_feature(final_selected_support_group, : , final_selected_support_fb(1) );  
test_data(:,:,2) = group_test_csp_feature(final_selected_support_group, : , final_selected_support_fb(1)+1);  
test_data(:,:,3) = group_test_csp_feature(final_selected_support_group, : , final_selected_support_fb(2) );  
test_data(:,:,4) = group_test_csp_feature(final_selected_support_group, : , final_selected_support_fb(2)+1);  

% suffle test data
shuffle_test_idx = randperm(28);
shuffle_test_data =  squeeze(test_data(:,shuffle_test_idx,:));
shuffle_csp_test_label =  original_test_label(:,shuffle_test_idx);
     

result = predict(SVMStruct, shuffle_test_data);

correct = 0;
    
for k = 1:length(result)

    if result(k) ==shuffle_csp_test_label(k)
        correct=correct+1;
    end
end

accy = correct/length(result)*100;

disp(accy);
disp(rem_sup_ch(final_selected_support_group));

