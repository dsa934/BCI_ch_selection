%{
Replicated CCS-RCSP (by Jinwoo Lee)

- 3th order butterworth filtering (8-30Hz)
- zscore norm (for raw eeg data) ( mean :0 , std :1 -> default)
- Pearson's correlation analysis (for every pair of eeg channels)
- N(tr) x N x S -> N(tr) x N(s) x S 

%}

%% load band passed data ( applied cross validation )
clear all
load('./data_csp/aw/train/train_10')
load('./data_csp/aw/test/test_10')
load('./data_csp/aw/train/train_label_10')
load('./data_csp/aw/test/test_label_10')

% alpha = [ 0, 0.01, 0.1, 0.2, 0.4, 0.6 ]
% beta = [ 0, 0.001, 0.01, 0.1, 0.2 ]
alpha = 0.01;
beta = 0.01;
train_trial = 252;
test_trial = 28;

%% CCS (Correlation based Channel Selection)

% set N
Ns = 14;

% train data
for trial_idx = 1: train_trial
    
    % zscore norm of each trial
    norm_data(trial_idx,:,:) = zscore(train_data(trial_idx,:,:));
    
    % find relevant channels based on correlation coeficient
    cor_matrix = corrcoef(squeeze(norm_data(trial_idx,:,:))');
    
    % mean value of correlation coeficient
    cor_list = mean(cor_matrix,2);
    [cor_value, cor_idx] = sort(cor_list, 'descend');
    
    % best cor_coef is selected every trials
    mean_cor_list(trial_idx,:) = cor_idx(1:Ns, :);
end

% test data
for trial_idx = 1:test_trial
    norm_test_data(trial_idx,:,:) = zscore(test_data(trial_idx,:,:));
end

% each trial 마다 Ns의 채널을 뽑는다 ( 뽑힌 채널의 수 : N_trial x Ns )
% 여기서 중복을 count하여 descend order로 정렬 후 Ns개 만큼의 채널을 뽑는다 
whole_channel = reshape(mean_cor_list, 1, Ns * train_trial);

uni_channel = unique(whole_channel);

% ch_cnt = [ ch_val, # of ch_cnt]
ch_cnt = zeros(length(uni_channel),2);
ch_cnt(:,1) = uni_channel;

for uni_idx = 1: length(uni_channel)
    
    for idx = 1 : Ns * train_trial
        if uni_channel(uni_idx) == whole_channel(idx)
            ch_cnt(uni_idx,2) = ch_cnt(uni_idx,2) +1;
        end
    end
end
[sort_cnt, idx] = sort(ch_cnt(:,2), 'descend');

temp_ch_list = ch_cnt(idx,:);
ch_list = temp_ch_list(1:Ns);


% channel selection based on correlation coeficient
for ch_idx = 1: length(ch_list)
    sel_train(:,ch_idx,:) = norm_data(:,ch_list(ch_idx),:);
    sel_test(:,ch_idx,:) = norm_test_data(:,ch_list(ch_idx),:);   
end

%% bandpass filtering (3rd order butter worth filtering 8-30Hz)

[bbb,aaa]=butter(3,[8/50 30/50]);

%squeeze -> to prevent filtfilt errors

for channel_idx = 1:length(ch_list)
    
    % train_data
    for train_idx = 1:train_trial
        bp_eeg_train(train_idx, channel_idx, :) = filtfilt(bbb, aaa, squeeze(sel_train(train_idx, channel_idx , :)));
    end
    
    % test_data
    for test_idx = 1:test_trial
        bp_eeg_test(test_idx, channel_idx, :) = filtfilt(bbb, aaa, squeeze(sel_test(test_idx, channel_idx , :)));
    end
end


%% CSP spatial filtering
csp_filter = jw_rcsp(bp_eeg_train, alpha, beta, Ns);
csp_test_filter = jw_rcsp(bp_eeg_test, alpha, beta, Ns);
 

% CSP filtered EEG data & make csp feature
for trial_idx = 1 : train_trial
    %csp filtering of train data
    bp_csp_train = csp_filter * squeeze(bp_eeg_train(trial_idx,:,:));    
    
    % make csp feature 
    % v_k = log( var(z_k) / sigam( var z_i) ) 
    max_var_eeg(trial_idx) = log10( var(bp_csp_train(1,:)) / (var(bp_csp_train(1,:)) + var(bp_csp_train(2,:)) ) );
    min_val_eeg(trial_idx) = log10( var(bp_csp_train(2,:)) / (var(bp_csp_train(1,:)) + var(bp_csp_train(2,:)) ) );   
    %max_var_eeg(trial_idx) = log10( var(bp_csp_train(1,:)));
    %min_val_eeg(trial_idx) = log10( var(bp_csp_train(2,:)));
    
end

csp_feature_train = [max_var_eeg ; min_val_eeg ]';

for trial_idx = 1 : test_trial
    %csp filtering of test data
    bp_csp_test = csp_test_filter * squeeze(bp_eeg_test(trial_idx,:,:));
    
    % make csp feature 
    test_max_var_eeg(trial_idx) = log10( var(bp_csp_test(1,:)) / (var(bp_csp_test(1,:)) + var(bp_csp_test(2,:)) ) );
    test_min_val_eeg(trial_idx) = log10( var(bp_csp_test(2,:)) / (var(bp_csp_test(1,:)) + var(bp_csp_test(2,:)) ) );   
    %test_max_var_eeg(trial_idx) = log10( var(bp_csp_test(1,:)));
    %test_min_val_eeg(trial_idx) = log10( var(bp_csp_test(2,:)));
end
csp_feature_test = [test_max_var_eeg ; test_min_val_eeg ]';


% train & test

%libsvm_options = '-t 2 ';
trained_model = svmtrain32(train_label', csp_feature_train); 

% shuffle test_data
shuffle_idx = randperm(28);
shuffle_csp_test_feature =  csp_feature_test(shuffle_idx,:);
test_label = test_label';
shuffle_csp_test_label =  test_label(shuffle_idx,:);

[predicted_label, accuracy, decision_values] = svmpredict32(shuffle_csp_test_label, shuffle_csp_test_feature, trained_model);
