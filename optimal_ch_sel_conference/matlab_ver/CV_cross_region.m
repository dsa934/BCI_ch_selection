%% load data
clear all

% train_shape = [ 252, 118, 501]
% test_shape = [28, 118, 501]
load('C:\Users\dsa93\Desktop\compare_paper_other_algorithm\park_optimal_channelsel_conference\BCI_com_iii_Iva\data_100Hz\al\train\train_1');
load('C:\Users\dsa93\Desktop\compare_paper_other_algorithm\park_optimal_channelsel_conference\BCI_com_iii_Iva\data_100Hz\al\train\train_label_1');
load('C:\Users\dsa93\Desktop\compare_paper_other_algorithm\park_optimal_channelsel_conference\BCI_com_iii_Iva\data_100Hz\al\test\test_1');
load('C:\Users\dsa93\Desktop\compare_paper_other_algorithm\park_optimal_channelsel_conference\BCI_com_iii_Iva\data_100Hz\al\test\test_label_1');

% set num_of subregion
num_sub_region = 8;

%% 1. covariance matrix based channel selection ( need 20 channels)

% 1-1 bandpass filtering 4 - 32 Hz
[bbb,aaa]=butter(4,[4/50 32/50]);

num_train = 252;
num_test = 28;

for ch_idx =1:118
    % train trials
    for k=1:num_train
        cut_train(k,:)=filtfilt(bbb,aaa,squeeze(train_data(k,ch_idx,:)));
    end
    bp_train(:,:,ch_idx)=cut_train(:,51:300);

end

% seperate left/right train
left_bp_train = bp_train(1:num_train/2,:,:);
right_bp_train = bp_train( (num_train/2)+1:end,:,:);

% 1-2 
% a K -channel mean covariance matrix
% (X'X) / tr(X'X)
for trial_idx = 1:num_train/2
    RH_cov(trial_idx, :, : )= ( squeeze(left_bp_train(trial_idx,:,:))' * squeeze(left_bp_train(trial_idx,:,:)) ) / trace( squeeze(left_bp_train(trial_idx,:,:))' * squeeze(left_bp_train(trial_idx,:,:))  );
end

for trial_idx = 1: num_train/2
    RF_cov(trial_idx, :, : )= ( squeeze(right_bp_train(trial_idx,:,:))' * squeeze(right_bp_train(trial_idx,:,:)) ) / trace( squeeze(right_bp_train(trial_idx,:,:))' * squeeze(right_bp_train(trial_idx,:,:))  );
end

% mean convariance
mean_RH = squeeze(mean(RH_cov, 1));
mean_RF = squeeze(mean(RF_cov, 1));

% calculate channel score matirx V
% 여기시발 논문이 +로 써있어서 3일해맴 시발년아 -임 difference라서
V = abs( mean_RH - mean_RF) ;

% 자기자신과의 조합은 제외 
% for remove_idx = 1 :118
%     V(remove_idx, remove_idx) =0; 
% end

sel_ch_num = 118;
H_channel_list = zeros(20,1);

for ch_idx = 1 : sel_ch_num
    
    if length(unique(H_channel_list(H_channel_list~=0))) <= 20

        max_val = max(max(V));
    
        [max_row, max_col] = find(V==max_val);
        H_channel_list(ch_idx:ch_idx+1,:) = max_row';
    
        V(max_row, max_col) = 0;
    end
end

ck_for_duplicate = H_channel_list;


H_channel_list = H_channel_list(H_channel_list~=0);

% sub_region channel list
sub_region_ch_list = unique(H_channel_list,'stable');
sub_region_ch_list = sub_region_ch_list(1:20);

final_H_list = sub_region_ch_list;


% 중복 count
ck_cnt=zeros(20,2);
ck_cnt(:,1) = final_H_list;

for ck_idx = 1 : 20
    
    for whole_ck = 1: length(ck_for_duplicate)
        
        if final_H_list(ck_idx) == ck_for_duplicate(whole_ck)
            ck_cnt(ck_idx,2) = ck_cnt(ck_idx,2)+1;
        end
    end
    
end
sort_ck = sortrows(ck_cnt,2, 'descend');

target = mode(H_channel_list,'all');

% target channels 

target_val = sort_ck(1,1);

[final_target_idx, ] = find(final_H_list == target_val); 

set_len_region = length(final_H_list)-num_sub_region+1;

%% 2  filterbank signal-> cross-combining region -> filterbanks csp
% original signal : left/right/test_eeg
% filterbank csp 4-32Hz 


%% 2-1. 4-8Hz
% siganl : train/test_data = [ 252/28 , 118, 501 ] (trial, ch, time)
% fb1_RH = [112, 250, 118] 
[fb1_max,fb1_min]=butter(4,[4/50 8/50]);

for ch_idx =1:118
    % train trials
    for k=1:num_train
        fb1_cut_train(k,:)=filtfilt(bbb,aaa,squeeze(train_data(k,ch_idx,:)));
    end
    fb1_bp_train(:,:,ch_idx)=fb1_cut_train(:,51:300);
       
    % test trials 
    for k=1:num_test
        fb1_cut_test(k,:)=filtfilt(bbb,aaa,squeeze(test_data(k,ch_idx,:)));
    end
    fb1_bp_test(:,:,ch_idx)=fb1_cut_test(:,51:300);
end

% seperate train -> left/rightg
fb1_RH_bp_train = fb1_bp_train(1:num_train/2,:,:);
fb1_RF_bp_train = fb1_bp_train( (num_train/2)+1:end,:,:);


% cross combin channels
% fb1_xx_cross_region = [ 14,112,250,8] ( area, trial, time, channel] ->
% X^(p)_i,q, 
for sub_idx = 1: set_len_region
    % left
    fb1_RH_cross_region(sub_idx, : ,: ,1) = fb1_RH_bp_train(:,:, final_H_list(final_target_idx));
    fb1_RH_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb1_RH_bp_train(:,:, final_H_list(sub_idx : sub_idx+num_sub_region-1));
    
    % right
    fb1_RF_cross_region(sub_idx, : ,: ,1) = fb1_RF_bp_train(:,:, final_H_list(final_target_idx));
    fb1_RF_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb1_RF_bp_train(:,:, final_H_list(sub_idx: sub_idx+num_sub_region-1));
        
    % test
    fb1_test_cross_region(sub_idx, : ,: ,1) = fb1_bp_test(:,:, final_H_list(final_target_idx));
    fb1_test_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb1_bp_test(:,:, final_H_list(sub_idx: sub_idx+num_sub_region-1));        
end

% filter outputshpae= [ 2x(1+adjacant channels)]
% csp_filter= [14, 2 , 7 ] 
% fb1_RH_cross_region = [ 14, 112, 250, 7] ;
fb1_csp_filter = cross_region_csp(fb1_RH_cross_region, fb1_RF_cross_region);

for region_idx = 1: set_len_region
    
    % left csp filtered eeg
    % RH_Z = [14, 2,250]
    for trial_idx= 1:num_train/2
        fb1_RH_Z(region_idx, : ,: ) = squeeze(fb1_csp_filter(region_idx,:,:)) * squeeze(fb1_RH_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb1_RH_max_val_z(region_idx, trial_idx) = log10( var(fb1_RH_Z(region_idx,1,:)));% / (var(RH_Z(region_idx,1,:)) + var(RH_Z(region_idx,2,:))) );
        fb1_RH_min_val_z(region_idx, trial_idx) = log10( var(fb1_RH_Z(region_idx,2,:)));% / (var(RH_Z(region_idx,1,:)) + var(RH_Z(region_idx,2,:))));
    end
    RH_fb1_csp_feature(region_idx, 1, : ) = fb1_RH_max_val_z(region_idx, : ,: );
    RH_fb1_csp_feature(region_idx, 2, : ) = fb1_RH_min_val_z(region_idx, : ,: );
    
    
    % right
    for trial_idx= 1:num_train/2
        fb1_RF_Z(region_idx, : ,: ) = squeeze(fb1_csp_filter(region_idx,:,:)) * squeeze(fb1_RF_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb1_RF_max_val_z(region_idx, trial_idx) = log10( var(fb1_RF_Z(region_idx,1,:)));%/ (var(RF_Z(region_idx,1,:)) + var(RF_Z(region_idx,2,:))));
        fb1_RF_min_val_z(region_idx, trial_idx) = log10( var(fb1_RF_Z(region_idx,2,:)));%/ (var(RF_Z(region_idx,1,:)) + var(RF_Z(region_idx,2,:))));
    end
    RF_fb1_csp_feature(region_idx, 1, : ) = fb1_RF_max_val_z(region_idx, : ,: );
    RF_fb1_csp_feature(region_idx, 2, : ) = fb1_RF_min_val_z(region_idx, : ,: );
    
    % test
    for trial_idx =1:num_test
        fb1_TE_Z(region_idx, :, : ) = squeeze(fb1_csp_filter(region_idx,:,:)) * squeeze(fb1_test_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb1_TE_max_val_z(region_idx, trial_idx) = log10( var(fb1_TE_Z(region_idx,1,:)));%/ (var(TE_Z(region_idx,1,:)) + var(TE_Z(region_idx,2,:))));
        fb1_TE_min_val_z(region_idx, trial_idx) = log10( var(fb1_TE_Z(region_idx,2,:)));%/ (var(TE_Z(region_idx,1,:)) + var(TE_Z(region_idx,2,:))));
    end
    TE_fb1_csp_feature(region_idx, 1, : ) = fb1_TE_max_val_z(region_idx, : ,: );
    TE_fb1_csp_feature(region_idx, 2, : ) = fb1_TE_min_val_z(region_idx, : ,: );
            
end

% left right summation
for region_idx = 1: set_len_region
    fb1_train(region_idx,:,:) = [ squeeze(RH_fb1_csp_feature(region_idx,:,:)),squeeze(RF_fb1_csp_feature(region_idx,:,:)) ]';
end
%
last_fb1_test = TE_fb1_csp_feature;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2-1. 8-12Hz
% siganl : train/test_data = [ 252/28 , 118, 501 ] (trial, ch, time)
% fb2_RH = [112, 250, 118] 
[fb2_max,fb2_min]=butter(4,[8/50 12/50]);

for ch_idx =1:118
    % train trials
    for k=1:num_train
        fb2_cut_train(k,:)=filtfilt(bbb,aaa,squeeze(train_data(k,ch_idx,:)));
    end
    fb2_bp_train(:,:,ch_idx)=fb2_cut_train(:,51:300);
       
    % test trials 
    for k=1:num_test
        fb2_cut_test(k,:)=filtfilt(bbb,aaa,squeeze(test_data(k,ch_idx,:)));
    end
    fb2_bp_test(:,:,ch_idx)=fb2_cut_test(:,51:300);
end

% seperate train -> left/rightg
fb2_RH_bp_train = fb2_bp_train(1:num_train/2,:,:);
fb2_RF_bp_train = fb2_bp_train( (num_train/2)+1:end,:,:);


% cross combin channels
% fb2_xx_cross_region = [ 14,112,250,8] ( area, trial, time, channel] ->
% X^(p)_i,q, 
for sub_idx = 1: set_len_region
    % left
    fb2_RH_cross_region(sub_idx, : ,: ,1) = fb2_RH_bp_train(:,:, final_H_list(final_target_idx));
    fb2_RH_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb2_RH_bp_train(:,:, final_H_list(sub_idx : sub_idx+num_sub_region-1));
    
    % right
    fb2_RF_cross_region(sub_idx, : ,: ,1) = fb2_RF_bp_train(:,:, final_H_list(final_target_idx));
    fb2_RF_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb2_RF_bp_train(:,:, final_H_list(sub_idx: sub_idx+num_sub_region-1));
        
    % test
    fb2_test_cross_region(sub_idx, : ,: ,1) = fb2_bp_test(:,:, final_H_list(final_target_idx));
    fb2_test_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb2_bp_test(:,:, final_H_list(sub_idx: sub_idx+num_sub_region-1));        
end

% filter outputshpae= [ 2x(1+adjacant channels)]
% csp_filter= [14, 2 , 7 ] 
% fb2_RH_cross_region = [ 14, 112, 250, 7] ;
fb2_csp_filter = cross_region_csp(fb2_RH_cross_region, fb2_RF_cross_region);

for region_idx = 1: set_len_region
    
    % left csp filtered eeg
    % RH_Z = [14, 2,250]
    for trial_idx= 1:num_train/2
        fb2_RH_Z(region_idx, : ,: ) = squeeze(fb2_csp_filter(region_idx,:,:)) * squeeze(fb2_RH_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb2_RH_max_val_z(region_idx, trial_idx) = log10( var(fb2_RH_Z(region_idx,1,:)));% / (var(RH_Z(region_idx,1,:)) + var(RH_Z(region_idx,2,:))) );
        fb2_RH_min_val_z(region_idx, trial_idx) = log10( var(fb2_RH_Z(region_idx,2,:)));% / (var(RH_Z(region_idx,1,:)) + var(RH_Z(region_idx,2,:))));
    end
    RH_fb2_csp_feature(region_idx, 1, : ) = fb2_RH_max_val_z(region_idx, : ,: );
    RH_fb2_csp_feature(region_idx, 2, : ) = fb2_RH_min_val_z(region_idx, : ,: );
    
    
    % right
    for trial_idx= 1:num_train/2
        fb2_RF_Z(region_idx, : ,: ) = squeeze(fb2_csp_filter(region_idx,:,:)) * squeeze(fb2_RF_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb2_RF_max_val_z(region_idx, trial_idx) = log10( var(fb2_RF_Z(region_idx,1,:)));%/ (var(RF_Z(region_idx,1,:)) + var(RF_Z(region_idx,2,:))));
        fb2_RF_min_val_z(region_idx, trial_idx) = log10( var(fb2_RF_Z(region_idx,2,:)));%/ (var(RF_Z(region_idx,1,:)) + var(RF_Z(region_idx,2,:))));
    end
    RF_fb2_csp_feature(region_idx, 1, : ) = fb2_RF_max_val_z(region_idx, : ,: );
    RF_fb2_csp_feature(region_idx, 2, : ) = fb2_RF_min_val_z(region_idx, : ,: );
    
    % test
    for trial_idx =1:num_test
        fb2_TE_Z(region_idx, :, : ) = squeeze(fb2_csp_filter(region_idx,:,:)) * squeeze(fb2_test_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb2_TE_max_val_z(region_idx, trial_idx) = log10( var(fb2_TE_Z(region_idx,1,:)));%/ (var(TE_Z(region_idx,1,:)) + var(TE_Z(region_idx,2,:))));
        fb2_TE_min_val_z(region_idx, trial_idx) = log10( var(fb2_TE_Z(region_idx,2,:)));%/ (var(TE_Z(region_idx,1,:)) + var(TE_Z(region_idx,2,:))));
    end
    TE_fb2_csp_feature(region_idx, 1, : ) = fb2_TE_max_val_z(region_idx, : ,: );
    TE_fb2_csp_feature(region_idx, 2, : ) = fb2_TE_min_val_z(region_idx, : ,: );
            
end

% left right summation
for region_idx = 1: set_len_region
    fb2_train(region_idx,:,:) = [ squeeze(RH_fb2_csp_feature(region_idx,:,:)),squeeze(RF_fb2_csp_feature(region_idx,:,:)) ]';
end
%
last_fb2_test = TE_fb2_csp_feature;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2-1. 12-16Hz
% siganl : train/test_data = [ 252/28 , 118, 501 ] (trial, ch, time)
% fb3_RH = [112, 250, 118] 
[fb3_max,fb3_min]=butter(4,[12/50 16/50]);

for ch_idx =1:118
    % train trials
    for k=1:num_train
        fb3_cut_train(k,:)=filtfilt(bbb,aaa,squeeze(train_data(k,ch_idx,:)));
    end
    fb3_bp_train(:,:,ch_idx)=fb3_cut_train(:,51:300);
       
    % test trials 
    for k=1:num_test
        fb3_cut_test(k,:)=filtfilt(bbb,aaa,squeeze(test_data(k,ch_idx,:)));
    end
    fb3_bp_test(:,:,ch_idx)=fb3_cut_test(:,51:300);
end

% seperate train -> left/rightg
fb3_RH_bp_train = fb3_bp_train(1:num_train/2,:,:);
fb3_RF_bp_train = fb3_bp_train( (num_train/2)+1:end,:,:);


% cross combin channels
% fb3_xx_cross_region = [ 14,112,250,8] ( area, trial, time, channel] ->
% X^(p)_i,q, 
for sub_idx = 1: set_len_region
    % left
    fb3_RH_cross_region(sub_idx, : ,: ,1) = fb3_RH_bp_train(:,:, final_H_list(final_target_idx));
    fb3_RH_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb3_RH_bp_train(:,:, final_H_list(sub_idx : sub_idx+num_sub_region-1));
    
    % right
    fb3_RF_cross_region(sub_idx, : ,: ,1) = fb3_RF_bp_train(:,:, final_H_list(final_target_idx));
    fb3_RF_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb3_RF_bp_train(:,:, final_H_list(sub_idx: sub_idx+num_sub_region-1));
        
    % test
    fb3_test_cross_region(sub_idx, : ,: ,1) = fb3_bp_test(:,:, final_H_list(final_target_idx));
    fb3_test_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb3_bp_test(:,:, final_H_list(sub_idx: sub_idx+num_sub_region-1));        
end

% filter outputshpae= [ 2x(1+adjacant channels)]
% csp_filter= [14, 2 , 7 ] 
% fb3_RH_cross_region = [ 14, 112, 250, 7] ;
fb3_csp_filter = cross_region_csp(fb3_RH_cross_region, fb3_RF_cross_region);

for region_idx = 1: set_len_region
    
    % left csp filtered eeg
    % RH_Z = [14, 2,250]
    for trial_idx= 1:num_train/2
        fb3_RH_Z(region_idx, : ,: ) = squeeze(fb3_csp_filter(region_idx,:,:)) * squeeze(fb3_RH_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb3_RH_max_val_z(region_idx, trial_idx) = log10( var(fb3_RH_Z(region_idx,1,:)));% / (var(RH_Z(region_idx,1,:)) + var(RH_Z(region_idx,2,:))) );
        fb3_RH_min_val_z(region_idx, trial_idx) = log10( var(fb3_RH_Z(region_idx,2,:)));% / (var(RH_Z(region_idx,1,:)) + var(RH_Z(region_idx,2,:))));
    end
    RH_fb3_csp_feature(region_idx, 1, : ) = fb3_RH_max_val_z(region_idx, : ,: );
    RH_fb3_csp_feature(region_idx, 2, : ) = fb3_RH_min_val_z(region_idx, : ,: );
    
    
    % right
    for trial_idx= 1:num_train/2
        fb3_RF_Z(region_idx, : ,: ) = squeeze(fb3_csp_filter(region_idx,:,:)) * squeeze(fb3_RF_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb3_RF_max_val_z(region_idx, trial_idx) = log10( var(fb3_RF_Z(region_idx,1,:)));%/ (var(RF_Z(region_idx,1,:)) + var(RF_Z(region_idx,2,:))));
        fb3_RF_min_val_z(region_idx, trial_idx) = log10( var(fb3_RF_Z(region_idx,2,:)));%/ (var(RF_Z(region_idx,1,:)) + var(RF_Z(region_idx,2,:))));
    end
    RF_fb3_csp_feature(region_idx, 1, : ) = fb3_RF_max_val_z(region_idx, : ,: );
    RF_fb3_csp_feature(region_idx, 2, : ) = fb3_RF_min_val_z(region_idx, : ,: );
    
    % test
    for trial_idx =1:num_test
        fb3_TE_Z(region_idx, :, : ) = squeeze(fb3_csp_filter(region_idx,:,:)) * squeeze(fb3_test_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb3_TE_max_val_z(region_idx, trial_idx) = log10( var(fb3_TE_Z(region_idx,1,:)));%/ (var(TE_Z(region_idx,1,:)) + var(TE_Z(region_idx,2,:))));
        fb3_TE_min_val_z(region_idx, trial_idx) = log10( var(fb3_TE_Z(region_idx,2,:)));%/ (var(TE_Z(region_idx,1,:)) + var(TE_Z(region_idx,2,:))));
    end
    TE_fb3_csp_feature(region_idx, 1, : ) = fb3_TE_max_val_z(region_idx, : ,: );
    TE_fb3_csp_feature(region_idx, 2, : ) = fb3_TE_min_val_z(region_idx, : ,: );
            
end

% left right summation
for region_idx = 1: set_len_region
    fb3_train(region_idx,:,:) = [ squeeze(RH_fb3_csp_feature(region_idx,:,:)),squeeze(RF_fb3_csp_feature(region_idx,:,:)) ]';
end
%
last_fb3_test = TE_fb3_csp_feature;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2-1. 16-20Hz
% siganl : train/test_data = [ 252/28 , 118, 501 ] (trial, ch, time)
% fb4_RH = [112, 250, 118] 
[fb4_max,fb4_min]=butter(4,[16/50 20/50]);

for ch_idx =1:118
    % train trials
    for k=1:num_train
        fb4_cut_train(k,:)=filtfilt(bbb,aaa,squeeze(train_data(k,ch_idx,:)));
    end
    fb4_bp_train(:,:,ch_idx)=fb4_cut_train(:,51:300);
       
    % test trials 
    for k=1:num_test
        fb4_cut_test(k,:)=filtfilt(bbb,aaa,squeeze(test_data(k,ch_idx,:)));
    end
    fb4_bp_test(:,:,ch_idx)=fb4_cut_test(:,51:300);
end

% seperate train -> left/rightg
fb4_RH_bp_train = fb4_bp_train(1:num_train/2,:,:);
fb4_RF_bp_train = fb4_bp_train( (num_train/2)+1:end,:,:);


% cross combin channels
% fb4_xx_cross_region = [ 14,112,250,8] ( area, trial, time, channel] ->
% X^(p)_i,q, 
for sub_idx = 1: set_len_region
    % left
    fb4_RH_cross_region(sub_idx, : ,: ,1) = fb4_RH_bp_train(:,:, final_H_list(final_target_idx));
    fb4_RH_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb4_RH_bp_train(:,:, final_H_list(sub_idx : sub_idx+num_sub_region-1));
    
    % right
    fb4_RF_cross_region(sub_idx, : ,: ,1) = fb4_RF_bp_train(:,:, final_H_list(final_target_idx));
    fb4_RF_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb4_RF_bp_train(:,:, final_H_list(sub_idx: sub_idx+num_sub_region-1));
        
    % test
    fb4_test_cross_region(sub_idx, : ,: ,1) = fb4_bp_test(:,:, final_H_list(final_target_idx));
    fb4_test_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb4_bp_test(:,:, final_H_list(sub_idx: sub_idx+num_sub_region-1));        
end

% filter outputshpae= [ 2x(1+adjacant channels)]
% csp_filter= [14, 2 , 7 ] 
% fb4_RH_cross_region = [ 14, 112, 250, 7] ;
fb4_csp_filter = cross_region_csp(fb4_RH_cross_region, fb4_RF_cross_region);

for region_idx = 1: set_len_region
    
    % left csp filtered eeg
    % RH_Z = [14, 2,250]
    for trial_idx= 1:num_train/2
        fb4_RH_Z(region_idx, : ,: ) = squeeze(fb4_csp_filter(region_idx,:,:)) * squeeze(fb4_RH_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb4_RH_max_val_z(region_idx, trial_idx) = log10( var(fb4_RH_Z(region_idx,1,:)));% / (var(RH_Z(region_idx,1,:)) + var(RH_Z(region_idx,2,:))) );
        fb4_RH_min_val_z(region_idx, trial_idx) = log10( var(fb4_RH_Z(region_idx,2,:)));% / (var(RH_Z(region_idx,1,:)) + var(RH_Z(region_idx,2,:))));
    end
    RH_fb4_csp_feature(region_idx, 1, : ) = fb4_RH_max_val_z(region_idx, : ,: );
    RH_fb4_csp_feature(region_idx, 2, : ) = fb4_RH_min_val_z(region_idx, : ,: );
    
    
    % right
    for trial_idx= 1:num_train/2
        fb4_RF_Z(region_idx, : ,: ) = squeeze(fb4_csp_filter(region_idx,:,:)) * squeeze(fb4_RF_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb4_RF_max_val_z(region_idx, trial_idx) = log10( var(fb4_RF_Z(region_idx,1,:)));%/ (var(RF_Z(region_idx,1,:)) + var(RF_Z(region_idx,2,:))));
        fb4_RF_min_val_z(region_idx, trial_idx) = log10( var(fb4_RF_Z(region_idx,2,:)));%/ (var(RF_Z(region_idx,1,:)) + var(RF_Z(region_idx,2,:))));
    end
    RF_fb4_csp_feature(region_idx, 1, : ) = fb4_RF_max_val_z(region_idx, : ,: );
    RF_fb4_csp_feature(region_idx, 2, : ) = fb4_RF_min_val_z(region_idx, : ,: );
    
    % test
    for trial_idx =1:num_test
        fb4_TE_Z(region_idx, :, : ) = squeeze(fb4_csp_filter(region_idx,:,:)) * squeeze(fb4_test_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb4_TE_max_val_z(region_idx, trial_idx) = log10( var(fb4_TE_Z(region_idx,1,:)));%/ (var(TE_Z(region_idx,1,:)) + var(TE_Z(region_idx,2,:))));
        fb4_TE_min_val_z(region_idx, trial_idx) = log10( var(fb4_TE_Z(region_idx,2,:)));%/ (var(TE_Z(region_idx,1,:)) + var(TE_Z(region_idx,2,:))));
    end
    TE_fb4_csp_feature(region_idx, 1, : ) = fb4_TE_max_val_z(region_idx, : ,: );
    TE_fb4_csp_feature(region_idx, 2, : ) = fb4_TE_min_val_z(region_idx, : ,: );
            
end

% left right summation
for region_idx = 1: set_len_region
    fb4_train(region_idx,:,:) = [ squeeze(RH_fb4_csp_feature(region_idx,:,:)),squeeze(RF_fb4_csp_feature(region_idx,:,:)) ]';
end
%
last_fb4_test = TE_fb4_csp_feature;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2-1. 8-12Hz
% siganl : train/test_data = [ 252/28 , 118, 501 ] (trial, ch, time)
% fb5_RH = [112, 250, 118] 
[fb5_max,fb5_min]=butter(4,[20/50 24/50]);

for ch_idx =1:118
    % train trials
    for k=1:num_train
        fb5_cut_train(k,:)=filtfilt(bbb,aaa,squeeze(train_data(k,ch_idx,:)));
    end
    fb5_bp_train(:,:,ch_idx)=fb5_cut_train(:,51:300);
       
    % test trials 
    for k=1:num_test
        fb5_cut_test(k,:)=filtfilt(bbb,aaa,squeeze(test_data(k,ch_idx,:)));
    end
    fb5_bp_test(:,:,ch_idx)=fb5_cut_test(:,51:300);
end

% seperate train -> left/rightg
fb5_RH_bp_train = fb5_bp_train(1:num_train/2,:,:);
fb5_RF_bp_train = fb5_bp_train( (num_train/2)+1:end,:,:);


% cross combin channels
% fb5_xx_cross_region = [ 14,112,250,8] ( area, trial, time, channel] ->
% X^(p)_i,q, 
for sub_idx = 1: set_len_region
    % left
    fb5_RH_cross_region(sub_idx, : ,: ,1) = fb5_RH_bp_train(:,:, final_H_list(final_target_idx));
    fb5_RH_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb5_RH_bp_train(:,:, final_H_list(sub_idx : sub_idx+num_sub_region-1));
    
    % right
    fb5_RF_cross_region(sub_idx, : ,: ,1) = fb5_RF_bp_train(:,:, final_H_list(final_target_idx));
    fb5_RF_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb5_RF_bp_train(:,:, final_H_list(sub_idx: sub_idx+num_sub_region-1));
        
    % test
    fb5_test_cross_region(sub_idx, : ,: ,1) = fb5_bp_test(:,:, final_H_list(final_target_idx));
    fb5_test_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb5_bp_test(:,:, final_H_list(sub_idx: sub_idx+num_sub_region-1));        
end

% filter outputshpae= [ 2x(1+adjacant channels)]
% csp_filter= [14, 2 , 7 ] 
% fb5_RH_cross_region = [ 14, 112, 250, 7] ;
fb5_csp_filter = cross_region_csp(fb5_RH_cross_region, fb5_RF_cross_region);

for region_idx = 1: set_len_region
    
    % left csp filtered eeg
    % RH_Z = [14, 2,250]
    for trial_idx= 1:num_train/2
        fb5_RH_Z(region_idx, : ,: ) = squeeze(fb5_csp_filter(region_idx,:,:)) * squeeze(fb5_RH_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb5_RH_max_val_z(region_idx, trial_idx) = log10( var(fb5_RH_Z(region_idx,1,:)));% / (var(RH_Z(region_idx,1,:)) + var(RH_Z(region_idx,2,:))) );
        fb5_RH_min_val_z(region_idx, trial_idx) = log10( var(fb5_RH_Z(region_idx,2,:)));% / (var(RH_Z(region_idx,1,:)) + var(RH_Z(region_idx,2,:))));
    end
    RH_fb5_csp_feature(region_idx, 1, : ) = fb5_RH_max_val_z(region_idx, : ,: );
    RH_fb5_csp_feature(region_idx, 2, : ) = fb5_RH_min_val_z(region_idx, : ,: );
    
    
    % right
    for trial_idx= 1:num_train/2
        fb5_RF_Z(region_idx, : ,: ) = squeeze(fb5_csp_filter(region_idx,:,:)) * squeeze(fb5_RF_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb5_RF_max_val_z(region_idx, trial_idx) = log10( var(fb5_RF_Z(region_idx,1,:)));%/ (var(RF_Z(region_idx,1,:)) + var(RF_Z(region_idx,2,:))));
        fb5_RF_min_val_z(region_idx, trial_idx) = log10( var(fb5_RF_Z(region_idx,2,:)));%/ (var(RF_Z(region_idx,1,:)) + var(RF_Z(region_idx,2,:))));
    end
    RF_fb5_csp_feature(region_idx, 1, : ) = fb5_RF_max_val_z(region_idx, : ,: );
    RF_fb5_csp_feature(region_idx, 2, : ) = fb5_RF_min_val_z(region_idx, : ,: );
    
    % test
    for trial_idx =1:num_test
        fb5_TE_Z(region_idx, :, : ) = squeeze(fb5_csp_filter(region_idx,:,:)) * squeeze(fb5_test_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb5_TE_max_val_z(region_idx, trial_idx) = log10( var(fb5_TE_Z(region_idx,1,:)));%/ (var(TE_Z(region_idx,1,:)) + var(TE_Z(region_idx,2,:))));
        fb5_TE_min_val_z(region_idx, trial_idx) = log10( var(fb5_TE_Z(region_idx,2,:)));%/ (var(TE_Z(region_idx,1,:)) + var(TE_Z(region_idx,2,:))));
    end
    TE_fb5_csp_feature(region_idx, 1, : ) = fb5_TE_max_val_z(region_idx, : ,: );
    TE_fb5_csp_feature(region_idx, 2, : ) = fb5_TE_min_val_z(region_idx, : ,: );
            
end

% left right summation
for region_idx = 1: set_len_region
    fb5_train(region_idx,:,:) = [ squeeze(RH_fb5_csp_feature(region_idx,:,:)),squeeze(RF_fb5_csp_feature(region_idx,:,:)) ]';
end
%
last_fb5_test = TE_fb5_csp_feature;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2-1. 8-12Hz
% siganl : train/test_data = [ 252/28 , 118, 501 ] (trial, ch, time)
% fb6_RH = [112, 250, 118] 
[fb6_max,fb6_min]=butter(4,[24/50 28/50]);

for ch_idx =1:118
    % train trials
    for k=1:num_train
        fb6_cut_train(k,:)=filtfilt(bbb,aaa,squeeze(train_data(k,ch_idx,:)));
    end
    fb6_bp_train(:,:,ch_idx)=fb6_cut_train(:,51:300);
       
    % test trials 
    for k=1:num_test
        fb6_cut_test(k,:)=filtfilt(bbb,aaa,squeeze(test_data(k,ch_idx,:)));
    end
    fb6_bp_test(:,:,ch_idx)=fb6_cut_test(:,51:300);
end

% seperate train -> left/rightg
fb6_RH_bp_train = fb6_bp_train(1:num_train/2,:,:);
fb6_RF_bp_train = fb6_bp_train( (num_train/2)+1:end,:,:);


% cross combin channels
% fb6_xx_cross_region = [ 14,112,250,8] ( area, trial, time, channel] ->
% X^(p)_i,q, 
for sub_idx = 1: set_len_region
    % left
    fb6_RH_cross_region(sub_idx, : ,: ,1) = fb6_RH_bp_train(:,:, final_H_list(final_target_idx));
    fb6_RH_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb6_RH_bp_train(:,:, final_H_list(sub_idx : sub_idx+num_sub_region-1));
    
    % right
    fb6_RF_cross_region(sub_idx, : ,: ,1) = fb6_RF_bp_train(:,:, final_H_list(final_target_idx));
    fb6_RF_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb6_RF_bp_train(:,:, final_H_list(sub_idx: sub_idx+num_sub_region-1));
        
    % test
    fb6_test_cross_region(sub_idx, : ,: ,1) = fb6_bp_test(:,:, final_H_list(final_target_idx));
    fb6_test_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb6_bp_test(:,:, final_H_list(sub_idx: sub_idx+num_sub_region-1));        
end

% filter outputshpae= [ 2x(1+adjacant channels)]
% csp_filter= [14, 2 , 7 ] 
% fb6_RH_cross_region = [ 14, 112, 250, 7] ;
fb6_csp_filter = cross_region_csp(fb6_RH_cross_region, fb6_RF_cross_region);

for region_idx = 1: set_len_region
    
    % left csp filtered eeg
    % RH_Z = [14, 2,250]
    for trial_idx= 1:num_train/2
        fb6_RH_Z(region_idx, : ,: ) = squeeze(fb6_csp_filter(region_idx,:,:)) * squeeze(fb6_RH_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb6_RH_max_val_z(region_idx, trial_idx) = log10( var(fb6_RH_Z(region_idx,1,:)));% / (var(RH_Z(region_idx,1,:)) + var(RH_Z(region_idx,2,:))) );
        fb6_RH_min_val_z(region_idx, trial_idx) = log10( var(fb6_RH_Z(region_idx,2,:)));% / (var(RH_Z(region_idx,1,:)) + var(RH_Z(region_idx,2,:))));
    end
    RH_fb6_csp_feature(region_idx, 1, : ) = fb6_RH_max_val_z(region_idx, : ,: );
    RH_fb6_csp_feature(region_idx, 2, : ) = fb6_RH_min_val_z(region_idx, : ,: );
    
    
    % right
    for trial_idx= 1:num_train/2
        fb6_RF_Z(region_idx, : ,: ) = squeeze(fb6_csp_filter(region_idx,:,:)) * squeeze(fb6_RF_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb6_RF_max_val_z(region_idx, trial_idx) = log10( var(fb6_RF_Z(region_idx,1,:)));%/ (var(RF_Z(region_idx,1,:)) + var(RF_Z(region_idx,2,:))));
        fb6_RF_min_val_z(region_idx, trial_idx) = log10( var(fb6_RF_Z(region_idx,2,:)));%/ (var(RF_Z(region_idx,1,:)) + var(RF_Z(region_idx,2,:))));
    end
    RF_fb6_csp_feature(region_idx, 1, : ) = fb6_RF_max_val_z(region_idx, : ,: );
    RF_fb6_csp_feature(region_idx, 2, : ) = fb6_RF_min_val_z(region_idx, : ,: );
    
    % test
    for trial_idx =1:num_test
        fb6_TE_Z(region_idx, :, : ) = squeeze(fb6_csp_filter(region_idx,:,:)) * squeeze(fb6_test_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb6_TE_max_val_z(region_idx, trial_idx) = log10( var(fb6_TE_Z(region_idx,1,:)));%/ (var(TE_Z(region_idx,1,:)) + var(TE_Z(region_idx,2,:))));
        fb6_TE_min_val_z(region_idx, trial_idx) = log10( var(fb6_TE_Z(region_idx,2,:)));%/ (var(TE_Z(region_idx,1,:)) + var(TE_Z(region_idx,2,:))));
    end
    TE_fb6_csp_feature(region_idx, 1, : ) = fb6_TE_max_val_z(region_idx, : ,: );
    TE_fb6_csp_feature(region_idx, 2, : ) = fb6_TE_min_val_z(region_idx, : ,: );
            
end

% left right summation
for region_idx = 1: set_len_region
    fb6_train(region_idx,:,:) = [ squeeze(RH_fb6_csp_feature(region_idx,:,:)),squeeze(RF_fb6_csp_feature(region_idx,:,:)) ]';
end
%
last_fb6_test = TE_fb6_csp_feature;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2-1. 8-12Hz
% siganl : train/test_data = [ 252/28 , 118, 501 ] (trial, ch, time)
% fb7_RH = [112, 250, 118] 
[fb7_max,fb7_min]=butter(4,[28/50 32/50]);

for ch_idx =1:118
    % train trials
    for k=1:num_train
        fb7_cut_train(k,:)=filtfilt(bbb,aaa,squeeze(train_data(k,ch_idx,:)));
    end
    fb7_bp_train(:,:,ch_idx)=fb7_cut_train(:,51:300);
       
    % test trials 
    for k=1:num_test
        fb7_cut_test(k,:)=filtfilt(bbb,aaa,squeeze(test_data(k,ch_idx,:)));
    end
    fb7_bp_test(:,:,ch_idx)=fb7_cut_test(:,51:300);
end

% seperate train -> left/rightg
fb7_RH_bp_train = fb7_bp_train(1:num_train/2,:,:);
fb7_RF_bp_train = fb7_bp_train( (num_train/2)+1:end,:,:);


% cross combin channels
% fb7_xx_cross_region = [ 14,112,250,8] ( area, trial, time, channel] ->
% X^(p)_i,q, 
for sub_idx = 1: set_len_region
    % left
    fb7_RH_cross_region(sub_idx, : ,: ,1) = fb7_RH_bp_train(:,:, final_H_list(final_target_idx));
    fb7_RH_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb7_RH_bp_train(:,:, final_H_list(sub_idx : sub_idx+num_sub_region-1));
    
    % right
    fb7_RF_cross_region(sub_idx, : ,: ,1) = fb7_RF_bp_train(:,:, final_H_list(final_target_idx));
    fb7_RF_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb7_RF_bp_train(:,:, final_H_list(sub_idx: sub_idx+num_sub_region-1));
        
    % test
    fb7_test_cross_region(sub_idx, : ,: ,1) = fb7_bp_test(:,:, final_H_list(final_target_idx));
    fb7_test_cross_region(sub_idx, : ,: ,2:num_sub_region+1) = fb7_bp_test(:,:, final_H_list(sub_idx: sub_idx+num_sub_region-1));        
end

% filter outputshpae= [ 2x(1+adjacant channels)]
% csp_filter= [14, 2 , 7 ] 
% fb7_RH_cross_region = [ 14, 112, 250, 7] ;
fb7_csp_filter = cross_region_csp(fb7_RH_cross_region, fb7_RF_cross_region);

for region_idx = 1: set_len_region
    
    % left csp filtered eeg
    % RH_Z = [14, 2,250]
    for trial_idx= 1:num_train/2
        fb7_RH_Z(region_idx, : ,: ) = squeeze(fb7_csp_filter(region_idx,:,:)) * squeeze(fb7_RH_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb7_RH_max_val_z(region_idx, trial_idx) = log10( var(fb7_RH_Z(region_idx,1,:)));% / (var(RH_Z(region_idx,1,:)) + var(RH_Z(region_idx,2,:))) );
        fb7_RH_min_val_z(region_idx, trial_idx) = log10( var(fb7_RH_Z(region_idx,2,:)));% / (var(RH_Z(region_idx,1,:)) + var(RH_Z(region_idx,2,:))));
    end
    RH_fb7_csp_feature(region_idx, 1, : ) = fb7_RH_max_val_z(region_idx, : ,: );
    RH_fb7_csp_feature(region_idx, 2, : ) = fb7_RH_min_val_z(region_idx, : ,: );
    
    
    % right
    for trial_idx= 1:num_train/2
        fb7_RF_Z(region_idx, : ,: ) = squeeze(fb7_csp_filter(region_idx,:,:)) * squeeze(fb7_RF_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb7_RF_max_val_z(region_idx, trial_idx) = log10( var(fb7_RF_Z(region_idx,1,:)));%/ (var(RF_Z(region_idx,1,:)) + var(RF_Z(region_idx,2,:))));
        fb7_RF_min_val_z(region_idx, trial_idx) = log10( var(fb7_RF_Z(region_idx,2,:)));%/ (var(RF_Z(region_idx,1,:)) + var(RF_Z(region_idx,2,:))));
    end
    RF_fb7_csp_feature(region_idx, 1, : ) = fb7_RF_max_val_z(region_idx, : ,: );
    RF_fb7_csp_feature(region_idx, 2, : ) = fb7_RF_min_val_z(region_idx, : ,: );
    
    % test
    for trial_idx =1:num_test
        fb7_TE_Z(region_idx, :, : ) = squeeze(fb7_csp_filter(region_idx,:,:)) * squeeze(fb7_test_cross_region(region_idx, trial_idx, :, :))' ;
        
        fb7_TE_max_val_z(region_idx, trial_idx) = log10( var(fb7_TE_Z(region_idx,1,:)));%/ (var(TE_Z(region_idx,1,:)) + var(TE_Z(region_idx,2,:))));
        fb7_TE_min_val_z(region_idx, trial_idx) = log10( var(fb7_TE_Z(region_idx,2,:)));%/ (var(TE_Z(region_idx,1,:)) + var(TE_Z(region_idx,2,:))));
    end
    TE_fb7_csp_feature(region_idx, 1, : ) = fb7_TE_max_val_z(region_idx, : ,: );
    TE_fb7_csp_feature(region_idx, 2, : ) = fb7_TE_min_val_z(region_idx, : ,: );
            
end

% left right summation
for region_idx = 1: set_len_region
    fb7_train(region_idx,:,:) = [ squeeze(RH_fb7_csp_feature(region_idx,:,:)),squeeze(RF_fb7_csp_feature(region_idx,:,:)) ]';
end
%
last_fb7_test = TE_fb7_csp_feature;

%% select FBCSP features  by MIBIF( mutual information based individual feature algorithm
% region p의 csp feature 는 -> v^(p)_%0,1 ... v^(p)_q,2  where q : # of filter

% region 별로 7개의 filtercsp를 모으기
% fbx_train = [14,224,2 ] = [regions, trials, csp_features]
for region_idx = 1 : set_len_region
    
    region_csp_feature(region_idx, : ,: ) = [squeeze(fb1_train(region_idx, :, :)) , squeeze(fb2_train(region_idx, :, :)), squeeze(fb3_train(region_idx, :, :)), ...
        squeeze(fb4_train(region_idx, :, :)), squeeze(fb5_train(region_idx, :, :)), squeeze(fb6_train(region_idx, :, :)), squeeze(fb7_train(region_idx, :, :)) ];
    
    region_csp_test_feature(region_idx, : ,:) = [squeeze(last_fb1_test(region_idx, :, :)) ; squeeze(last_fb2_test(region_idx, :, :)); squeeze(last_fb3_test(region_idx, :, :)); ...
        squeeze(last_fb4_test(region_idx, :, :)); squeeze(last_fb5_test(region_idx, :, :)); squeeze(last_fb6_test(region_idx, :, :)); squeeze(last_fb7_test(region_idx, :, :)) ];
end
