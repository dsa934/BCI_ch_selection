
function [csp_feature, csp_test_feature] = support_csp_filter(RH_input, RF_input, test_input, num_each_support_ch)

% fb1_RH / RF = RH/RF_input
% RH/RF(test)_input = [63, 112(56), 250, 15]
% num_each_support_ch = [63 x 1 ], 그룹 내 유효채널 수 

% set init params
num_RH_trials = length(RH_input(1,:,1,1));
num_RF_trials = length(RF_input(1,:,1,1));
num_test_trials = length(test_input(1,:,1,1)); 
num_groups = length(RH_input(:,1,1,1));

% 그룹별 csp feature계산하기 
% 그룹별로 동시 계산 불가 -> 그룹별 유효채널수가 다름 
for g_idx = 1 : num_groups
    
    % RH/RF_data = [ 112, 250,각 그룹별 유효채널 수 ]
    RH_data = squeeze(RH_input(g_idx, :,:, 1:num_each_support_ch(g_idx)));
    RF_data = squeeze(RF_input(g_idx, :,:, 1:num_each_support_ch(g_idx)));
    TE_data = squeeze(test_input(g_idx, :,:,1:num_each_support_ch(g_idx)));

    
    % each_group_csp_filetr = [ 2, 유효 채널 수 ], m=1 이라 2 features
    each_group_csp_filter = group_csp(RH_data, RF_data, num_RH_trials,num_RF_trials );

    % RH
    for csp_rh_idx = 1: num_RH_trials
        % Z_RH = [ 2 x time sample points] 
        Z_RH = each_group_csp_filter * squeeze(RH_data(csp_rh_idx,:,:))';
        Z_rh_max_feature(csp_rh_idx,:) = log10( var(Z_RH(1,:)) / (var(Z_RH(1,:)) + var(Z_RH(2,:))) );
        Z_rh_min_feature(csp_rh_idx,:) = log10( var(Z_RH(2,:)) / (var(Z_RH(1,:)) + var(Z_RH(2,:))) );

    end
    % RH_feat = [2, trials] 
    RH_feat = [Z_rh_max_feature;Z_rh_min_feature];
 
    
    % RF
    for csp_rf_idx = 1: num_RF_trials
        
        % Z_RH = [ 2 x time sample points] 
        Z_RF = each_group_csp_filter * squeeze(RF_data(csp_rf_idx,:,:))';
        %disp(size(squeeze(RF_data(:,csp_rf_idx,:))))
        Z_rf_max_feature(csp_rf_idx,:) = log10( var(Z_RF(1,:)) / (var(Z_RF(1,:))+var(Z_RF(2,:))));
        Z_rf_min_feature(csp_rf_idx,:) = log10( var(Z_RF(2,:)) / (var(Z_RF(1,:))+var(Z_RF(2,:))) );
        
    end
    % RH_feat = [2, trials] 
    RF_feat = [Z_rf_max_feature;Z_rf_min_feature];

    % group_feature = [ 그룹 수, 2, trials(RH-RF순서) ] 
    group_feature(g_idx, :,:) = [ RH_feat, RF_feat];

    
    % test data
    for csp_test_idx = 1: num_test_trials

        % Z_TE = [ 2 x time sample points] 
        Z_TE = each_group_csp_filter * squeeze(TE_data(csp_test_idx,:,:))';
        %disp(size(squeeze(TE_data(:,csp_test_idx,:))))
        Z_test_max_feature(csp_test_idx,:) = log10( var(Z_TE(1,:)) / (var(Z_TE(1,:))+var(Z_TE(2,:))));
        Z_test_min_feature(csp_test_idx,:) = log10( var(Z_TE(2,:)) / (var(Z_TE(1,:))+var(Z_TE(2,:))) );

    end
    % test_feat = [2, trials] 
    TE_feat = [Z_test_max_feature,Z_test_min_feature];  

    test_group_feature(g_idx,:,:) = TE_feat;
end
    csp_feature = group_feature;
    csp_test_feature = test_group_feature;

end