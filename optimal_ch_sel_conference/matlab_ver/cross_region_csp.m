function csp_filter = cross_region_csp(input_RH_data, input_RF_data)

% input data shape = [14,112,250,7] = [ara_num, trial, time, ch ]

%find num of trials
left_num_trials = length( input_RH_data(1,:,1,1));
right_num_trials = length( input_RF_data(1,:,1,1));

num_ch = length( input_RH_data(1,1,1,:));
regions = length( input_RH_data(:,1,1,1));

RH_data = input_RH_data;
RF_data = input_RF_data;

for region_idx = 1: regions
    
    % norm data, (X'X) / tr(X'X)  
    % RH_cov = [14, 112, 7, 7]
    for trial_idx = 1:left_num_trials
        RH_cov(region_idx, trial_idx, :, : )= ( squeeze(RH_data(region_idx, trial_idx,:,:))' * squeeze(RH_data(region_idx, trial_idx,:,:)) ) / trace( squeeze(RH_data(region_idx, trial_idx,:,:))' * squeeze(RH_data(region_idx,trial_idx,:,:))  );
    end
    
    for trial_idx = 1:right_num_trials
         RF_cov(region_idx, trial_idx, :, : )= ( squeeze(RF_data(region_idx, trial_idx,:,:))' * squeeze(RF_data(region_idx, trial_idx,:,:)) ) / trace( squeeze(RF_data(region_idx, trial_idx,:,:))' * squeeze(RF_data(region_idx,trial_idx,:,:))  );
    end
    
    % mean_RH, cov_sum = [14, 7,7]
    mean_RH(region_idx,:,:,:) = mean(RH_cov(region_idx,:,:,:), 2);
    mean_RF(region_idx,:,:,:) = mean(RF_cov(region_idx,:,:,:), 2);
    cov_sum(region_idx,:,:) = mean_RH(region_idx,:,:,:) + mean_RF(region_idx,:,:,:);
    
    % eigen value decomposition
    % eigen vec, val = [14,7,7]
    [eigen_vec(region_idx,:,:), eigen_val(region_idx,:,:)] = eig(squeeze(cov_sum(region_idx,:,:)));
    P(region_idx,:,:) = sqrt(inv(squeeze(eigen_val(region_idx,:,:)))) * squeeze(eigen_vec(region_idx,:,:))';

    % s_RH = [14,8,8]
    s_RH(region_idx,:,:) = squeeze(P(region_idx,:,:)) * squeeze(mean_RH(region_idx,:,:)) * squeeze(P(region_idx,:,:))';
    s_RF(region_idx,:,:) = squeeze(P(region_idx,:,:)) * squeeze(mean_RF(region_idx,:,:)) * squeeze(P(region_idx,:,:))';

    % s_RH, S_RF -> re_eigen value decomposition
    % RF_eigen_val + RH_eigen_val = I  - (1)
    % eigenvalue 값에 따라 한 의도의 분산최대가 되면 다른쪽은 최소가됨 by (1)

    [RH_eigen_vec(region_idx,:,:), RH_eigen_val(region_idx,:,:)] = eig(squeeze(s_RH(region_idx,:,:)));
    [RF_eigen_vec(region_idx,:,:), RF_eigen_val(region_idx,:,:)] = eig(squeeze(s_RF(region_idx,:,:)));

    % nw = [14,8,8]
    nw(region_idx,:,:) = squeeze(P(region_idx,:,:))' * squeeze(RH_eigen_vec(region_idx,:,:));
    
    % BB,BBB = [14,8,8]
    BB(region_idx,:,:) = squeeze(nw(region_idx,:,:))' * squeeze(mean_RH(region_idx,:,:)) * squeeze(nw(region_idx,:,:));
    BBB(region_idx,:,:) = squeeze(nw(region_idx,:,:))' * squeeze(mean_RF(region_idx,:,:)) * squeeze(nw(region_idx,:,:));

    % am1 = [14,1]
    [amp1(region_idx,:), loc1(region_idx,:)] = max(diag(squeeze(BB(region_idx,:,:))));
    [amp2(region_idx,:), loc2(region_idx,:)] = max(diag(squeeze(BBB(region_idx,:,:))));

    wn(region_idx,:,:) = squeeze(nw(region_idx,:,:))';

    % csp_filter= [14,2,7]
    % filter outputshpae= [ 2x(1+adjacant channels)]

    csp_filter(region_idx,1,:) = wn(region_idx,loc1(region_idx,:),:);
    csp_filter(region_idx,2,:) = wn(region_idx,loc2(region_idx,:),:);
    
end



end
