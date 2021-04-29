function csp_filter = jw_csp(input_data)

% find num of trials
num_trials = length( input_data(:,1,1));
num_ch = length( input_data(1,:,1));

% seperate class
RH_data = input_data(1:num_trials/2,:,:);
RF_data = input_data((num_trials/2)+1:num_trials,:,:);

% get normalized covariance matrix
% (X'X) / tr(X'X)
for trial_idx = 1:(num_trials/2)
    RH_cov(trial_idx, :, : )= ( squeeze(RH_data(trial_idx,:,:)) * squeeze(RH_data(trial_idx,:,:))' ) / trace( squeeze(RH_data(trial_idx,:,:)) * squeeze(RH_data(trial_idx,:,:))'  );
    RF_cov(trial_idx, :, : )= ( squeeze(RF_data(trial_idx,:,:)) * squeeze(RF_data(trial_idx,:,:))' ) / trace( squeeze(RF_data(trial_idx,:,:)) * squeeze(RF_data(trial_idx,:,:))'  );
end

mean_RH = mean(RH_cov, 1);
mean_RF = mean(RF_cov, 1);
cov_sum = mean_RH + mean_RF;
disp("hi")
disp(RH_cov.shape)

% eigen value decomposition
% [V,D] = eig(a)  고유값으로 이루어진 대각행렬D(고유값,lamda), 각 열이 이에 대항하는 우고유벡터인 행렬 V(고유vec)
% 백색화 변환행렬 : Q = sqrt(lamda^(-1))*U'
% -> 계산한 각 클래스 평균 공분산 행렬이 동일한 고유벡터를 갖도록 함 
[eigen_vec, eigen_val] = eig(squeeze(cov_sum));
P = sqrt(inv(eigen_val)) * eigen_vec';

s_RH = P * squeeze(mean_RH) * P';
s_RF = P * squeeze(mean_RF) * P';

% s_RH, S_RF -> re_eigen value decomposition
% RF_eigen_val + RH_eigen_val = I  - (1)
% eigenvalue 값에 따라 한 의도의 분산최대가 되면 다른쪽은 최소가됨 by (1)
[RH_eigen_vec, RH_eigen_val] = eig(s_RH);
[RF_eigen_vec, RF_eigen_val] = eig(s_RF);


nw = P' * RH_eigen_vec;
BB = nw' * squeeze(mean_RH) * nw;
BBB = nw' * squeeze(mean_RF) * nw;
disp(size(BB))
[amp1, loc1] = max(diag(BB));
[amp2, loc2] = max(diag(BBB));

wn = nw';

csp_filter(:,:) = [wn(loc1,:);wn(loc2,:)];

end