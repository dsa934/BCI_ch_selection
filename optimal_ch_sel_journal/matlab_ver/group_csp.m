
function csp_filter = group_csp(RH_data, RF_data, num_RH, num_RF)

% RH/RF_data = [ 112, 250,각 그룹별 유효채널 수 ]
% get normalized covariance matrix
% (X'X) / tr(X'X)
% RF/RH_cov = [ 112, 유효채널 수, 유효채널 수,]
for trial_idx = 1:num_RH
    RH_cov(trial_idx, :, : )= ( squeeze(RH_data(trial_idx,:,:))' * squeeze(RH_data(trial_idx,:,:)) ) / trace( squeeze(RH_data(trial_idx,:,:))' * squeeze(RH_data(trial_idx,:,:))  );
end

for trial_idx = 1:num_RF
    RF_cov(trial_idx, :, : )= ( squeeze(RF_data(trial_idx,:,:))' * squeeze(RF_data(trial_idx,:,:)) ) / trace( squeeze(RF_data(trial_idx,:,:))' * squeeze(RF_data(trial_idx,:,:))  );
end


mean_RH = mean(RH_cov, 1);
mean_RF = mean(RF_cov, 1);
cov_sum = mean_RH + mean_RF;

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

[amp1, loc1] = max(diag(BB));
[amp2, loc2] = max(diag(BBB));

wn = nw';

csp_filter(:,:) = [wn(loc1,:);wn(loc2,:)];

end