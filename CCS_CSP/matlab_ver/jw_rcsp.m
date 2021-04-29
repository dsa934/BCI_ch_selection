function csp_filter = jw_rcsp(input_data, alpha , beta, Ns)


%set basic params
num_trials = length( input_data(:,1,1));
num_ch = length( input_data(1,:,1));

% seperate class
RH_data = input_data(1:num_trials/2,:,:);
RF_data = input_data((num_trials/2)+1:num_trials,:,:);

% get normalized covariance matrix
% cov = (X'X) / tr(X'X)
% cov^ = cov(X')          -> 원래 다른 subject(aa, ...)의 값들을 이용한 공분산을 만들어야 하는데
% 이 논문에서는 그렇게 하지 않는다고 선언, 대신 pair-wise한 값을 사용
for trial_idx = 1:(num_trials/2)
    RH_cov(trial_idx, :, : )= ( squeeze(RH_data(trial_idx,:,:)) * squeeze(RH_data(trial_idx,:,:))' ) / trace( squeeze(RH_data(trial_idx,:,:)) * squeeze(RH_data(trial_idx,:,:))'  );
    RF_cov(trial_idx, :, : )= ( squeeze(RF_data(trial_idx,:,:)) * squeeze(RF_data(trial_idx,:,:))' ) / trace( squeeze(RF_data(trial_idx,:,:)) * squeeze(RF_data(trial_idx,:,:))'  );
    
    RH_hat_cov(trial_idx, :, :) = cov(squeeze(RH_data(trial_idx,:,:))');
    RF_hat_cov(trial_idx, :, :) = cov(squeeze(RF_data(trial_idx,:,:))');
    
end

mean_RH = mean(RH_cov, 1);
mean_RF = mean(RF_cov, 1);

mean_hat_RH = mean(RH_hat_cov, 1);
mean_hat_RF = mean(RF_hat_cov, 1);

% p^{class}_alpha , class = {RF, RH}
P_RH =   (  ( (1-alpha) * mean_RH)  + (alpha * mean_hat_RH ) ) ./ num_trials;
P_RF =   (  ( (1-alpha) * mean_RF)  + (alpha * mean_hat_RF ) ) ./ num_trials;


% 정규화 평균공분산 행렬
Iden = eye(Ns,Ns);
Q_RH = (1 - beta) * squeeze(P_RH) + (beta/num_trials) * trace(squeeze(P_RH)) * Iden ; 
Q_RF = (1 - beta) * squeeze(P_RF) + (beta/num_trials) * trace(squeeze(P_RF)) * Iden ; 

% 정규화 평균 공분산 행렬 이후의 과정은 일반 CSP와 같음 
% 논문에 alpha, beta값이 정해지지 않아서 최적의 alpha,beta를 실험을 통해 찾거나
% FDA, LDA와 같은 방법으로 비교해서 최적 alpha, beta를 찾을 것으로 예상
avg_cov = Q_RH + Q_RF;
[eigen_vec, eigen_val] = eig(avg_cov);

% 백색화 변환 행렬 (정규화 평균공분산의)
P = sqrt(inv(eigen_val)) * eigen_vec';

s_RH = P * Q_RH * P';
s_RF = P * Q_RF * P';

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