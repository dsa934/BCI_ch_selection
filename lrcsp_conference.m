clear all;% clc
addpath(genpath("C:/Users/JINHYO/Desktop/BCIreboot/ml/TangentSpace/lib"))

%% Configuration
sidx = 1;
score_dict = containers.Map;
sbj_acc = zeros(5,1);
for subject = ["al" "aa" "av" "aw" "ay"]
    N = 18;              % local region : 18 or 118
    fs = 100;            % sample rate : 100 or 1000
    band_range = [9 30]; % according to LRCSP
    HighGamma = false; % true, false
    normalization = "trace"; % trace, N-1
    mean_method = "Euclidean"; % Riemannian, Euclidean
    classifier = "SVM"; % LDA, SVM, KNN, NB,
    maxiter = 50;
    eps = 1e-10;
    % Print
    for Q = 1
        if sidx == 1
            fprintf("Experiment Settings:\n")
            fprintf(" - channel       : %d\n", N)
            fprintf(" - sampling rate : %dHz\n", fs)
            fprintf(" - band range    : %d~%dHz\n", band_range)
            fprintf(" - high gamma    : %d\n", HighGamma)
            fprintf(" - normalization : %s\n", normalization)
            fprintf(" - mean method   : %s\n", mean_method)
            fprintf(" - classifier    : %s\n", classifier) 
        end
    end
    %% Preparation
    % Lists for local regions 18 or 118
    for Q = 1
        if N == 18
            %18 channel (homunculus theory)
            chs=[50,% 1
                43, % 2
                44, % 3 
                52, % 4
                53, % 5 
                60, % 6
                61, % 7
                89, % 8
                54, % 9
                91, % 10
                55, % 11
                47, % 12 
                48, % 13 
                56, % 14 
                58, % 15 
                64, % 16 
                65, % 17
                93 % 18
                ]; 
            % 18 local region 
            lrs = {
                [1 2 4 6] % 1
                [2 1 4 3] % 2 
                [2 5 4 9 3] % 3
                [1 2 3 5 6 7 4] % 4
                [5 3 4 7 9] % 5 
                [6 1 4 7 8] % 6
                [7 4 5 6 8 9 10] % 7
                [8 6 7 10] % 8
                [9 5 11 10] % 9 
                [10 9 7 16 8 18] % 10
                [11 9 12 14 16] % 11
                [11 9 13 14 12] % 12
                [15 12 13 14] % 13 
                [11 12 13 14 15 16 17] % 14
                [15 13 14 17] % 15
                [16 14 17 18 11 9 10] % 16
                [14 15 17 18 16] % 17
                [18 10 17 16] % 18
                };
        end
        if N == 118
            %118 channels
            chs = 1:N;
            % 118 local region 
            lrs = {};
            lrs{1, 1}=[1,2,6,7,10];
            lrs{2, 1}=[1,2,3,7,11];
            lrs{3, 1}=[2,3,4,11,12];
            lrs{4, 1}=[3,4,5,8,12];
            lrs{5, 1}=[4,5,8,9,13];
            lrs{6, 1}=[1,6,7,10,14,15];
            lrs{7, 1}=[1,2,6,7,10,11,15,16,17];
            lrs{8, 1}=[4,5,9,8,12,13,19,20,21];
            lrs{9, 1}=[5,8,9,13,21,22];
            lrs{10, 1}=[6,7,10,14,15,16];
            lrs{11, 1}=[7,11,16,17,18];
            lrs{12, 1}=[8,12,18,19,20];
            lrs{13, 1}=[8,9,13,20,21,22];
            lrs{14, 1}=[6,10,14,15,23,31,32];
            lrs{15, 1}=[6,7,10,14,15,16,23,24,33];
            lrs{16, 1}=[7,10,11,15,16,17,24,25,34];
            lrs{17, 1}=[7,11,16,17,18,25,26,35];
            lrs{18, 1}=[11,12,17,18,19,26,27,36];
            lrs{19, 1}=[8,12,18,19,20,27,28,37];
            lrs{20, 1}=[8,12,13,19,20,21,28,29,38];
            lrs{21, 1}=[8,9,13,20,21,22,29,30,39];
            lrs{22, 1}=[9,13,21,22,30,40,41];
            lrs{23, 1}=[14,15,23,24,32,33,31];
            lrs{24, 1}=[15,16,23,24,25,33,34,43];
            lrs{25, 1}=[16,17,24,25,26,34,35,44];
            lrs{26, 1}=[17,18,25,26,27,35,36,45];
            lrs{27, 1}=[18,19,26,27,28,36,37,46];
            lrs{28, 1}=[19,20,27,28,29,37,38,47];
            lrs{29, 1}=[20,21,28,29,30,38,39,48];
            lrs{30, 1}=[21,22,29,30,39,40,41];
            lrs{31, 1}=[14,31,32,50];
            lrs{32, 1}=[14,23,31,32,33,42,50];
            lrs{33, 1}=[15,23,24,32,33,34,42,43,51];
            lrs{34, 1}=[16,24,25,33,34,35,43,44,52];
            lrs{35, 1}=[15,23,24,32,33,34,42,43,51]+2;
            lrs{36, 1}=[15,23,24,32,33,34,42,43,51]+3;
            lrs{37, 1}=[15,23,24,32,33,34,42,43,51]+4;
            lrs{38, 1}=[15,23,24,32,33,34,42,43,51]+5;
            lrs{39, 1}=[15,23,24,32,33,34,42,43,51]+6;
            lrs{40, 1}=[22,30,39,49,58,40,41];
            lrs{41, 1}=[22,40,41,58];
            lrs{42, 1}=[23,32,33,42,43,50,51,59];
            lrs{43, 1}=[24,33,34,42,43,44,51,52,60];
            lrs{44, 1}=[24,33,34,42,43,44,51,52,60]+1;
            lrs{45, 1}=[24,33,34,42,43,44,51,52,60]+2;
            lrs{46, 1}=[24,33,34,42,43,44,51,52,60]+3;
            lrs{47, 1}=[24,33,34,42,43,44,51,52,60]+4;
            lrs{48, 1}=[24,33,34,42,43,44,51,52,60]+5;
            lrs{49, 1}=[30,39,40,48,49,57,58,66];
            lrs{50, 1}=[31,32,42,50,51,59,67,68];
            lrs{51, 1}=[33,42,43,50,51,52,59,60,69];
            lrs{52, 1}=[33,42,43,50,51,52,59,60,69]+1;
            lrs{53, 1}=[33,42,43,50,51,52,59,60,69]+2;
            lrs{54, 1}=[33,42,43,50,51,52,59,60,69]+3;
            lrs{55, 1}=[33,42,43,50,51,52,59,60,69]+4;
            lrs{56, 1}=[33,42,43,50,51,52,59,60,69]+5;
            lrs{57, 1}=[33,42,43,50,51,52,59,60,69]+6;
            lrs{58, 1}=[40,41,49,57,58,66,76,77];
            lrs{59, 1}=[42,50,51,59,60,68,69,78];
            lrs{60, 1}=[43,51,52,59,60,61,69,70,79];
            lrs{61, 1}=[43,51,52,59,60,61,69,70,79]+1;
            lrs{62, 1}=[43,51,52,59,60,61,69,70,79]+2;
            lrs{63, 1}=[43,51,52,59,60,61,69,70,79]+3;
            lrs{64, 1}=[43,51,52,59,60,61,69,70,79]+4;
            lrs{65, 1}=[43,51,52,59,60,61,69,70,79]+5;
            lrs{66, 1}=[49,57,58,65,66,75,76,85];
            lrs{67, 1}=[50,67,68,86,87];
            lrs{68, 1}=[50,59,67,68,69,78,86,87];
            lrs{69, 1}=[51,59,60,68,69,70,78,79,88];
            lrs{70, 1}=[51,59,60,68,69,70,78,79,88]+1;
            lrs{71, 1}=[51,59,60,68,69,70,78,79,88]+2;
            lrs{72, 1}=[51,59,60,68,69,70,78,79,88]+3;
            lrs{73, 1}=[51,59,60,68,69,70,78,79,88]+4;
            lrs{74, 1}=[51,59,60,68,69,70,78,79,88]+5;
            lrs{75, 1}=[51,59,60,68,69,70,78,79,88]+6;
            lrs{76, 1}=[58,66,75,76,77,85,95,96];
            lrs{77, 1}=[58,76,77,95,96];
            lrs{78, 1}=[59,68,69,78,79,87,88,97];
            lrs{79, 1}=[60,69,70,78,79,80,88,89,98];
            lrs{80, 1}=[61,70,71,79,80,81,89,90];
            lrs{81, 1}=[62,71,72,80,81,82,90,91];
            lrs{82, 1}=[63,72,73,81,82,83,91,92];
            lrs{83, 1}=[63,72,73,81,82,83,91,92]+1;
            lrs{84, 1}=[63,72,73,81,82,83,91,92]+2;
            lrs{85, 1}=[66,75,76,84,85,94,95,102];
            lrs{86, 1}=[67,68,86,87,97,103];
            lrs{87, 1}=[67,68,78,86,87,88,97,98];
            lrs{88, 1}=[69,78,79,87,88,89,97,98,103];
            lrs{89, 1}=[70,79,80,88,89,90,98,99,104];
            lrs{90, 1}=[71,80,81,89,90,91,104,99];
            lrs{91, 1}=[72,81,82,90,91,92,99,100,106];
            lrs{92, 1}=[73,82,83,91,92,93,100,108];
            lrs{93, 1}=[74,83,84,92,93,94,100,108,101];
            lrs{94, 1}=[78,84,85,93,94,95,101,102,108];
            lrs{95, 1}=[76,77,85,94,95,96,101,102];
            lrs{96, 1}=[76,77,95,96,102,109];
            lrs{97, 1}=[86,87,88,97,98,103];
            lrs{98, 1}=[87,88,89,97,98,103,104,112];
            lrs{99, 1}=[89,90,91,99,104,105,106];
            lrs{100, 1}=[91,92,93,100,106,107,108];
            lrs{101, 1}=[93,94,95,101,102,108,109,114];
            lrs{102, 1}=[94,95,96,101,102,109];
            lrs{103, 1}=[97,98,103,104,112];
            lrs{104, 1}=[89,98,99,103,104,105,110,112];
            lrs{105, 1}=[99,104,105,106,110];
            lrs{106, 1}=[91,99,100,105,106,107,110,111];
            lrs{107, 1}=[100,106,107,108,111];
            lrs{108, 1}=[93,100,101,107,108,109,111,114];
            lrs{109, 1}=[101,102,108,109,114];
            lrs{110, 1}=[104,105,106,110,112,113,115];
            lrs{111, 1}=[106,107,108,111,114,113,116];
            lrs{112, 1}=[98,103,104,110,112,115,117];
            lrs{113, 1}=[106,110,111,113,115,116];
            lrs{114, 1}=[101,108,111,114,109,116,118];
            lrs{115, 1}=[110,112,113,115,117];
            lrs{116, 1}=[111,113,116,114,118];
            lrs{117, 1}=[112,115,113,117];
            lrs{118, 1}=[113,116,114,118];
        end
    end
      
    % Load Data
    for Q = 1
        data_path = sprintf("C:/Users/JINHYO/Desktop/BCIreboot/raw/%dHz/data_set_IVa_%s.mat", fs, subject);
        label_path = sprintf("C:/Users/JINHYO/Desktop/BCIreboot/raw/%dHz/true_labels_%s.mat", fs, subject);

        load(data_path);
        load(label_path);
    end

    % Initialize Variables
    for Q = 1
        raw = 0.1*double(cnt);    % (time, electrodes)
        raw = raw';               % (electrodes, time)
        raw = raw(chs, :);        % (selected electrodes, time)
        cue = transpose(mrk.pos); % (280, 1)
        K = size(cue, 1);         % number of cue(trial) = 280
        T = 2.5 * fs;             % length of interesting signal (visual cue 3.5초) -> cue+0.5 ~ cue+3초 (앞뒤 0.5초 제외)
        true_y = true_y';         % (280, 1), 1: right, 2: foot  
        competition_y = mrk.y';    % (280, 1), 1: right, 2: foot, nan: test  
    end
    
    % Split Trials
    eeg = zeros(N, T, K); % (electrodes, samples, trials)
    for k = 1:K
       start  = cue(k) + 0.5*fs;
       finish = cue(k) + 3.0*fs - 1;
       temp = raw(:, start:finish); % (electrodes, samples)
       eeg(:, :, k) = temp; % (electrodes, samples, trials)
    end


    %% Preprocessing
    % Bandpass Filter
    for Q = 1
        [band_b, band_a] = butter(4, [band_range(1)/(fs/2), band_range(2)/(fs/2)]);

        bp_eeg = zeros(N, T, K);
        for k = 1:K
            for n = 1:N
                temp = eeg(n, :, k); % (1, samples)
                temp = squeeze(temp); % (samples)
                bp_temp = filtfilt(band_b, band_a, temp);
                bp_eeg(n, :, k) = bp_temp;
            end
        end
        
        % High gamma
        if HighGamma
            [h_band_b, h_band_a] = butter(4, [140/(fs/2), 160/(fs/2)]);

            h_bp_eeg = zeros(N, T, K);
            for k = 1:K
                for n = 1:N
                    temp = eeg(n, :, k); % (1, samples)
                    temp = squeeze(temp); % (samples)
                    h_bp_temp = filtfilt(h_band_b, h_band_a, temp);
                    h_bp_eeg(n, :, k) = h_bp_temp;
                end
            end
            bp_eeg = bp_eeg + h_bp_eeg;
        end
        
    end

    % Make Covaiance Matrix
    for Q = 1
        Xspd = zeros(N, N, K);
        for k = 1:K
            temp = bp_eeg(:, :, k);        % (N, T)
            if normalization == "trace"
                Xspd(:,:,k) = temp * temp' / trace(temp*temp');
            elseif normalization == "N-1"
                Xspd(:,:,k) = cov(temp');
            end
        end
    end

    % Sanity Check : Symmetric Positive Definite
    sanity = zeros(K, 1);
    for Q = 1
        for k = 1:K
            spd = Xspd(:,:,k);
              % symmetric
            is_sym = issymmetric(spd);
              % positive definite
            is_pos = all(eig(spd) > eps);
            if is_sym & is_pos
                sanity(k,1) = 1;
            else
                sanity(k,1) = -1;
                warning("Custom:Sanity_check",...
                    "Positive Definite가 아닌 행렬이 발견되었습니다. 데이터셋에서 제외합니다. (%s %dth trial)", subject, k)
            end
        end

          % screen out insanity
        sanity_mask = sanity > 0;
        Xspd = Xspd(:,:,sanity_mask);
        true_y = true_y(sanity_mask, 1);
        competition_y = competition_y(sanity_mask,1);
        K = sum(sanity_mask);
    end

    %% Split
    % Train Test Split
    for Q = 1
       % competition test setting
       mask = zeros(size(true_y));
       Xspd_tr = [];
       y_tr = [];
       Xspd_te = [];
       y_te =[];
       ktr = 1;
       kte = 1;
       for k = 1:K
           if isnan(competition_y(k))
               % test
               mask(k, 1) = true;
               Xspd_te(:, :, kte) = Xspd(:, :, k);
               y_te(kte, 1) = true_y(k, 1);
               kte = kte + 1;
           else
               % train
               mask(k, 1) = false;
               Xspd_tr(:, :, ktr) = Xspd(:, :, k);
               y_tr(ktr, 1) = true_y(k, 1);
               ktr = ktr + 1;
           end

          % Split Training Class
            right_mask = (y_tr == 1);
            foot_mask = (y_tr == 2);
            right_spd = Xspd_tr(:, :, right_mask);
            foot_spd = Xspd_tr(:, :, foot_mask);

            K_tr = size(Xspd_tr, 3);
            K_te = size(Xspd_te, 3);
        end
    end
    
    %% Local Region Scoring    
    % Find Riemann each Class Means for Local Regions
    for Q = 1
        LR_classMeans = cell(N, 2);
        for n = 1:N
            lr_idx = lrs{n, 1};
            lr_right_spds = Xspd_tr(lr_idx, lr_idx, right_mask);
            lr_foot_spds = Xspd_tr(lr_idx, lr_idx, foot_mask);

            if mean_method == "Riemannian"
                [lr_right_mean, iter, ssrd] = riemann_mean(lr_right_spds, eps, maxiter);
                [lr_foot_mean, iter, ssrd] = riemann_mean(lr_foot_spds, eps, maxiter);
            elseif mean_method == "Euclidean"
                lr_right_mean = euclide_mean(lr_right_spds);
                lr_foot_mean = euclide_mean(lr_foot_spds);
            end
                
            LR_classMeans{n, 1} = lr_right_mean;
            LR_classMeans{n, 2} = lr_foot_mean;
        end
    end
    
    % Eigenvalue Disparity score
    for Q = 1
        eigenvalue_disparity_score = zeros(N,1);
        for n = 1:N
            % Class Mean Covariance matrix
            M1 = LR_classMeans{n,1};
            M2 = LR_classMeans{n,2};
            [U, Sigma] = eig(M1+M2);
            H = U * inv(sqrtm(Sigma)); % whitening
            [psi, D1] = eig(H'*M1*H);
            d1 = diag(D1);
                
            eigenvalue_disparity_score(n,1) = abs(max(d1) - min(d1));
        end
        score_dict(subject) = eigenvalue_disparity_score;
        [sort_val, sort_order] = sort(eigenvalue_disparity_score);
    end
    
    %% Local Region Feature extraction     
    % Feature Extraction : Riemmanian distance matrix -> vectorize
    for Q = 1
        LR_feat_tr = cell(N, 1);
        LR_feat_te = cell(N, 1);
        for n = 1:N
            % local region info
            lr_idx = lrs{n, 1};
            lr_SPD_tr = Xspd_tr(lr_idx, lr_idx, :);
            lr_SPD_te = Xspd_te(lr_idx, lr_idx, :);
            
            % Class Mean Covariance matrix
            M1 = LR_classMeans{n,1};
            M2 = LR_classMeans{n,2};
            
            % CSP
            [U, Sigma] = eig(M1+M2);
            H = U * inv(sqrtm(Sigma)); % whitening
            [psi, D1] = eig(H'*M1*H);
            W = (psi'*H')';
            d1 = diag(D1);
            [s_val, s_idx] = sort(d1);
            w_min = W(:,s_idx(1));
            w_max = W(:,s_idx(end));
            
              % apply filter
            % training
            lr_feat_tr = zeros(K_tr, 2);
            lr_feat_te = zeros(K_te, 2);
            for k = 1:K_tr
                lr_feat_tr(k, 1) = log(w_min' * lr_SPD_tr(:,:,k) * w_min);
                lr_feat_tr(k, 2) = log(w_max' * lr_SPD_tr(:,:,k) * w_max);
            end
            % test
            for k = 1:K_te
                lr_feat_te(k, 1) = log(w_min' * lr_SPD_te(:,:,k) * w_min);
                lr_feat_te(k, 2) = log(w_max' * lr_SPD_te(:,:,k) * w_max);
            end

            % save local CSP features
            LR_feat_tr{n,1} = lr_feat_tr;
            LR_feat_te{n,1} = lr_feat_te;
        end 
    end
    
    %% Classification
    for Q = 1
        reducing_acc = zeros(N,1);
        for i = 1:N
            selected_lrs = sort_order(i:N, 1);
            %% Train
            feat_tr = [];
            feat_te = [];
            for lr = selected_lrs'
               feat_tr = [feat_tr, LR_feat_tr{lr,1}];
               feat_te = [feat_te, LR_feat_te{lr,1}];
            end

            [Model, trpred, pred] = train_classifier(feat_tr, y_tr, feat_te, classifier);
            corrects = pred == y_te;
            acc = sum(corrects) / K_te;
            acc = round(acc*100, 2);
            reducing_acc(i,1) = acc;
        end
        [max_v, max_i] = max(flip(reducing_acc)); % max_i = 선택된 region 개수(동점의 경우 사용된 region이 적은 것으로 출력)
        fprintf("(%s) max acc : %.2f (%d)\n", subject, max_v, max_i)
        subject_reducing_acc{sidx, 1} = reducing_acc;
    end
    
    % Next subject
    sidx = sidx + 1;
end

% Figure
temp = 0;
for s = 1:5
    temp = temp + max(subject_reducing_acc{s,1});
end
fprintf("mean of best cases : %.2f\n\n", temp/5)

%% plot
subjects = ["al" "aa" "av" "aw" "ay"];
figure
hold on 
for s = 1:5
    plot(0:N-1, subject_reducing_acc{s,1})
    %title(sprintf("Excluding experiment %s (order : %s)", classifier, strrep(reducing_order, "_", " ")))
    ylim([40 100])
    ylabel("Test Accuracy")
    xlabel("Number of Excluded Local Regions")
end
legend(subjects, "Location", "Northwest")
% saveas(gcf, sprintf("Excluding experiment %dch %s %s %s %s.jpg", N, feature_space, feature_mode, classifier, reducing_order))
