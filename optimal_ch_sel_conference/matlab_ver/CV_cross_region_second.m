
% cross_region_revised_me 에서 넘어온 train/test data
% region_csp_feature = [14, 224, 14] 
% region_csp_test_feature = [14,14,56] -> test label 뒤집어야함

%% mutual information
% I(f ; w) = H(w) -H(w|f)
num_fb = length(region_csp_feature(1,1,:));
num_region = length(region_csp_feature(:,1,1));

% MI 계산시 반드시 ( M x1  , Nx1 )형태로 사용 
for region_idx = 1: num_region
    
   for fb_idx = 1: num_fb
        mi_fb(region_idx, fb_idx, :) = mi(region_csp_feature(region_idx, :, fb_idx)',train_label');
   end 
   [val(region_idx, :), idx(region_idx,:) ] = sort(mi_fb(region_idx,:), 'descend');
end

% selected two discriminative filter

for region_idx = 1:num_region
    
    temp_idx(region_idx,:) = idx(region_idx,1:2);
end


% selected high Mi score features
for region_idx = 1:num_region
          
    selected_region_csp_feature(region_idx, :, 1) = region_csp_feature(region_idx, : , temp_idx(region_idx,1) );  
    selected_region_csp_feature(region_idx, :, 2) = region_csp_feature(region_idx, : , temp_idx(region_idx,2) );  
    selected_region_csp_feature(region_idx, :, 3) = region_csp_feature(region_idx, : , temp_idx(region_idx,1) );  
    selected_region_csp_feature(region_idx, :, 4) = region_csp_feature(region_idx, : , temp_idx(region_idx,2) );  
    
        
    selected_region_test_csp_feature(region_idx, 1,:) = region_csp_test_feature(region_idx, temp_idx(region_idx,1) , : );
    selected_region_test_csp_feature(region_idx, 2,:) = region_csp_test_feature(region_idx, temp_idx(region_idx,2) , : );
    selected_region_test_csp_feature(region_idx, 3,:) = region_csp_test_feature(region_idx, temp_idx(region_idx,1) , : );
    selected_region_test_csp_feature(region_idx, 4,:) = region_csp_test_feature(region_idx, temp_idx(region_idx,2) , : );
  
end



%% training
for region_idx = 1 : num_region
    shuffle_train_idx = randperm(252);
    shuffle_csp_train_feature =  selected_region_csp_feature(:,shuffle_train_idx,:);
    shuffle_csp_train_label =  train_label(:,shuffle_train_idx);
     
    
    SVMStruct = fitcsvm(squeeze(shuffle_csp_train_feature(region_idx,:,:)), shuffle_csp_train_label');    
    
    % shuffle tets
    shuffle_idx = randperm(28);
    shuffle_csp_test_feature =  selected_region_test_csp_feature(:,:,shuffle_idx);
    shuffle_csp_test_label =  test_label(:,shuffle_idx);
    
    result = predict(SVMStruct,squeeze(shuffle_csp_test_feature(region_idx,:,:))');
    
    correct = 0;
    
    for k = 1:length(result)
        
        if result(k) ==shuffle_csp_test_label(k)
            correct=correct+1;
        end
    end
    accy(region_idx) = correct/length(result)*100;
end

disp(max(accy));
