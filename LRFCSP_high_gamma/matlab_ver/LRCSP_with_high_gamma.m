%{

"Motor Imagery Classification using local region CSP features 

 with high-gamma band"

by jinwoo Lee

%} 

%% load data
clear all
load('data_1000Hz\data_set_IVa_al');
load('data_1000Hz\true_labels_al');

%% init ( ref BCI Competition III-IVa )
cnt= 0.1*double(cnt);
cue=mrk.pos;
yy=mrk.y;
cue=transpose(cue);
temp=[];
cnt=cnt;
data_training=224;

for k=1:data_training
    temp=cnt(cue(k):cue(k)+5000,:);
    temp=temp';
    eeg(:,:,k)=temp;
    temp=0;
end
u=1;
for k=(data_training+1):280
    temp=cnt(cue(k):cue(k)+5000,:);
    temp=temp';
    test_eeg(:,:,u)=temp;
    u=u+1;
    temp=0;
end

% seperate class (right hand / foot)
rh=1;
rf=1;
for k=1:data_training
    if mrk.y(k)==1
        ll(rh)=k;
        rh=rh+1;
    else
        rr(rf)=k;
        rf=rf+1;
    end
end
l=length(ll);
r=length(rr);
free=min(l,r)-1;
tr=280-data_training;

for k=1:l
    rh_eeg(:,:,k)=eeg(:,:,ll(k));
end
for k=1:r
    rf_eeg(:,:,k)=eeg(:,:,rr(k));
end

% 18 channels in motor cortex regions
chn=[50,43,44,52,53,60,61,89,54,91,55,47,48,56,58,64,65,93]; 

for k=1:18
    sel_rh_eeg(k,:,:)=rh_eeg(chn(k),:,:);
    sel_rf_eeg(k,:,:)=rf_eeg(chn(k),:,:);
    sel_test_eeg(k,:,:)=test_eeg(chn(k),:,:);
end


%% bandpass filtering with mu, beta and high-gamma bands
[bbb,aaa]=butter(4,[9/500 30/500]);
[ddd,ccc]=butter(4,[140/500 160/500]);
vf=[];bp_rh=[];bp_rf=[];bp_test=[];mu_beta_rh=[];mu_beta_rf=[];mu_beta_test=[];result=[];fe1=[];fe2=[];fe3=[];result1=[];

for node=1:18
    for k=1:l
        mu_beta_rh(k,:)=filtfilt(bbb,aaa,sel_rh_eeg(node,:,k));
        high_gamma_rh(k,:)=filtfilt(ddd,ccc,sel_rh_eeg(node,:,k));
    end
    bp_rh(:,:,node)=mu_beta_rh(:,501:3000) + high_gamma_rh(:,501:3000);
    
    for k=1:r
        mu_beta_rf(k,:)=filtfilt(bbb,aaa,sel_rf_eeg(node,:,k));
        high_gamma_rf(k,:)=filtfilt(ddd,ccc,sel_rf_eeg(node,:,k));
    end
    bp_rf(:,:,node)=mu_beta_rf(:,501:3000) + high_gamma_rf(:,501:3000);
    
    for k=1:tr
        mu_beta_test(k,:)=filtfilt(bbb,aaa,sel_test_eeg(node,:,k));
        high_gamma_test(k,:)=filtfilt(ddd,ccc,sel_test_eeg(node,:,k));
    end
    bp_test(:,:,node)=mu_beta_test(:,501:3000) + high_gamma_test(:,501:3000);
    
end



%% region 1
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,1);bp_rh(trial,:,2);bp_rh(trial,:,4);bp_rh(trial,:,6)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,1);bp_rf(trial,:,2);bp_rf(trial,:,4);bp_rf(trial,:,6)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,1);bp_test(trial,:,2);bp_test(trial,:,4);bp_test(trial,:,6)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff1]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,1);bp_rh(trial,:,2);bp_rh(trial,:,4);bp_rh(trial,:,6)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_1=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,1);bp_rf(trial,:,2);bp_rf(trial,:,4);bp_rf(trial,:,6)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_1=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,1);bp_test(trial,:,2);bp_test(trial,:,4);bp_test(trial,:,6)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_1=[max_f_t;min_f_t]';
end
%% region 2
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,2);bp_rh(trial,:,1);bp_rh(trial,:,4);bp_rh(trial,:,3)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,2);bp_rf(trial,:,1);bp_rf(trial,:,4);bp_rf(trial,:,3)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,2);bp_test(trial,:,1);bp_test(trial,:,4);bp_test(trial,:,3)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff2]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,2);bp_rh(trial,:,1);bp_rh(trial,:,4);bp_rh(trial,:,3)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_2=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,2);bp_rf(trial,:,1);bp_rf(trial,:,4);bp_rf(trial,:,3)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_2=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,2);bp_test(trial,:,1);bp_test(trial,:,4);bp_test(trial,:,3)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_2=[max_f_t;min_f_t]';
end
%% region 3
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,2);bp_rh(trial,:,5);bp_rh(trial,:,4);bp_rh(trial,:,9);bp_rh(trial,:,3)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,2);bp_rf(trial,:,5);bp_rf(trial,:,4);bp_rf(trial,:,9);bp_rf(trial,:,3)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,2);bp_test(trial,:,5);bp_test(trial,:,4);bp_test(trial,:,9);bp_test(trial,:,3)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff3]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,2);bp_rh(trial,:,5);bp_rh(trial,:,4);bp_rh(trial,:,9);bp_rh(trial,:,3)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_3=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,2);bp_rf(trial,:,5);bp_rf(trial,:,4);bp_rf(trial,:,9);bp_rf(trial,:,3)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_3=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,2);bp_test(trial,:,5);bp_test(trial,:,4);bp_test(trial,:,9);bp_test(trial,:,3)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_3=[max_f_t;min_f_t]';
end
%% region 4
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,1);bp_rh(trial,:,2);bp_rh(trial,:,3);bp_rh(trial,:,5);bp_rh(trial,:,6);bp_rh(trial,:,7);bp_rh(trial,:,4)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,1);bp_rf(trial,:,2);bp_rf(trial,:,3);bp_rf(trial,:,5);bp_rf(trial,:,6);bp_rf(trial,:,7);bp_rf(trial,:,4)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,1);bp_test(trial,:,2);bp_test(trial,:,3);bp_test(trial,:,5);bp_test(trial,:,6);bp_test(trial,:,7);bp_test(trial,:,4)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff4]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,1);bp_rh(trial,:,2);bp_rh(trial,:,3);bp_rh(trial,:,5);bp_rh(trial,:,6);bp_rh(trial,:,7);bp_rh(trial,:,4)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_4=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,1);bp_rf(trial,:,2);bp_rf(trial,:,3);bp_rf(trial,:,5);bp_rf(trial,:,6);bp_rf(trial,:,7);bp_rf(trial,:,4)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_4=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,1);bp_test(trial,:,2);bp_test(trial,:,3);bp_test(trial,:,5);bp_test(trial,:,6);bp_test(trial,:,7);bp_test(trial,:,4)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_4=[max_f_t;min_f_t]';
end
%% region 5
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,5);bp_rh(trial,:,3);bp_rh(trial,:,4);bp_rh(trial,:,7);bp_rh(trial,:,9)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,5);bp_rf(trial,:,3);bp_rf(trial,:,4);bp_rf(trial,:,7);bp_rf(trial,:,9)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,5);bp_test(trial,:,3);bp_test(trial,:,4);bp_test(trial,:,7);bp_test(trial,:,9)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff5]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,5);bp_rh(trial,:,3);bp_rh(trial,:,4);bp_rh(trial,:,7);bp_rh(trial,:,9)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_5=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,5);bp_rf(trial,:,3);bp_rf(trial,:,4);bp_rf(trial,:,7);bp_rf(trial,:,9)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_5=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,5);bp_test(trial,:,3);bp_test(trial,:,4);bp_test(trial,:,7);bp_test(trial,:,9)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_5=[max_f_t;min_f_t]';
end
%% region 6
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,6);bp_rh(trial,:,1);bp_rh(trial,:,4);bp_rh(trial,:,7);bp_rh(trial,:,8)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,6);bp_rf(trial,:,1);bp_rf(trial,:,4);bp_rf(trial,:,7);bp_rf(trial,:,8)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,6);bp_test(trial,:,1);bp_test(trial,:,4);bp_test(trial,:,7);bp_test(trial,:,8)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff6]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,6);bp_rh(trial,:,1);bp_rh(trial,:,4);bp_rh(trial,:,7);bp_rh(trial,:,8)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_6=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,6);bp_rf(trial,:,1);bp_rf(trial,:,4);bp_rf(trial,:,7);bp_rf(trial,:,8)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_6=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,6);bp_test(trial,:,1);bp_test(trial,:,4);bp_test(trial,:,7);bp_test(trial,:,8)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_6=[max_f_t;min_f_t]';
end
%% region 7
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,7);bp_rh(trial,:,4);bp_rh(trial,:,5);bp_rh(trial,:,6);bp_rh(trial,:,8);bp_rh(trial,:,9);bp_rh(trial,:,10)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,7);bp_rf(trial,:,4);bp_rf(trial,:,5);bp_rf(trial,:,6);bp_rf(trial,:,8);bp_rf(trial,:,9);bp_rf(trial,:,10)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,7);bp_test(trial,:,4);bp_test(trial,:,5);bp_test(trial,:,6);bp_test(trial,:,8);bp_test(trial,:,9);bp_test(trial,:,10)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff7]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,7);bp_rh(trial,:,4);bp_rh(trial,:,5);bp_rh(trial,:,6);bp_rh(trial,:,8);bp_rh(trial,:,9);bp_rh(trial,:,10)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_7=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,7);bp_rf(trial,:,4);bp_rf(trial,:,5);bp_rf(trial,:,6);bp_rf(trial,:,8);bp_rf(trial,:,9);bp_rf(trial,:,10)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_7=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,7);bp_test(trial,:,4);bp_test(trial,:,5);bp_test(trial,:,6);bp_test(trial,:,8);bp_test(trial,:,9);bp_test(trial,:,10)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_7=[max_f_t;min_f_t]';
end
%% region 8
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,8);bp_rh(trial,:,6);bp_rh(trial,:,7);bp_rh(trial,:,10)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,8);bp_rf(trial,:,6);bp_rf(trial,:,7);bp_rf(trial,:,10)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,8);bp_test(trial,:,6);bp_test(trial,:,7);bp_test(trial,:,10)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff8]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,8);bp_rh(trial,:,6);bp_rh(trial,:,7);bp_rh(trial,:,10)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_8=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,8);bp_rf(trial,:,6);bp_rf(trial,:,7);bp_rf(trial,:,10)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_8=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,8);bp_test(trial,:,6);bp_test(trial,:,7);bp_test(trial,:,10)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_8=[max_f_t;min_f_t]';
end
%% region 9
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,9);bp_rh(trial,:,5);bp_rh(trial,:,11);bp_rh(trial,:,10)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,9);bp_rf(trial,:,5);bp_rf(trial,:,11);bp_rf(trial,:,10)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,9);bp_test(trial,:,5);bp_test(trial,:,11);bp_test(trial,:,10)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff9]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,9);bp_rh(trial,:,5);bp_rh(trial,:,11);bp_rh(trial,:,10)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_9=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,9);bp_rf(trial,:,5);bp_rf(trial,:,11);bp_rf(trial,:,10)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_9=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,9);bp_test(trial,:,5);bp_test(trial,:,11);bp_test(trial,:,10)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_9=[max_f_t;min_f_t]';
end
%% region 10
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,10);bp_rh(trial,:,9);bp_rh(trial,:,7);bp_rh(trial,:,16);bp_rh(trial,:,8);bp_rh(trial,:,18)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,10);bp_rf(trial,:,9);bp_rf(trial,:,7);bp_rf(trial,:,16);bp_rf(trial,:,8);bp_rf(trial,:,18)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,10);bp_test(trial,:,9);bp_test(trial,:,7);bp_test(trial,:,16);bp_test(trial,:,8);bp_test(trial,:,18)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff10]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,10);bp_rh(trial,:,9);bp_rh(trial,:,7);bp_rh(trial,:,16);bp_rh(trial,:,8);bp_rh(trial,:,18)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_10=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,10);bp_rf(trial,:,9);bp_rf(trial,:,7);bp_rf(trial,:,16);bp_rf(trial,:,8);bp_rf(trial,:,18)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_10=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,10);bp_test(trial,:,9);bp_test(trial,:,7);bp_test(trial,:,16);bp_test(trial,:,8);bp_test(trial,:,18)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_10=[max_f_t;min_f_t]';
end
%% region 11
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,11);bp_rh(trial,:,9);bp_rh(trial,:,12);bp_rh(trial,:,14);bp_rh(trial,:,16)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,11);bp_rf(trial,:,9);bp_rf(trial,:,12);bp_rf(trial,:,14);bp_rf(trial,:,16)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,11);bp_test(trial,:,9);bp_test(trial,:,12);bp_test(trial,:,14);bp_test(trial,:,16)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff11]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,11);bp_rh(trial,:,9);bp_rh(trial,:,12);bp_rh(trial,:,14);bp_rh(trial,:,16)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_11=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,11);bp_rf(trial,:,9);bp_rf(trial,:,12);bp_rf(trial,:,14);bp_rf(trial,:,16)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_11=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,11);bp_test(trial,:,9);bp_test(trial,:,12);bp_test(trial,:,14);bp_test(trial,:,16)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_11=[max_f_t;min_f_t]';
end
%% region 12
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,11);bp_rh(trial,:,9);bp_rh(trial,:,13);bp_rh(trial,:,14);bp_rh(trial,:,12)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,11);bp_rf(trial,:,9);bp_rf(trial,:,13);bp_rf(trial,:,14);bp_rf(trial,:,12)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,11);bp_test(trial,:,9);bp_test(trial,:,13);bp_test(trial,:,14);bp_test(trial,:,12)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff12]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,11);bp_rh(trial,:,9);bp_rh(trial,:,13);bp_rh(trial,:,14);bp_rh(trial,:,12)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_12=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,11);bp_rf(trial,:,9);bp_rf(trial,:,13);bp_rf(trial,:,14);bp_rf(trial,:,12)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_12=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,11);bp_test(trial,:,9);bp_test(trial,:,13);bp_test(trial,:,14);bp_test(trial,:,12)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_12=[max_f_t;min_f_t]';
end
%% region 13
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,15);bp_rh(trial,:,12);bp_rh(trial,:,13);bp_rh(trial,:,14)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,15);bp_rf(trial,:,12);bp_rf(trial,:,13);bp_rf(trial,:,14)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,15);bp_test(trial,:,12);bp_test(trial,:,13);bp_test(trial,:,14)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff13]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,15);bp_rh(trial,:,12);bp_rh(trial,:,13);bp_rh(trial,:,14)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_13=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,15);bp_rf(trial,:,12);bp_rf(trial,:,13);bp_rf(trial,:,14)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_13=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,15);bp_test(trial,:,12);bp_test(trial,:,13);bp_test(trial,:,14)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_13=[max_f_t;min_f_t]';
end
%% region 14
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,11);bp_rh(trial,:,12);bp_rh(trial,:,13);bp_rh(trial,:,14);bp_rh(trial,:,15);bp_rh(trial,:,16);bp_rh(trial,:,17)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,11);bp_rf(trial,:,12);bp_rf(trial,:,13);bp_rf(trial,:,14);bp_rf(trial,:,15);bp_rf(trial,:,16);bp_rf(trial,:,17)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,11);bp_test(trial,:,12);bp_test(trial,:,13);bp_test(trial,:,14);bp_test(trial,:,15);bp_test(trial,:,16);bp_test(trial,:,17)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff14]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,11);bp_rh(trial,:,12);bp_rh(trial,:,13);bp_rh(trial,:,14);bp_rh(trial,:,15);bp_rh(trial,:,16);bp_rh(trial,:,17)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_14=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,11);bp_rf(trial,:,12);bp_rf(trial,:,13);bp_rf(trial,:,14);bp_rf(trial,:,15);bp_rf(trial,:,16);bp_rf(trial,:,17)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_14=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,11);bp_test(trial,:,12);bp_test(trial,:,13);bp_test(trial,:,14);bp_test(trial,:,15);bp_test(trial,:,16);bp_test(trial,:,17)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_14=[max_f_t;min_f_t]';
end
%% regoin 15
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,15);bp_rh(trial,:,13);bp_rh(trial,:,14);bp_rh(trial,:,17)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,15);bp_rf(trial,:,13);bp_rf(trial,:,14);bp_rf(trial,:,17)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,15);bp_test(trial,:,13);bp_test(trial,:,14);bp_test(trial,:,17)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff15]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,15);bp_rh(trial,:,13);bp_rh(trial,:,14);bp_rh(trial,:,17)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_15=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,15);bp_rf(trial,:,13);bp_rf(trial,:,14);bp_rf(trial,:,17)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_15=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,15);bp_test(trial,:,13);bp_test(trial,:,14);bp_test(trial,:,17)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_15=[max_f_t;min_f_t]';
end
%% region 16
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,16);bp_rh(trial,:,14);bp_rh(trial,:,17);bp_rh(trial,:,18);bp_rh(trial,:,11);bp_rh(trial,:,9);bp_rh(trial,:,10)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,16);bp_rf(trial,:,14);bp_rf(trial,:,17);bp_rf(trial,:,18);bp_rf(trial,:,11);bp_rf(trial,:,9);bp_rf(trial,:,10)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,16);bp_test(trial,:,14);bp_test(trial,:,17);bp_test(trial,:,18);bp_test(trial,:,11);bp_test(trial,:,9);bp_test(trial,:,10)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff16]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,16);bp_rh(trial,:,14);bp_rh(trial,:,17);bp_rh(trial,:,18);bp_rh(trial,:,11);bp_rh(trial,:,9);bp_rh(trial,:,10)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_16=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,16);bp_rf(trial,:,14);bp_rf(trial,:,17);bp_rf(trial,:,18);bp_rf(trial,:,11);bp_rf(trial,:,9);bp_rf(trial,:,10)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_16=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,16);bp_test(trial,:,14);bp_test(trial,:,17);bp_test(trial,:,18);bp_test(trial,:,11);bp_test(trial,:,9);bp_test(trial,:,10)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_16=[max_f_t;min_f_t]';
end
%% region 17
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,14);bp_rh(trial,:,15);bp_rh(trial,:,17);bp_rh(trial,:,18);bp_rh(trial,:,16)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,14);bp_rf(trial,:,15);bp_rf(trial,:,17);bp_rf(trial,:,18);bp_rf(trial,:,16)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,14);bp_test(trial,:,15);bp_test(trial,:,17);bp_test(trial,:,18);bp_test(trial,:,16)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff17]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,14);bp_rh(trial,:,15);bp_rh(trial,:,17);bp_rh(trial,:,18);bp_rh(trial,:,16)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_17=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,14);bp_rf(trial,:,15);bp_rf(trial,:,17);bp_rf(trial,:,18);bp_rf(trial,:,16)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_17=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,14);bp_test(trial,:,15);bp_test(trial,:,17);bp_test(trial,:,18);bp_test(trial,:,16)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_17=[max_f_t;min_f_t]';
end
%% region 18
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[bp_rh(trial,:,18);bp_rh(trial,:,10);bp_rh(trial,:,17);bp_rh(trial,:,16)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[bp_rf(trial,:,18);bp_rf(trial,:,10);bp_rf(trial,:,17);bp_rf(trial,:,16)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[bp_test(trial,:,18);bp_test(trial,:,10);bp_test(trial,:,17);bp_test(trial,:,16)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff18]=jw_csp(co1,co2);
    for trial=1:l
        fe1=rt*[bp_rh(trial,:,18);bp_rh(trial,:,10);bp_rh(trial,:,17);bp_rh(trial,:,16)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_18=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[bp_rf(trial,:,18);bp_rf(trial,:,10);bp_rf(trial,:,17);bp_rf(trial,:,16)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_18=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[bp_test(trial,:,18);bp_test(trial,:,10);bp_test(trial,:,17);bp_test(trial,:,16)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_18=[max_f_t;min_f_t]';
end
%% sorting eigenvalue score
dif_t=[diff1,diff2,diff3,diff4,diff5,diff6,diff7,diff8,diff9,diff10,diff11,diff12,diff13,diff14,diff15,diff16,diff17,diff18];
[ko1,ko2]=sort(dif_t);

%% feature
lcsp=[f_l_1,f_l_2,f_l_3,f_l_4,f_l_5,f_l_6,f_l_7,f_l_8,f_l_9,f_l_10,f_l_11,f_l_12,f_l_13,f_l_14,f_l_15,f_l_16,f_l_17,f_l_18];
rcsp=[f_r_1,f_r_2,f_r_3,f_r_4,f_r_5,f_r_6,f_r_7,f_r_8,f_r_9,f_r_10,f_r_11,f_r_12,f_r_13,f_r_14,f_r_15,f_r_16,f_r_17,f_r_18];
tcsp=[f_t_1,f_t_2,f_t_3,f_t_4,f_t_5,f_t_6,f_t_7,f_t_8,f_t_9,f_t_10,f_t_11,f_t_12,f_t_13,f_t_14,f_t_15,f_t_16,f_t_17,f_t_18];

%% compute accy according to eigenvalue score
for yyu=1:length(ko2)
    clcsp=[];crcsp=[];ctcsp=[];
    for yu=yyu:length(ko2)
        teemp1=lcsp(:,2*ko2(yu)-1:2*ko2(yu));
        clcsp=[clcsp,teemp1];
        teemp2=rcsp(:,2*ko2(yu)-1:2*ko2(yu));
        crcsp=[crcsp,teemp2];
        teemp3=tcsp(:,2*ko2(yu)-1:2*ko2(yu));
        ctcsp=[ctcsp,teemp3];
    end
    vf=[clcsp;crcsp];
    lvf=[ones(l,1);ones(r,1)+1];
    options.MaxIter = 100000;
    SVMStruct = fitcsvm(vf,lvf);
    tvl=true_y((data_training+1):end);
    result = predict(SVMStruct,ctcsp);
    correct=0;
    for k=1:length(result)
        if result(k)==tvl(k)
            correct=correct+1;
        end
    end
    accy(yyu)=correct/length(result)*100;
end
disp(max(accy))
