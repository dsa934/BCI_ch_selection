
%{

Restoration of papers related to the BCI channels.

"Optimal channel selection using covariance matrix and cross-combining region in EEG-based BCI"

by jinwoo Lee

%} 


%% load data
clear all
load('C:\Users\dsa93\Desktop\compare_paper_other_algorithm\park_optimal_channelsel_conference\data_100Hz\ay\original\data_set_IVa_ay');
load('C:\Users\dsa93\Desktop\compare_paper_other_algorithm\park_optimal_channelsel_conference\data_100Hz\ay\original\true_labels_ay');


%% init params
ttrue_y=true_y-1;
cnt= 0.1*double(cnt);
cue=mrk.pos;
yy=mrk.y;
mrk.y=ttrue_y;
cue=transpose(cue);
temp=[];
numt=280;

% using fixed time windwo 0-3s
for k=1:numt
    temp=cnt(cue(k):cue(k)+500,:);
    temp=temp';
    eeg(:,:,k)=temp;
    temp=0;
end


st=1;
stt=1;

% seperate RH,RF data
right_hand_idx=1;
right_food_idx=1;
for k=1:numt
    if mrk.y(k)==0
        ll(right_hand_idx)=k;
        right_hand_idx=right_hand_idx+1;
    else
        rr(right_food_idx)=k;
        right_food_idx=right_food_idx+1;
    end
end
l=length(ll);
r=length(rr);
for k=1:l
    sort_eeg(k,:,:)=eeg(:,:,ll(k));
end
for k=1:r
    sort_eeg(l+k,:,:)=eeg(:,:,rr(k));
end

% make label data
sort_eeg_label=[ones(l,1);ones(r,1)+1]-1;


% save data
save('C:\Users\dsa93\Desktop\compare_paper_other_algorithm\park_optimal_channelsel_conference\data_100Hz\ay\original\sort_eeg.mat','sort_eeg');
save('C:\Users\dsa93\Desktop\compare_paper_other_algorithm\park_optimal_channelsel_conference\data_100Hz\ay\original\sort_eeg_label.mat','sort_eeg_label');

