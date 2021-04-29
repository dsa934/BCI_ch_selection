
%{

Restoration of papers related to the BCI channels.

"Correlation-based channel selection and regularized feature optimization for MI-based BCI"

by jinwoo Lee

%}

%% data load
clear all
load('data_set_IVa_al')
load('true_labels_al')

%% Initialization (fixed time window)
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
    temp=cnt(cue(k):cue(k)+300,:);
    temp=temp';
    eeg(:,:,k)=temp;
    temp=0;
end


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
% [right hand : right_foot] -> [0,1]
sort_eeg_label=[ones(l,1);ones(r,1)+1]-1;


% save data
save('./data_csp/al/original/sort_eeg.mat','sort_eeg');
save('./data_csp/al/original/sort_eeg_label.mat','sort_eeg_label');