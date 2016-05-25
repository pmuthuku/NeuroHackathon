clear; clc;

%% PV_data

ip_path = '../Data_Processing/training_data/dev/';

PV_files = dir([ip_path 'PV*.mat']);

PV_train = [];
for i = 1:size(PV_files, 1)
    
   data = load([ip_path PV_files(i).name]);
   PV_train = [PV_train; data.frames];
    
end


ip_path = '../Data_Processing/training_data/test/';

PV_files = dir([ip_path 'PV*.mat']);

PV_test = [];
for i = 1:size(PV_files, 1)
    
    data = load([ip_path PV_files(i).name]);
    PV_test = [PV_test; data.frames];
    
end


%% Pyr_data

ip_path = '../Data_Processing/training_data/dev/';

Pyr_files = dir([ip_path 'Pyr*.mat']);

Pyr_train = [];
for i = 1:size(Pyr_files, 1)
    
   data = load([ip_path Pyr_files(i).name]);
   Pyr_train = [Pyr_train; data.frames];
    
end


ip_path = '../Data_Processing/training_data/test/';

Pyr_files = dir([ip_path 'Pyr*.mat']);

Pyr_test = [];
for i = 1:size(Pyr_files, 1)
    
    data = load([ip_path Pyr_files(i).name]);
    Pyr_test = [Pyr_test; data.frames];
    
end


%% SST_data

ip_path = '../Data_Processing/training_data/dev/';

SST_files = dir([ip_path 'SST*.mat']);

SST_train = [];
for i = 1:size(SST_files, 1)
    
   data = load([ip_path SST_files(i).name]);
   SST_train = [SST_train; data.frames];
    
end


ip_path = '../Data_Processing/training_data/test/';

SST_files = dir([ip_path 'SST*.mat']);

SST_test = [];
for i = 1:size(SST_files, 1)
    
    data = load([ip_path SST_files(i).name]);
    SST_test = [SST_test; data.frames];
    
end

clearvars -except PV_train PV_test Pyr_train Pyr_test SST_train SST_test;


%% Compute the bases and projections

[PV_vectors, PV_values, PV_mean] = pc_evectors(PV_train', 15);
[Pyr_vectors, Pyr_values, Pyr_mean] = pc_evectors(Pyr_train', 15);
[SST_vectors, SST_values, SST_mean] = pc_evectors(SST_train', 15);


% Compute projections for the train set
PV_train_projections = PV_train - repmat(PV_mean', size(PV_train,1), 1);
PV_train_projections = PV_train_projections * PV_vectors;

Pyr_train_projections = Pyr_train - repmat(Pyr_mean', size(Pyr_train,1), 1);
Pyr_train_projections = Pyr_train_projections * Pyr_vectors;

SST_train_projections = SST_train - repmat(SST_mean', size(SST_train,1), 1);
SST_train_projections = SST_train_projections * SST_vectors;


%% Test data!!

% PV first
PV_on_PV = PV_test - repmat(PV_mean', size(PV_test,1), 1);
PV_on_PV = PV_on_PV * PV_vectors;

PV_on_Pyr = PV_test - repmat(Pyr_mean', size(PV_test,1), 1);
PV_on_Pyr = PV_on_Pyr * Pyr_vectors;

PV_on_SST = PV_test - repmat(SST_mean', size(PV_test,1), 1);
PV_on_SST = PV_on_SST * SST_vectors;


PV_dist_PV = pdist2(PV_on_PV, PV_train_projections);
PV_dist_PV = min(PV_dist_PV, [], 2);

PV_dist_Pyr = pdist2(PV_on_Pyr, Pyr_train_projections);
PV_dist_Pyr = min(PV_dist_Pyr, [], 2);

PV_dist_SST = pdist2(PV_on_SST, SST_train_projections);
PV_dist_SST = min(PV_dist_SST, [], 2);

PV_all_dists = [PV_dist_PV PV_dist_Pyr PV_dist_SST];

[~, min_indxs] = min(PV_all_dists, [], 2);

PV_accuracy = sum(min_indxs(:)==1);
PV_accuracy = PV_accuracy/length(min_indxs);



% Pyr next
Pyr_on_PV = Pyr_test - repmat(PV_mean', size(Pyr_test,1), 1);
Pyr_on_PV = Pyr_on_PV * PV_vectors;

Pyr_on_Pyr = Pyr_test - repmat(Pyr_mean', size(Pyr_test,1), 1);
Pyr_on_Pyr = Pyr_on_Pyr * Pyr_vectors;

Pyr_on_SST = Pyr_test - repmat(SST_mean', size(Pyr_test,1), 1);
Pyr_on_SST = Pyr_on_SST * SST_vectors;


Pyr_dist_PV = pdist2(Pyr_on_PV, PV_train_projections);
Pyr_dist_PV = min(Pyr_dist_PV, [], 2);

Pyr_dist_Pyr = pdist2(Pyr_on_Pyr, Pyr_train_projections);
Pyr_dist_Pyr = min(Pyr_dist_Pyr, [], 2);

Pyr_dist_SST = pdist2(Pyr_on_SST, SST_train_projections);
Pyr_dist_SST = min(Pyr_dist_SST, [], 2);

Pyr_all_dists = [Pyr_dist_PV Pyr_dist_Pyr Pyr_dist_SST];
[~, min_indxs] = min(Pyr_all_dists, [], 2);

Pyr_accuracy = sum(min_indxs(:)==2);
Pyr_accuracy = Pyr_accuracy/length(min_indxs);


% SST finally
SST_on_PV = SST_test - repmat(PV_mean', size(SST_test,1), 1);
SST_on_PV = SST_on_PV * PV_vectors;

SST_on_Pyr = SST_test - repmat(Pyr_mean', size(SST_test,1), 1);
SST_on_Pyr = SST_on_Pyr * Pyr_vectors;

SST_on_SST = SST_test - repmat(SST_mean', size(SST_test,1), 1);
SST_on_SST = SST_on_SST * SST_vectors;


SST_dist_PV = pdist2(SST_on_PV, PV_train_projections);
SST_dist_PV = min(SST_dist_PV, [], 2);

SST_dist_Pyr = pdist2(SST_on_Pyr, Pyr_train_projections);
SST_dist_Pyr = min(SST_dist_Pyr, [], 2);

SST_dist_SST = pdist2(SST_on_SST, SST_train_projections);
SST_dist_SST = min(SST_dist_SST, [], 2);

SST_all_dists = [SST_dist_PV SST_dist_Pyr SST_dist_SST];
[~, min_indxs] = min(SST_all_dists, [], 2);

SST_accuracy = sum(min_indxs(:)==3);
SST_accuracy = SST_accuracy/length(min_indxs);

clearvars -except PV_accuracy Pyr_accuracy SST_accuracy;











