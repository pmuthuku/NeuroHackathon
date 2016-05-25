clear; clc;

data_path = '../Data/InVivo/';

% List of files to be processed
lis = dir([data_path '/spiketimes*.mat']);


for i=1:size(lis,1)
   
    spiketimes = load([data_path '/' lis(i).name]);
    for j = 2:size(spiketimes.M, 1)
        frames = get_unlabeled_windowed_frames(spiketimes.M(j,:));
        
        [~,filenm,~] = fileparts(lis(i).name);
        
        op_filenm = ['testing_data/' filenm '_' num2str(j) '.mat'];
        save(op_filenm, 'frames');
        
    end
   
end