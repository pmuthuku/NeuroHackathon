clear; clc;
filenames = [65;67;80;81;82;83;84];
for ifile = 1%:length(filenames) % Iterate through the different experiments
Unlab_spiketimes = load(['../Data/InVivo/spiketimes_exp' num2str(filenames(ifile)) '.mat']);
Unlab_spiketimes.M = Unlab_spiketimes.M';
Unlab_frames = [];
% Each column of PV_spiketimes.M is an upstate of neurons firing.
% I'm going to treat this as one frame.
for i = 2:size(Unlab_spiketimes.M, 2)
    i
    if(~isnan(Unlab_spiketimes.M(1,i))) %NaNs denote boundaries
        frames = get_unlabeled_windowed_frames(Unlab_spiketimes.M(:,i));
    end
    
    Unlab_frames = [Unlab_frames; frames];
    
end
end