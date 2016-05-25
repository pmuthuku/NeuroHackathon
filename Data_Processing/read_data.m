clear; clc;


%% PV data

PV_spiketimes = load('../Data/InVitro/PV_spiketimes.mat');

PV_frames = [];

% Each column of PV_spiketimes.M is an upstate of neurons firing.
for i = 1:size(PV_spiketimes.M, 2)
    
    if(~isnan(PV_spiketimes.M(1,i))) %NaNs denote boundaries
        frames = get_PV_windowed_frames(PV_spiketimes.M(:,i));
    
        if size(frames, 1) > 0
            filenm = ['training_data/PV_' num2str(PV_spiketimes.M(1,i)) '.mat'];
            save(filenm, 'frames');
        
            PV_frames = [PV_frames; frames];
        end
    end
    
    
end


%% Pyr data

Pyr_spiketimes = load('../Data/InVitro/Pyr_spiketimes.mat');

Pyr_frames = [];

% Each column of Pyr_spiketimes.M is an upstate
for i = 1:size(Pyr_spiketimes.M, 2)
    
    if(~isnan(Pyr_spiketimes.M(1,i)))
        frames = get_Pyr_windowed_frames(Pyr_spiketimes.M(:,i));
        
        if size(frames, 1) > 0
            filenm = ['training_data/Pyr_' num2str(Pyr_spiketimes.M(1,i)) '.mat'];
            save(filenm, 'frames');
            
            Pyr_frames = [Pyr_frames; frames];  
        end
    end
    
end

%% SST data

SST_spiketimes = load('../Data/InVitro/SST_spiketimes.mat');

SST_frames = [];

for i = 1:size(SST_spiketimes.M, 2)
    
    if(~isnan(SST_spiketimes.M(1,i)))
        frames = get_SST_windowed_frames(SST_spiketimes.M(:,i));
        
        if size(frames, 1) > 0
           filenm = ['training_data/SST_' num2str(SST_spiketimes.M(1,i)) '.mat'];
           save(filenm, 'frames');
           
           SST_frames = [SST_frames; frames];
        end
    end
    
end

save('all_frames.mat', 'PV_frames', 'Pyr_frames', 'SST_frames');

