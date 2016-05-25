function frames = get_PV_windowed_frames(upstate_vector)
% Converts spike times into a spike waveform and
% breaks waveform into frames

start_time = upstate_vector(2);
upstate_time = upstate_vector(3) - start_time;

sample_shift = 0.004; %4ms measurement interval

time_intervals = 0:sample_shift:upstate_time;

upstate_waveform = zeros(1, length(time_intervals));

for j = 7:length(upstate_vector)
    if isnan(upstate_vector(j))
        continue;
    end
    spike_time = upstate_vector(j) - start_time;
    spike_sample = round(spike_time/sample_shift);
    
    upstate_waveform(spike_sample) = 1;
    
end

% %debug
% stem(time_intervals, upstate_waveform,'*');
% pause(1);

window_size = 1;    % 1 second
window_size = window_size/sample_shift; % in samples

window_shift = 0.5; % 0.5 second shift
window_shift = window_shift/sample_shift; % in samples

frames = enframe(upstate_waveform, window_size, window_shift);

%disp(size(frames));

end