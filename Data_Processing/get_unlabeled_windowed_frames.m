function frames = get_unlabeled_windowed_frames(upstate_vector)

% Assumes start time is 1 second before first spike
start_time = upstate_vector(1) - 1;

lastspiketime = find(isnan(upstate_vector),1,'first') - 1;
upstate_time = upstate_vector(lastspiketime) - start_time;

sample_shift = 0.004; %4ms measurement interval

time_intervals = 0:sample_shift:(upstate_time+0.5);

upstate_waveform = zeros(1, length(time_intervals));


upstate_vector(isnan(upstate_vector)) = [];
spike_sample = round((upstate_vector - start_time)/sample_shift);
upstate_waveform(spike_sample) = 1;

% %debug
% stem(time_intervals, upstate_waveform);
% pause(1);

window_size = 1;    % 1 second
window_size = window_size/sample_shift; % in samples

window_shift = 0.5; % 0.5 second shift
window_shift = window_shift/sample_shift; % in samples

frames = enframe(upstate_waveform, window_size, window_shift);



end

function frames = enframe(waveform, window_size, window_shift)
% Breaks waveform into frames

k = 1; i = 1;
frames = zeros(round(2*(length(waveform)/window_size))-2, window_size);

while(k + window_size <= length(waveform))
    frame = waveform(k:k+window_size-1);
    k = k + window_shift;
    i = i + 1;
    
    %disp(length(frame));
    frames(i,:) = frame;
    
%     plot(frame);
%     title(i);
%     pause(0.5);
end

% Get last frame here
if k < length(waveform)
    frame = waveform(k:end);
    frame = [frame zeros(1,window_size-length(frame))];
    
    frames = [frames; frame];
end

end