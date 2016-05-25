function frames = enframe(waveform, window_size, window_shift)
% Breaks waveform into frames

if (mod(window_size,2)==0)
   window_size = window_size+1; 
end

half_wind_size = (window_size-1)/2;

% Find all spikes in the waveform
non_zero_indexs = find(waveform);

% Initialize frames
frames = zeros(length(non_zero_indexs), window_size);

k = 1;

for i = non_zero_indexs
   
    if ((i - half_wind_size) < 1 || (i + half_wind_size) > length(waveform))
       continue; 
    end
    
    frames(k,:) = waveform(i-half_wind_size:i+half_wind_size);
    k = k + 1;
end

frames(k:end,:) = [];

% %debug
% for i = 1:size(frames,1)
%    plot(frames(i,:));
%    title(i);
%    pause(0.5);
% end

end