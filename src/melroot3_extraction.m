function [ stack_F ] = melroot3_extraction( y,fs )
%Extract MelRoot3 features and stack with a 160ms window
%Input:     y:  speech signal
%           fs: sampling rate (Hz)
%Output:    F:  MelRoot3 features, colums correspond to the dimension of
%               features


if size(y,2)~=1
    y = y(:,1);
end
if fs~=16000
    [y,fs] = resample(y,16000,fs); % If sampling rate is not 16kHz, resample it to 16kHz.
    fs = 16000;
end


F1=melbank_r3(y,fs,'a',12,320,160); % Extract MelRoot3 from each frame of speech. frame length is 20ms with 10ms overlap. 
stack_F = [];
for i = 1:16:size(F1,2)-15
    mtx = F1(:,i:i+15);
    vec = reshape(mtx,1,12*16);
    stack_F = [stack_F;vec];
end

end

