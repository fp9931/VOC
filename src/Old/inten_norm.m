function [ y_nor ] = inten_norm( y,fs )
%Normalize the intensity of the input signal y to the same level of the
%reference signal (provided)
%Inpput: y:     the input speech signal
%Output: y_nor: the normalized signal


wav_ref=audioread('reference.wav');
[m,n]=size(wav_ref);
normal_power = 1/m*sum(wav_ref.*wav_ref);
if size(y,2)~=1
    y = y(:,1);
end
if fs~=16000
    [y,fs] = resample(y,16000,fs); % If sampling rate is not 16kHz, resample it to 16kHz.
    fs = 16000;
end
y_nor =sqrt(normal_power)/std(y)*y;

end

