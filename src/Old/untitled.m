close all
clear all
clc
path = 'C:\Users\mimos\Projects\VOC\Results\burst_time';

folder_audio = natsortfiles(dir(fullfile(path, '*.wav')));
folder_timings = natsortfiles(dir(fullfile(path, '*.TextGrid')));
dur_VOT = [];
mean_durVOT = [];
names = cell(153,1);
vot = zeros(length(folder_audio),1);
start = zeros(length(folder_audio),1);
stop = zeros(length(folder_audio),1);
max_amp_db = zeros(length(folder_audio),1);
var_freq = zeros(length(folder_audio),1);
mean_freq = zeros(length(folder_audio),1);
peak_freq = zeros(length(folder_audio),1);
energy_burst = zeros(length(folder_audio),1);
energy_drop_db = zeros(length(folder_audio),1);
hf_lf_ratio = zeros(length(folder_audio),1);
SM_feature = zeros(length(folder_audio),12);
DCT_2D_tot = zeros(length(folder_audio),9);
tot_names = cell(length(folder_audio),1);
% k = 1;

burst_win = 5;
vowel_win = 10; 
for i=1:length(folder_audio)
    
    name = folder_audio(i).name(1:8);
    name_complete = folder_audio(i).name(1:end-4);
    tot_names{i} = name_complete;
    % if i>1
    %     if string(name) ~= string(prev_name)
    %         mean_durVOT = [mean_durVOT; mean(dur_VOT)];
    %         names{k} = prev_name;
    %         k = k+1;
    %         dur_VOT = [];
    % 
    %     end
    % end
    audio_name = fullfile(folder_audio(i).folder, folder_audio(i).name);
    txg_name = fullfile(folder_timings(i).folder, folder_timings(i).name);
    tg_id = tgDetectEncoding(txg_name);
    tg = tgRead(txg_name,tg_id);
    vot(i) = tg.tier{1,2}.T2(2) - tg.tier{1,2}.T1(2);
    % dur_VOT = [dur_VOT; vot(i)];
    start(i) = tg.tier{1,2}.T1(2);
    stop(i) = tg.tier{1,2}.T2(2);

    % prev_name = name;

    % Features
    [y,fs] = audioread(audio_name);
    filtered_y = y;

    % order = 5;
    % fc = [50, 1000]/(fs/2);
    % [b,a] = butter(order,fc,"bandpass");
    % filtered_y = filtfilt(b,a,y);
    % filtered_y = filtered_y-mean(filtered_y);
    % filtered_y = filtered_y / max(abs(filtered_y));

    burst_time = start(i);
    vowel_time = stop(i);

    burst_win_ms = round(burst_win / 1000*fs);
    vowel_win_ms = round(vowel_win / 1000*fs);

    burst_start = round(burst_time * fs);
    burst_end = round(vowel_time*fs)-1;
    burst = filtered_y(burst_start:burst_end);

    vowel_start = round(vowel_time * fs);
    vowel_end = length(filtered_y);
    vowel = filtered_y(vowel_start:vowel_end);
            

    % burst_start = max(1, round(burst_time * fs) - burst_win/2);
    % burst_end = min(length(filtered_y), burst_start + burst_win - 1);
    % burst = filtered_y(burst_start:burst_end);
    % 
    % vowel_start = round(vowel_time * fs);
    % vowel_end = min(length(filtered_y), vowel_start + vowel_win - 1);
    % vowel = filtered_y(vowel_start:vowel_end);

    % (k) Burst max amplitude in dB
    max_amp = max(abs(burst));
    max_amp_db(i) = 20 * log10(max_amp);

    % (l-o) Frequencfiltered_y analysis of burst
    N = 512;
    win = hamming(length(burst));
    spec = abs(fft(burst .* win, N));
    f = linspace(0, fs/2, N/2+1);
    psd = spec(1:N/2+1).^2;

    % Frequency bands
    hf_idx = find(f > 2000); % >2kHz
    lf_idx = find(f <= 2000); % ≤2kHz
    hf_power = sum(psd(hf_idx));
    lf_power = sum(psd(lf_idx));
    hf_lf_ratio(i) = hf_power / (lf_power);

    % Mean frequency
    mean_freq(i) = sum(f .* psd') / sum(psd);
    % Variance of frequency
    var_freq(i) = sum(((f - mean_freq(i)).^2) .* psd') / sum(psd);
    % Peak frequency
    [~, peak_idx] = max(psd);
    peak_freq(i) = f(peak_idx);

    % (p) Energy drop between burst and vowel onset
    energy_burst(i) = mean(burst.^2);
    energy_vowel = mean(vowel.^2);
    energy_drop_db(i) = 10 * log10((energy_burst(i)) / (energy_vowel));


    %SM FEATURES
    x = filtered_y(start(i)*fs:stop(i)*fs);
    % Parameters
    frame_duration_ms = 6;      % in milliseconds
    frame_shift_ms = 1;         % in milliseconds
    fft_size = 1024;            % FFT length
    
    % Convert durations from ms to samples
    frame_length = round(fs * frame_duration_ms / 1000);  % samples per frame
    frame_shift = round(fs * frame_shift_ms / 1000);      % hop size
    
    % Number of frames
    num_frames = floor((length(x) - frame_length) / frame_shift) + 1;
    
    % Preallocate
    X_fft = zeros(fft_size, num_frames);
    
    % Hamming window
    win = hamming(frame_length);
    
    % Frame-by-frame FFT
    for k = 1:num_frames
        idx_start = (k-1)*frame_shift + 1;
        idx_end = idx_start + frame_length - 1;
        frame = x(idx_start:idx_end);
        windowed = frame .* win;
        
        % Zero-padding to fft_size and FFT computation
        spectrum = fft(windowed, fft_size);
        X_fft(:, k) = spectrum;
    end

    P = real(X_fft).^2 + imag(X_fft).^2;
    p = P./sum(P(1:fft_size/2,:));
   
    f = linspace(0, fs, fft_size)';
    L1 = zeros(1,width(p));
    L2 = zeros(1,width(p));
    L3 = zeros(1,width(p));
    L4 = zeros(1,width(p));

    for k=1:width(p)
        L1(k) = sum(p(1:fft_size/2,k).*f(1:fft_size/2));
        L2(k) = sum((p(1:fft_size/2,k).*((f(1:fft_size/2)-L1(k)).^2)).^(1/2));
        L3(k) = sum((p(1:fft_size/2,k).*((f(1:fft_size/2)-L1(k)).^3)).^(1/3));
        L4(k) = sum((p(1:fft_size/2,k).*((f(1:fft_size/2)-L1(k)).^4)).^(1/4));
    end

    L1_dct = dct(L1);
    L2_dct = dct(L2); 
    L3_dct = dct(L3); 
    L4_dct = dct(L4);  
    
    if length(L1) < 3
        L1_dct3 = NaN(3);
        L2_dct3 = NaN(3);
        L3_dct3 = NaN(3);
        L4_dct3 = NaN(3);
    else
        L1_dct3 = L1_dct(1:3);
        L2_dct3 = L2_dct(1:3);
        L3_dct3 = L3_dct(1:3);
        L4_dct3 = L4_dct(1:3);
    end
    
    SM = [L1_dct3, L2_dct3, L3_dct3, L4_dct3];
    for k=1:length(SM)
        SM_feature(i,k) = SM(k);
    end

    % 2D-DCT Features
    if (stop(i)+0.02)*fs < length(filtered_y)
        x = filtered_y(start(i)*fs:(stop(i)+0.02)*fs);
    else
        x = filtered_y(start(i)*fs:end);
    end

    win_length_ms = 20;          % 20 ms window
    hop_length_ms = 10;          % 10 ms hop
    nfft = 512;                  % Number of FFT points
    win_length = round(fs * win_length_ms / 1000);
    hop_length = round(fs * hop_length_ms / 1000);

    frames = buffer(x, win_length, win_length - hop_length, 'nodelay');
    w = hamming(win_length);

    % Initialize LP spectro-temporal matrix
    num_frames = size(frames, 2);
    lp_spectrogram = zeros(nfft/2, num_frames);

    for k = 1:num_frames
        frame = frames(:, k) .* w;

        % LPC and LP spectrum
        a = lpc(frame, 10);
        [H, ~] = freqz(1, a, nfft, fs);
        lp_spectrum = 20 * log10(abs(H(1:nfft/2))); % keep positive freqs only

        lp_spectrogram(:, k) = lp_spectrum;
    end

    a = dct2(lp_spectrogram);
    if width(a)<3
        a_resized = NaN(3,3);
    else
        a_resized = a(1:3,1:3);
    end

    dct_2d = reshape(a_resized, 1, []);
    for k=1:length(dct_2d)
        DCT_2D_tot(i,k) = dct_2d(k);
    end

    

    %MFCC Features
    % x = filtered_y(stop(i)*fs:(stop(i)+0.02)*fs);
    % fft_len = 256;
    % num_mel_filters = 26;   
    % pre_emphasized = filter([1 -0.97], 1, x);
    % 
    % spectrum = abs(fft(pre_emphasized, fft_len)).^2;
    % spectrum = spectrum(1:fft_len/2+1); 
    % 
    % mel_fb = melfb(num_mel_filters, fft_len, fs);  % size: [num_mel_filters x (fft_len/2+1)]
    % mel_energies = mel_fb * spectrum;
    % log_mel = log(mel_energies + eps); 
    % 
    % mfcc = dct(log_mel);     % Returns [num_mel_filters x 1]
    % mfcc = mfcc(1:13); 

end

%%
column_name = ["SBJ","VOT","Start","Stop","Max burst Amplitude (dB)", "HF LF ratio", "Mean freq", "Var freq", "Peak freq", "Energy burst", "Energy drop", "SM L1 1", "SM L1 2", "SM L1 3", "SM L2 1", "SM L2 2", "SM L2 3", "SM L3 1", "SM L3 2", "SM L3 3", "SM L4 1", "SM L4 2", "SM L4 3", "2D-DCT 1", "2D-DCT 2", "2D-DCT 3", "2D-DCT 4", "2D-DCT 5", "2D-DCT 6", "2D-DCT 7", "2D-DCT 8", "2D-DCT 9"];
% % tab = table(mean_durVOT,names,'VariableNames',column_name);
% % writetable(tab,fullfile("C:\Users\mimos\Projects\VOC\Results\test_burst",'MeanVOT.xlsx'))

tab_tot = table(tot_names,vot,start,stop,max_amp_db,hf_lf_ratio,mean_freq,var_freq,peak_freq,energy_burst,energy_drop_db, SM_feature(:,1),SM_feature(:,2),SM_feature(:,3),SM_feature(:,4),SM_feature(:,5),SM_feature(:,6),SM_feature(:,7),SM_feature(:,8),SM_feature(:,9),SM_feature(:,10),SM_feature(:,11),SM_feature(:,12), DCT_2D_tot(:,1),DCT_2D_tot(:,2),DCT_2D_tot(:,3),DCT_2D_tot(:,4),DCT_2D_tot(:,5),DCT_2D_tot(:,6),DCT_2D_tot(:,7),DCT_2D_tot(:,8),DCT_2D_tot(:,9),'VariableNames',column_name);
writetable(tab_tot,fullfile("C:\Users\mimos\Projects\VOC\Results\test_burst",'VOT_2.xlsx'))
