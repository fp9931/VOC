clear all
close all
clc

%% KA

audio_path = 'C:\Users\mimos\Projects\VOC\Data\Audio\rhythmKA';
timings_path =  "C:\Users\mimos\Projects\VOC\Data\Timings\rhythmKA";

subj = dir(timings_path);
subj_clean = subj(3:end);
ae_KA = [];

for sbj=1:length(subj_clean)
    if ~strcmp(subj_clean(sbj).name(1:2), 'PZ')
        continue
    end
    if strcmp(subj_clean(sbj).name(1:end-5), 'PZ068')
        continue
    end
    if strcmp(subj_clean(sbj).name(1:end-5), 'PZ017')
        continue
    end
    audio = fullfile(audio_path,[subj_clean(sbj).name(1:end-5), '_rhythmKA.wav']);                
    res = readtable(fullfile(timings_path, [subj_clean(sbj).name(1:end-5), '.xlsx']));
    starts = res.Start;
    stops = res.Stop;

    [y,fs] = audioread(audio);
    y = y(:,1);

    y_mask = zeros(length(y),1);
    for k=1:length(starts)
        y_mask(floor(starts(k)*fs)+1:floor(stops(k)*fs)-1,:) = 1;
    end

    y = y(y_mask==1);
    y_nor = inten_norm(y,fs);
    F = melroot3_extraction(y_nor,16000);
    ae_KA = [ae_KA; artic_ent(F,size(F,1))];
end

%% PA

audio_path = 'C:\Users\mimos\Projects\VOC\Data\Audio\rhythmPA';
timings_path =  "C:\Users\mimos\Projects\VOC\Data\Timings\rhythmPA";

subj = dir(timings_path);
subj_clean = subj(3:end);
ae_PA = [];

for sbj=1:length(subj_clean)
    if ~strcmp(subj_clean(sbj).name(1:2), 'PZ')
        continue
    end
    if strcmp(subj_clean(sbj).name(1:end-5), 'PZ068')
        continue
    end
    if strcmp(subj_clean(sbj).name(1:end-5), 'PZ017')
        continue
    end
    audio = fullfile(audio_path,[subj_clean(sbj).name(1:end-5), '_rhythmPA.wav']);                
    res = readtable(fullfile(timings_path, [subj_clean(sbj).name(1:end-5), '.xlsx']));
    starts = res.Start;
    stops = res.Stop;

    [y,fs] = audioread(audio);
    y = y(:,1);

    y_mask = zeros(length(y),1);
    for k=1:length(starts)
        y_mask(floor(starts(k)*fs)+1:floor(stops(k)*fs)-1,:) = 1;
    end

    y = y(y_mask==1);
    y_nor = inten_norm(y,fs);
    F = melroot3_extraction(y_nor,16000);
    ae_PA = [ae_PA; artic_ent(F,size(F,1))];
end

%% TA

audio_path = 'C:\Users\mimos\Projects\VOC\Data\Audio\rhythmTA';
timings_path =  "C:\Users\mimos\Projects\VOC\Data\Timings\rhythmTA";

subj = dir(timings_path);
subj_clean = subj(3:end);
ae_TA = [];

for sbj=1:length(subj_clean)
    if ~strcmp(subj_clean(sbj).name(1:2), 'PZ')
        continue
    end
    if strcmp(subj_clean(sbj).name(1:end-5), 'PZ068')
        continue
    end
    if strcmp(subj_clean(sbj).name(1:end-5), 'PZ017')
        continue
    end
    audio = fullfile(audio_path,[subj_clean(sbj).name(1:end-5), '_rhythmTA.wav']);                
    res = readtable(fullfile(timings_path, [subj_clean(sbj).name(1:end-5), '.xlsx']));
    starts = res.Start;
    stops = res.Stop;

    [y,fs] = audioread(audio);
    y = y(:,1);

    y_mask = zeros(length(y),1);
    for k=1:length(starts)
        y_mask(floor(starts(k)*fs)+1:floor(stops(k)*fs)-1,:) = 1;
    end

    y = y(y_mask==1);
    y_nor = inten_norm(y,fs);
    F = melroot3_extraction(y_nor,16000);
    ae_TA = [ae_TA; artic_ent(F,size(F,1))];
end