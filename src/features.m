% clc
% clear all
% close all

% Define general path
currentPath = pwd;
pathParts = strsplit(currentPath, filesep);
numParts = numel(pathParts);
newPathParts = pathParts(1:numParts-1);
rootPath = strjoin(newPathParts, filesep);
t_path = fullfile(rootPath, '\Results\timings');
features_path = fullfile(rootPath,'\Results');

data_path = fullfile(rootPath, '\Data');
folders = dir(data_path);
folders_clean = folders(3:end);

F_overall = {};
F_overall_manual = {};

name = {};
task = {};
num_rep = {};
duration = {};
articulation_duration = {};
mean_duration = {};
std_duration = {};
median_duration = {};
ae = {};
ae_manual = {};

min_length = 1000;
min_length_manual = 1000;
len = 0;
count = 1;
for f=6:length(folders_clean)
    phonation_path = fullfile(data_path, folders_clean(f).name);
    timings_path = fullfile(t_path, folders_clean(f).name);
    files = dir(fullfile(phonation_path, '*.wav'));
    files_timings = dir(fullfile(timings_path, '*.xlsx'));
    
    if f==6
        len = length(files);
    end
    lenbefore = len;
    for i=1:length(files)
        len = length(files);
        clc; disp(['Processing file ', num2str(i)])
   
        [y, fs] = audioread(fullfile(files(i).folder, files(i).name));
        timings = readtable(fullfile(files_timings(i).folder, files_timings(i).name));

        start = timings.Start;
        stop = timings.Stop;

        artic_dur = 0;
        for j=1:length(stop)
            artic_dur = artic_dur + (stop(j)-start(j));
        end

        duration_interval = stop-start;
        mean_duration{count} = mean(duration_interval);
        std_duration{count} = std(duration_interval);
        median_duration{count} = median(duration_interval);
        
        name{count} = files(i).name(1:5);
        task{count} = files(i).name(end-5:end-4);
        num_rep{count} = length(start);
        duration{count} = stop(end) - start(1);
        articulation_duration{count} = artic_dur;
        speech_rate{count} = length(start)/(stop(end) - start(1));
        articulation_rate{count} = length(start)/artic_dur;

        vad = vadsohn(y,fs);
        y_new = y(vad==1);
        y_nor = inten_norm(y_new,fs);
        F = melroot3_extraction(y_nor,16000);

        if height(F) < min_length
            min_length = height(F);
        end

        F_overall{count} = F;

        y_mask = zeros(length(y),1);
        for k=1:length(start)
            start_sample = start(k)*fs;
            if start_sample == 0
                start_sample = 1;
            end
            end_sample = stop(k)*fs;
            y_mask(start_sample:end_sample) = 1;
        end

        y_voiced = y(y_mask == 1);
        y_nor_manual = inten_norm(y_voiced,fs);

        F_manual = melroot3_extraction(y_nor_manual,16000);

        if height(F_manual) < min_length_manual
            min_length_manual = height(F_manual);
        end

        F_overall_manual{count} = F_manual;

        padded_vad = [0; vad; 0];
        differences = diff(padded_vad);
        th=2*std(y);

        start_ = find(differences == 1);
        stop_ = find(differences == -1)-1; 

        k = 1;
        for l=1:length(start_)
            if max(y(start_(l):stop_(l)))<th
               idx_to_remove(k) = l;
               k = k+1;
            end
        end

        start_true = start_;
        stop_true = stop_;
        if exist('idx_to_remove')
            start_true(idx_to_remove) = [];
            stop_true(idx_to_remove) = [];
        end

        ind=zeros(size(y));

        for j=1:length(start_true)
            s = start_true(j);
            e = stop_true(j);
            ind(s:e) = 1;
        end

        count_ind=sum(ind);
        ar{count}=count_ind/size(ind,1);


        t = stop_true(end)/fs - start_true(1)/fs;
        ind = ind(:)';
        ind = [0 ind 0];
        au = diff(ind);
        af{count} = sum(au==1)/t;


        ind = zeros(length(y),1);
        for s=1:length(start)
            start_sample = start(s)*fs;
            if start_sample == 0
                start_sample = 1;
            end
            end_sample = stop(s)*fs;
            ind(start_sample:end_sample) = 1;
        end

        t = stop(end) - start(1);

        count_ind=sum(ind);
        ar_manual{count}=count_ind/size(ind,1);

        ind = ind(:)';
        ind = [0 ind 0];
        au = diff(ind);
        af_manual{count} = sum(au==1)/t;

        clear idx_to_remove

        count = count +1;
    end
end

for i=1:length(F_overall)
    ae{i} = artic_ent(F_overall{i},min_length);
end
for i=1:length(F_overall_manual)
    ae_manual{i} = artic_ent(F_overall_manual{i},min_length_manual);
end
%%
% Save matrix 
column_name = ["name","task", "tot repetition", "overall duration", "mean duration", "std duration", "median duration","articulation duration", "speech rate", "articualtion rate", "activation frequency", "activation ratio", "articulation entropy", "activation frequency manual", "activation ratio manual", "articulation entropy manual"];
tab = table(name',task',num_rep', duration', mean_duration', std_duration',median_duration',articulation_duration',speech_rate',articulation_rate',af',ar', ae', af_manual', ar_manual', ae_manual','VariableNames',column_name);
writetable(tab,fullfile(features_path,'syllables_entire.xlsx'))
