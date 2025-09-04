clc
clear all
close all

% Define general path
currentPath = pwd;
pathParts = strsplit(currentPath, filesep);
numParts = numel(pathParts);
newPathParts = pathParts(1:numParts-1);
rootPath = strjoin(newPathParts, filesep);
timings_path = fullfile(rootPath, '\Results\timings');

data_path = fullfile(rootPath, '\Data');
folders = dir(data_path);
folders_clean = folders(3:end);

for f=1:length(folders_clean)-3

    phonation_path = fullfile(data_path, folders_clean(f).name);
    files = dir(fullfile(phonation_path, '*.wav'));
    
    column_name = ["ID", "Start", "Stop"];
    names = cell(length(files),1);
    times = zeros(length(files),2);
    
    for i=1:length(files)
        clc; disp(['Processing file ', num2str(i)])
        names{i} = files(i).name(1:5);
    
        [y, fs] = audioread(fullfile(files(i).folder, files(i).name));
    
        order = 5;
        fc = [50, 1000]/(fs/2);
        [b,a] = butter(order,fc,"bandpass");
        filtered_y = filtfilt(b,a,y);

        filtered_y = filtered_y-mean(filtered_y);
        filtered_y = filtered_y/max(filtered_y);
    
        y_abs = abs(filtered_y);
        lp_cutoff = 10 / (fs/2);
        [b_lp, a_lp] = butter(2, lp_cutoff, "low");
        envelope = filtfilt(b_lp, a_lp, y_abs);
    
        th_time = 0.2*mean(envelope);
    
        above_points = envelope >=th_time;
    
        limits = diff([0; above_points]);
    
        local_min = find(islocalmin(envelope) ~= 0);
    
        start = 0;
        stop = length(envelope);
    
        idx_start = find(limits == 1);
        idx_stop = find(limits == -1);
    
        if (length(idx_stop) < length(idx_start))
            idx_stop = [idx_stop;length(envelope)];
        end
        
        start_temp = [];
        stop_temp = [];
        for ind=1:length(idx_start)
    
            if (idx_stop(ind)-idx_start(ind))/fs >= 1
                start_temp = [start_temp idx_start(ind)];
                stop_temp = [stop_temp idx_stop(ind)];
            end
    
        end
        
        if ~isempty(start_temp)
            start_true = start_temp(1);
            stop_true = stop_temp(1);
            if length(start_temp) > 1
                for j=2:length(start_temp)
                    if (start_temp(j)-stop_temp(j-1))/fs < 1
                        start_true = start_temp(1);
                        stop_true = stop_temp(j);
                    end
                end
            end
    
            before_start = local_min(local_min < start_true);
            start = before_start(end);
            after_stop = local_min(local_min > stop_true);
            if ~isempty(after_stop)
                stop = after_stop(1);
            end
    
        else
            start = NaN;
            stop = NaN;
        end
    
        times(i,1) = start;
        times(i,2) = stop;
    
    end
     
    % tab = table(names,times(:,1)./fs,times(:,2)./fs,'VariableNames',column_name);
    % writetable(tab,fullfile(timings_path,[folders_clean(f).name,'.xlsx']))

end
