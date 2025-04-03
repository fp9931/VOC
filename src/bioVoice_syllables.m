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

for f=7:length(folders_clean)

    phonation_path = fullfile(data_path, folders_clean(f).name);
    files = dir(fullfile(phonation_path, '*.wav'));
    
    column_name = ["Start", "Stop"];
    times = zeros(length(files),2);
    
    for i=1:length(files)
        clc; disp(['Processing file ', num2str(i)])
        name = files(i).name(1:5);
    
        [y, fs] = audioread(fullfile(files(i).folder, files(i).name));

        y = y-mean(y);
        y = y/max(y);
    
        order = 5;
        fc = [50, 1000]/(fs/2);
        [b,a] = butter(order,fc,"bandpass");
        filtered_y = filtfilt(b,a,y);

        window = 0.02*fs;
        overlap = round(window/2);
        step_size = window - overlap;
        num_rep = floor((length(filtered_y) - overlap)/step_size);
    
        ste = zeros(num_rep,1);
        for k = 1:num_rep   
            start = (k-1)*step_size+1;
            stop = start + window -1;
            segment = filtered_y(start:stop);
            
            %Short time energy
            st_energy = sum(segment.^2);
            st_energy = (st_energy/length(segment));
            ste(k) = log10(st_energy);    
        end

        nl = 2000;
        [counts, edges] = histcounts(ste, nl);
        edges = edges(1:end-1);
        
        tu = otsu_threshold(counts, edges);

        ste_below_tu = ste(ste < tu);
        [counts_low, edges_low] = histcounts(ste_below_tu, nl);
        edges_low = edges_low(1:end-1);
        tl = otsu_threshold(counts_low, edges_low);
        
        V = zeros(size(ste));
        tu = median(ste);   %Usare solo mediana come th
        V(ste >= tu) = 1;
        % V(ste < tl) = 0;
        % V(ste >= tl & ste < tu) = 1;
        for k = 2:length(V)-1
            if V(k) == 1 && V(k-1) == 0 && V(k+1) == 0
                V(k) = 0;
            end
        end

        limits = diff([0; V]);
        start_temp = find(limits == 1);
        stop_temp = find(limits == -1);

        local_min = find(islocalmin(ste) ~= 0);
        start = [];
        stop = []; 
        before_start = [];
        after_stop = [];
        for j=1:length(start_temp)
            before_start = local_min(local_min <= start_temp(j));
            if ~isempty(before_start)
                start = [start; before_start(end)];
            else
                start = 0;
            end
        end
        if ~isempty(stop_temp)
            for j=1:length(stop_temp)
                after_stop = local_min(local_min >= stop_temp(j));
                if ~isempty(after_stop)
                    stop = [stop; after_stop(1)];
                else
                    if ~isempty(stop)
                        stop = [stop; length(ste)];
                    else
                        stop = length(ste);
                    end
                    
                end
            end

        else
            stop = length(ste);
        end

        if (length(stop) < length(start))
            stop = [stop;length(ste)];
        end
        
        start_time = (start - 1) * step_size;
        stop_time = (stop - 1) * step_size;
        
        idx_to_remove = [];
        for s = 1:length(start_time)
            if stop_time(s) - start_time(s) <= 0.1*fs
                idx_to_remove = [idx_to_remove; s]; 
            end
        end
        
        if ~isempty(idx_to_remove)
            start_time(idx_to_remove) = [];
            stop_time(idx_to_remove) = [];
        end

        % for s = 1:length(start_time)-1
        %     if start_time(s+1) - stop_time(s) <= 0.4*fs
        %         val = (start_time(s+1)+stop_time(s))/2;
        %         start_time(s+1) = val;
        %         stop_time(s) = val;
        %     end
        % end

        
            % figure();plot(ste);hold on;title(files(i).name(1:5))
            % xline(start);
            % xline(stop_temp, 'r');
            % yline(tu, 'r')

        figure();plot(y);hold on;title(files(i).name(1:5))
        xline(start_time);
        xline(stop_time);
 
        tab = table(start_time./fs,stop_time./fs,'VariableNames',column_name);
        writetable(tab,fullfile(timings_path,folders_clean(f).name,[name,'.xlsx']))
       
    end
end
     
function threshold = otsu_threshold(counts, edges)
    total = sum(counts);
    sumB = 0;
    wB = 0;
    maximum = 0;
    sum1 = sum(edges .* counts);
    
    for i = 1:length(counts)
        wB = wB + counts(i);
        if wB == 0
            continue;
        end
        wF = total - wB;
        if wF == 0
            break;
        end
        sumB = sumB + edges(i) * counts(i);
        mB = sumB / wB;
        mF = (sum1 - sumB) / wF;
        
        % Between-class variance
        varBetween = wB * wF * (mB - mF)^2;
        
        % Check if new maximum found
        if varBetween > maximum
            maximum = varBetween;
            threshold = edges(i);
        end
    end
end