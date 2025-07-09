clc
clear all
close all

% Define general path
currentPath = pwd;
pathParts = strsplit(currentPath, filesep);
numParts = numel(pathParts);
newPathParts = pathParts(1:numParts-1);
rootPath = strjoin(newPathParts, filesep);
timings_path = fullfile(rootPath, '\Results\timings\test13_05');

data_path = fullfile(rootPath, '\Data');
folders = dir(data_path);
folders_clean = folders(3:end);

for f=8:length(folders_clean)

    phonation_path = fullfile(data_path, folders_clean(f).name);
    files = dir(fullfile(phonation_path, '*.wav'));
    
    column_name = ["Start", "Stop"];
    times = zeros(length(files),2);
    
    for i=1:length(files)
        clc; disp(['Processing file ', num2str(i)])
        name = files(i).name(1:5);
    
        [y, fs] = audioread(fullfile(files(i).folder, files(i).name));
    
        order = 5;
        fc = [50, 1000]/(fs/2);
        [b,a] = butter(order,fc,"bandpass");
        filtered_y = filtfilt(b,a,y);

        filtered_y = filtered_y-mean(filtered_y);
        filtered_y = filtered_y/max(filtered_y);

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

        smooth_ste = smoothdata(ste);

        nl = 2000;
        [counts, edges] = histcounts(smooth_ste, nl);
        edges = edges(1:end-1);

        tu = otsu_threshold(counts, edges);

        ste_below_tu = ste(smooth_ste < tu);
        [counts_low, edges_low] = histcounts(ste_below_tu, nl);
        edges_low = edges_low(1:end-1);
        tl = otsu_threshold(counts_low, edges_low);
        
        V = zeros(size(smooth_ste));
        % tu = median(smooth_ste);   %Usare solo mediana come th
        V(smooth_ste >= tu) = 1;

        % V(ste < tl) = 0;
        % V(ste >= tl & ste < tu) = 1;
        for k = 2:length(V)-1
            if V(k) == 1 && V(k-1) == 0 && V(k+1) == 0
                V(k) = 0;
            end
        end

        limits = diff([0; V]);
        start = find(limits == 1);
        stop = find(limits == -1);

        if isempty(stop)
            stop = length(smooth_ste);
        end
        if isempty(start)
            start = 0;
        end

        if (length(stop) < length(start))
            stop = [stop;length(smooth_ste)];
        end
        
        start_time = (start - 1) * step_size;
        stop_time = (stop - 1) * step_size;
        
        if i == 153
            figure();plot(y);hold on;title([files(i).name(1:5), ' Energy'])
            xline(start_time);
            xline(stop_time, 'r');
        % % yline(tu, 'b')
        end

        tab = table(start_time./fs,stop_time./fs,'VariableNames',column_name);
        writetable(tab,fullfile(timings_path,folders_clean(f).name,[name,'.xlsx']))
       
    end

    break
    
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