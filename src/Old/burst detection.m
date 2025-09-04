clear all
close all
clc

currentPath = pwd;
pathParts = strsplit(currentPath, filesep);
numParts = numel(pathParts);
newPathParts = pathParts(1:numParts-1);
rootPath = strjoin(newPathParts, filesep);
timings_path = fullfile(rootPath, '\Results\timings');

data_path = fullfile(rootPath, '\Data');
folders = dir(data_path);
folders_clean = folders(8:end);

durations = [];

for f=1:length(folders_clean)

    syllables_path = fullfile(data_path, folders_clean(f).name);
    files = dir(fullfile(syllables_path, '*.wav'));

    for i=60:length(files)
        name = files(i).name(1:5);
        timing_file = fullfile(timings_path, folders_clean(f).name,[name,'.xlsx']);
        timings = readtable(timing_file);

        start = timings.Start;
        stop = timings.Stop;

        [y, fs] = audioread(fullfile(files(i).folder, files(i).name));

        order = 5;
        fc = [50, 1000]/(fs/2);
        [b,a] = butter(order,fc,"bandpass");
        filtered_y = filtfilt(b,a,y);

        filtered_y = filtered_y-mean(filtered_y);
        filtered_y = filtered_y/max(filtered_y);

        for j=1:length(start)
            clc; disp(['Processing file ', num2str(i),': rep ', num2str(j)])

            effective_start = start(j); % 10 milliseconds before
            % segment = y(round(effective_start*fs):round(stop(j)*fs));            
            segment_y = filtered_y(floor(effective_start*fs):floor(stop(j)*fs));

            % Zero-Crossing Rate
            % window = 0.02*fs;
            % overlap = round(0.8*window);
            % step_size = window-overlap;
            % frame_length = window;
            % hop_length = overlap;
            % num_frames = floor((length(segment_y) - overlap)/step_size);
            % zcr = zeros(num_frames,1);
            % for k=1:num_frames
            %     start_in = (k-1)*step_size+1;
            %     stop_in = start_in + window -1;
            %     segment = segment_y(start_in:stop_in);
            %     zcr(k) = zerocrossrate(segment);
            % end
            % figure()
            % plot(smoothdata(zcr))

            vad = vadsohn(segment_y,fs);
            figure()
            plot(segment_y)
            hold on
            plot(vad*max(segment_y), 'r', 'LineWidth',2)
            title(files(i).name(1:end-4))

            % 
            % energy = sqrt(movmean(segment_y.^2, frame_length));
            % energy_frames = buffer(energy,frame_length,frame_length-hop_length);
            % rms_energy = sqrt(mean(energy_frames.^2,1));
            % 
            % num_frames = length(rms_energy);
            % times = (0:num_frames-1) * hop_length + 1;
            % 
            % aux = zeros(length(rms_energy));
            % th = mean(rms_energy)*0.1;
            % dRMS = [diff(rms_energy), 0];
            % idx_vowel = find(dRMS == max(dRMS));
            % ddRMS = [diff(dRMS), 0];
            % [peaks, locs] = findpeaks(ddRMS, "MinPeakProminence", 0.005);
            % idx_burst = locs(1);
            % 
            % vowel_onset = times(idx_vowel);
            % vowel_onset_time = (vowel_onset)/fs + effective_start;
            % burst_onset = times(idx_burst);
            % burst_onset_time = (burst_onset)/fs + effective_start;
            % 
            % figure()
            % plot(segment_y)
            % xline(vowel_onset)
            % xline(burst_onset)
            % title([name, 'rep', num2str(j)])
        end
        break
    end
    break
end

%% Per salvare le singole ripetizioni dopo averle segmentate con il metodo in bioVoice_syllables.m
clear all
close all
clc

currentPath = pwd;
pathParts = strsplit(currentPath, filesep);
numParts = numel(pathParts);
newPathParts = pathParts(1:numParts-1);
rootPath = strjoin(newPathParts, filesep);
timings_path = fullfile(rootPath, '\Results\timings');

data_path = fullfile(rootPath, '\Data');
folders = dir(data_path);
folders_clean = folders(8:end);

durations = [];

for f=1:length(folders_clean)

    syllables_path = fullfile(data_path, folders_clean(f).name);
    files = dir(fullfile(syllables_path, '*.wav'));

    for i=1:length(files)
        name = files(i).name(1:5);
        timing_file = fullfile(timings_path, folders_clean(f).name,[name,'.xlsx']);
        timings = readtable(timing_file);

        [y, fs] = audioread(fullfile(files(i).folder, files(i).name));
        start = timings.Start;
        stop = timings.Stop;

        for j=1:length(start)
            clc; disp(['Processing file ', files(i).name,': rep ', num2str(j)])
            effective_start = start(j)-0.005;
            if effective_start <= 0
                effective_start = 0.001;
            end
            segment = y(floor(effective_start*fs):floor(stop(j)*fs));           
            audiowrite(fullfile("D:/Dottorato/Progetti/VOC/Dr.VOT/data/raw/tot",[name, '_', folders_clean(f).name(end-1:end), 'rep_', num2str(j), '.wav']),segment, fs)
        end
    end
end