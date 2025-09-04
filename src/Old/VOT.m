clc
clear all
close all

% Define general path
currentPath = pwd;
pathParts = strsplit(currentPath, filesep);
numParts = numel(pathParts);
newPathParts = pathParts(1:numParts-1);
rootPath = strjoin(newPathParts, filesep);

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
    end
end