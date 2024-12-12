clear all;
close all;
clc;

% You may need to add voicebox (a Matlab toolbox) into your path
code_folder = "C:\Users\francesco\anaconda3\envs\ALS\ALS-ML";
path_HC = fullfile(code_folder, "Data\Healthy Control\Normal\Audio\");
path_ALS = fullfile(code_folder, "Data\ALS\Normal\Audio\");
path_HC_PA = fullfile(code_folder, "Data\Healthy Control\PA\Audio\");
path_ALS_PA = fullfile(code_folder, "Data\ALS\PA\Audio\");
path_HC_PATAKA = fullfile(code_folder, "Data\Healthy Control\PATAKA\Audio\");
path_ALS_PATAKA = fullfile(code_folder, "Data\ALS\PATAKA\Audio\");

audio_trim(path_HC,1);
audio_trim(path_ALS,1);

audio_trim(path_HC,2);
audio_trim(path_ALS,2);

audio_trim(path_HC_PA,1);
audio_trim(path_ALS_PA,1);

audio_trim(path_HC_PA,2);
audio_trim(path_ALS_PA,2);

audio_trim(path_HC_PATAKA,1);
audio_trim(path_ALS_PATAKA,1);

audio_trim(path_HC_PATAKA,2);
audio_trim(path_ALS_PATAKA,2);

function audio_trim(path,f)
    if f==1
        file_csv=fullfile(path, "table.csv");
    else
        file_csv=fullfile(path, "table_th.csv");
    end
    if exist(file_csv, 'file')
        fileID=fopen(file_csv, "w");
        fclose(fileID);
    end
    
    fileID=fopen(file_csv, "w");
    files = dir(fullfile(path, "*.wav"));
    for i=1:numel(files)
        filename=fullfile(path, files(i).name);
        [y, fs]=audioread(filename);
        th=2*std(y);

        vad=vadsohn(y,fs);

            % Concatenate 0s at the beginning and end of vad
        padded_vad = [0; vad; 0];
        
        % Compute differences between consecutive elements
        differences = abs(diff(padded_vad));
        
        % Find indices where differences are equal to 1
        aux = find(differences == 1);
    
        for j = 1:2:numel(aux)-1
            start_sample = aux(j);
            end_sample = aux(j+1)-1;
            if f==2
                if max(y(start_sample:end_sample))>=th
                    name=strrep(files(i).name, ".wav", "");
                    fprintf(fileID, '%s,%d,%d\n', name, start_sample, end_sample);
                end
            else
                name=strrep(files(i).name, ".wav", "");
                fprintf(fileID, '%s,%d,%d\n', name, start_sample, end_sample);
            end
        end
    end
    fclose(fileID);
end