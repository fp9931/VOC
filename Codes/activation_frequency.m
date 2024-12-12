%% Activation frequency
% Function  to extrapolate the activation frequency feature from audio
% files. Activation frequency is a feature computed counting the number of
% active segment normalized by the overall activity time in the audio file.
%
%INPUT:
%           path: path to the folder containing the audio files
%
%           csv: csv file containig start and end time of audio segments
%           computed with VAD not thresholded
%
%           csv_th: csv file containig start and end time of audio segments
%           computed with VAD thresholded
%
%           varargin: csv file containig start and end time of audio segments
%           computed with vosk (optional parameter)
%
% OUTPUT:
%           af: 2xN array (or 3xN if vosk csv is passed) containing the
%           value of the activation frequency for each audio file. each row is a csv, N is the
%           number of patients

function af = activation_frequency(path, csv, csv_th, varargin)

    files = dir(fullfile(path, "*.wav"));

    for i=1:numel(files)
        filename=fullfile(path, files(i).name);
        [y, fs]=audioread(filename);
        ind=zeros(size(y));
        s = inf;
        e = 0;
        for j=1:height(csv)
            name=append(string(csv{j,1}), '.wav');
            if strcmp(name, files(i).name)
                if csv{j,2}<s
                    s=csv{j,2};
                end
                if e<csv{j,3}
                    e=csv{j,3};
                end
                ind(csv{j,2}:csv{j,3})=1;
            end
        end
        e = e/fs; 
        s =s/fs;
        t=e-s;

        s = inf;
        e = 0;
        ind_th=zeros(size(y));
        for j=1:height(csv_th)
            name=append(string(csv_th{j,1}), '.wav');
            if strcmp(name, files(i).name)
                if csv_th{j,2}<s
                    s=csv_th{j,2};
                end
                if e<csv_th{j,3}
                    e=csv_th{j,3};
                end
                ind_th(csv_th{j,2}:csv_th{j,3})=1;
            end
        end
        e = e/fs; 
        s =s/fs;
        t_th=e-s;
        ind=ind(:)';
        ind = [0, ind, 0];
        aux=diff(ind);
        af(1,i)=sum(aux==1)/t;
        
        ind_th=ind_th(:)';
        ind_th = [0 ind_th 0];
        au = diff(ind_th);
        af(2,i)=sum(au==1)/t_th;
        
        if length(varargin)>=1
            ind_vosk=zeros(size(y));
            csv_vosk=varargin{1};
            s = inf;
            e = 0;
            for j=1:height(csv_vosk)
                name=append(string(csv_vosk{j,1}), '.wav');
                if strcmp(name, files(i).name)
                if csv{j,2}<s
                    s=csv{j,2};
                end
                if e<csv{j,3}
                    e=csv{j,3};
                end
                    ind_vosk(csv_vosk{j,2}:csv_vosk{j,3})=1;
                end
            end
            t = (e-s)/s;
            ind_vosk=ind_vosk(:)';
            ind_vosk = [0 ind_vosk 0];
            aux=diff(ind_vosk);
            af(3,i)=sum(aux==1)/t;
        end
    end
end