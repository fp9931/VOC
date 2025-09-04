path = "D:\Dottorato\Progetti\VOC\Dr.VOT\data\raw\mimosa";

folder_audio = natsortfiles(dir(fullfile(path, '*.wav')));
folder_tg = natsortfiles(dir(fullfile(path, '*.txt')));
idx_to_remove = [];
for i=1:length(folder_audio)
    name = folder_audio(i).name(1:end-4);
    name_tg = [name, '_fPI.txt'];
    name_tg_m = [name, '_mPI.txt'];

    ok = 0;
    for j=1:length(folder_tg)
        idx_f = find(string(folder_tg(j).name) == name_tg);
        idx_m = find(string(folder_tg(j).name) == name_tg_m);
        if ~isempty(idx_f) || ~isempty(idx_m)
            disp(name)
            ok = 1;
            break
        end
    end
    if ok == 0
        idx_to_remove = [idx_to_remove i];
    end
end

% for k=1:length(idx_to_remove)
%     delete(fullfile(folder_audio(idx_to_remove(k)).folder, folder_audio(idx_to_remove(k)).name));
% end



