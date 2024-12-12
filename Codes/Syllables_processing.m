clear all;
close all;
clc;
%% Elaboration of PA files
%There are computation also on the not threshold files. Those will not be
%used in the regression
code_folder = pwd;
path_HC = fullfile(code_folder, "Data\Healthy Control\PA\Audio");
csv_HC = readtable(code_folder+"\Data\Healthy Control\PA\Audio\table.csv");
csv_HC_th = readtable(code_folder+"\Data\Healthy Control\PA\Audio\table_th.csv");

path_SLA = fullfile(code_folder, "Data\ALS\PA\Audio");
csv_SLA = readtable(code_folder+ "\Data\ALS\PA\Audio\table.csv");
csv_SLA_th = readtable(code_folder+"\Data\ALS\PA\Audio\table_th.csv");
%% Computation of Activation Ratio
ar_HC=activation_ratio(path_HC, csv_HC, csv_HC_th);
ar_SLA=activation_ratio(path_SLA, csv_SLA,csv_SLA_th);
%% Computation of Activation Frequency
af_HC=activation_frequency(path_HC,csv_HC, csv_HC_th);
af_SLA=activation_frequency(path_SLA,csv_SLA, csv_SLA_th);
%% Computation of Articulation Entropy. Not used
% ae_HC = ae_extraction(path_HC, csv_HC);
% ae_HC_th = ae_extraction(path_HC, csv_HC_th);
% 
% ae_SLA = ae_extraction(path_SLA, csv_SLA);
% ae_SLA_th = ae_extraction(path_SLA, csv_SLA_th);
%% Scatterplot
figure
hold on
scatter(af_HC(1,:),ar_HC(1,:),"red", "filled");
scatter(af_SLA(1,:), ar_SLA(1,:),"blue","filled");
title("Without treshold, PA files")
xlabel("Activation Frequency")
ylabel("Activation Ratio")
legend("HC", "ALS")
hold off
saveas(gcf, fullfile(code_folder, "Figures\Scatterplot_AR_over_AF_PA.png"));

figure
hold on
scatter(af_HC(2,:),ar_HC(2,:),"red", "filled");
scatter(af_SLA(2,:), ar_SLA(2,:),"blue","filled");
title("With treshold, PA files")
xlabel("Activation Frequency")
ylabel("Activation Ratio")
legend("HC", "ALS")
hold off

saveas(gcf, fullfile(code_folder, "Figures\Scatterplot_AR_over_AF_PA_w_th.png"));
%% Boxplots of AR
figure('Position', [100, 100, 1200, 800])
boxplot([ar_HC(1,:)'; ar_SLA(1,:)'], [zeros(10,1); 1+zeros(9,1)]);
labels = {'HC', 'ALS'};
set(gca, 'XTickLabel', labels);
ylabel('Activation Ratio');
title("Activation Ratio of PA files without threshold");
%signicance test
[p, h, stats] = ranksum(ar_HC(1,:), ar_SLA(1,:));
num_asterisks = '';
if h == 1
    if p < 0.001
        num_asterisks = '***';
    elseif p < 0.01
        num_asterisks = '**';
    elseif p < 0.05
        num_asterisks = '*';
    end
end
if ~isempty(num_asterisks)
    % Disegna una linea sopra i boxplot
    max_y = max([ar_HC(1,:)'; ar_SLA(1,:)']);
    line([1, 2], [max_y + 0.1, max_y + 0.1], 'Color', 'k', 'LineWidth', 0.7);
    line([1, 1], [max_y + 0.9, max_y + 0.1], 'Color', 'k', 'LineWidth', 0.7);
    line([2, 2], [max_y + 0.9, max_y + 0.1], 'Color', 'k', 'LineWidth', 0.7);

    % Aggiungi gli asterischi sopra la linea
    text(1.5, max_y + 0.12, num_asterisks, 'HorizontalAlignment', 'center', 'FontSize', 16);
end
xlim([0, 3]);
ylim([min([ar_HC(1,:)'; ar_SLA(1,:)']) max([ar_HC(1,:)'; ar_SLA(1,:)']) + 0.15]);
% annotation('textbox', [0.8, 0.1, 0.3, 0.1], 'String', ...
%     sprintf('Wilcoxon test\np-value: %.4f', ...
%      p), ...
%     'FitBoxToText', 'on', 'BackgroundColor', 'white');
saveas(gcf, fullfile(code_folder,"Figures\Activation_Ratio_PA.png"));


figure('Position', [100, 100, 1200, 800])
boxplot([ar_HC(2,:)'; ar_SLA(2,:)'], [zeros(10,1); 1+zeros(9,1)]);
labels = {'HC', 'ALS'};
set(gca, 'XTickLabel', labels);
ylabel('Activation Ratio');
title("Activation Ratio of PA files with threshold");
%signicance test
[p, h, stats] = ranksum(ar_HC(2,:), ar_SLA(2,:));
if h == 1
    if p < 0.001
        num_asterisks = '***';
    elseif p < 0.01
        num_asterisks = '**';
    elseif p < 0.05
        num_asterisks = '*';
    end
end
if ~isempty(num_asterisks)
    % Disegna una linea sopra i boxplot
    max_y = max([ar_HC(2,:)'; ar_SLA(2,:)']);
    line([1, 2], [max_y + 0.1, max_y + 0.1], 'Color', 'k', 'LineWidth', 0.7);
    line([1, 1], [max_y + 0.9, max_y + 0.1], 'Color', 'k', 'LineWidth', 0.7);
    line([2, 2], [max_y + 0.9, max_y + 0.1], 'Color', 'k', 'LineWidth', 0.7);

    % Aggiungi gli asterischi sopra la linea
    text(1.5, max_y + 0.12, num_asterisks, 'HorizontalAlignment', 'center', 'FontSize', 16);
end
xlim([0, 3]);
ylim([min([ar_HC(2,:)'; ar_SLA(2,:)']) max([ar_HC(2,:)'; ar_SLA(2,:)']) + 0.15]);
% annotation('textbox', [0.8, 0.1, 0.3, 0.1], 'String', ...
%     sprintf('Wilcoxon test\np-value: %.4f', ...
%      p), ...
%     'FitBoxToText', 'on', 'BackgroundColor', 'white');
saveas(gcf, fullfile(code_folder,"Figures\Activation_Ratio_PA_w_th.png"));

%% Boxplot of AF
figure('Position', [100, 100, 1200, 800])
boxplot([af_HC(1,:)'; af_SLA(1,:)'], [zeros(10,1); 1+zeros(9,1)]);
labels = {'HC', 'ALS'};
set(gca, 'XTickLabel', labels);
ylabel('Activation Frequency');
title("Activation Frequency of PA files without threshold");
%signicance test
[p, h, stats] = ranksum(af_HC(1,:), af_SLA(1,:));
num_asterisks = '';
if h == 1
    if p < 0.001
        num_asterisks = '***';
    elseif p < 0.01
        num_asterisks = '**';
    elseif p < 0.05
        num_asterisks = '*';
    end
end
if ~isempty(num_asterisks)
    % Disegna una linea sopra i boxplot
    max_y = max([af_HC(1,:)'; af_SLA(1,:)']);
    line([1, 2], [max_y + 0.5, max_y + 0.5], 'Color', 'k', 'LineWidth', 0.7);
    line([1, 1], [max_y + 0.4, max_y + 0.5], 'Color', 'k', 'LineWidth', 0.7);
    line([2, 2], [max_y + 0.4, max_y + 0.5], 'Color', 'k', 'LineWidth', 0.7);

    % Aggiungi gli asterischi sopra la linea
    text(1.5, max_y + 0.6, num_asterisks, 'HorizontalAlignment', 'center', 'FontSize', 16);
end
xlim([0, 3]);
ylim([min([af_HC(1,:)'; af_SLA(1,:)']) max([af_HC(1,:)'; af_SLA(1,:)']) + 0.8]);
% annotation('textbox', [0.8, 0.1, 0.3, 0.1], 'String', ...
%     sprintf('Wilcoxon test\np-value: %.4f', ...
%      p), ...
%     'FitBoxToText', 'on', 'BackgroundColor', 'white');
saveas(gcf, fullfile(code_folder,"Figures\Activation_Frequency_PA.png"));

figure('Position', [100, 100, 1200, 800])
boxplot([af_HC(2,:)'; af_SLA(2,:)'], [zeros(10,1); 1+zeros(9,1)]);
labels = {'HC', 'ALS'};
set(gca, 'XTickLabel', labels);
ylabel('Activation Frequency');
title("Activation Frequency of PA files with threshold");
%signicance test
[p,h,stats] = ranksum(af_HC(2,:), af_SLA(2,:));
num_asterisks = '';
if h == 1
    if p < 0.001
        num_asterisks = '***';
    elseif p < 0.01
        num_asterisks = '**';
    elseif p < 0.05
        num_asterisks = '*';
    end
end
if ~isempty(num_asterisks)
    % Disegna una linea sopra i boxplot
    max_y = max([af_HC(2,:)'; af_SLA(2,:)']);
    line([1, 2], [max_y + 0.5, max_y + 0.5], 'Color', 'k', 'LineWidth', 0.7);
    line([1, 1], [max_y + 0.4, max_y + 0.5], 'Color', 'k', 'LineWidth', 0.7);
    line([2, 2], [max_y + 0.4, max_y + 0.5], 'Color', 'k', 'LineWidth', 0.7);

    % Aggiungi gli asterischi sopra la linea
    text(1.5, max_y + 0.6, num_asterisks, 'HorizontalAlignment', 'center', 'FontSize', 16);
end
xlim([0, 3]);
ylim([min([af_HC(2,:)'; af_SLA(2,:)']) max([af_HC(2,:)'; af_SLA(2,:)']) + 0.8]);
% annotation('textbox', [0.8, 0.1, 0.1, 0.1], 'String', ...
%     sprintf('Wilcoxon test\np-value: %.4f', ...
%      p), ...
%     'FitBoxToText', 'on', 'BackgroundColor', 'white');
saveas(gcf, fullfile(code_folder,"Figures\Activation_Frequency_PA_w_th.png"));
%% Elaboration of PATAKA files
path_HC = fullfile(code_folder, "Data\Healthy Control\PATAKA\Audio");
csv_HC = readtable("Healthy Control\PATAKA\Audio\table.csv");
csv_HC_th = readtable("Healthy Control\PATAKA\Audio\table_th.csv");

path_SLA = fullfile(code_folder, "Data\SLA\PATAKA\Audio");
csv_SLA = readtable("SLA\PATAKA\Audio\table.csv");
csv_SLA_th = readtable("SLA\PATAKA\Audio\table_th.csv");
%% Computation of Activation Ratio
ar_HC=activation_ratio(path_HC, csv_HC, csv_HC_th);
ar_SLA=activation_ratio(path_SLA, csv_SLA,csv_SLA_th);
%% Computation of Activation Frequency
af_HC=activation_frequency(path_HC,csv_HC, csv_HC_th);
af_SLA=activation_frequency(path_SLA,csv_SLA, csv_SLA_th);
%% Computation of Articulation Entropy. Not used
% ae_HC = ae_extraction(path_HC, csv_HC);
% ae_HC_th = ae_extraction(path_HC, csv_HC_th);
% 
% ae_SLA = ae_extraction(path_SLA, csv_SLA);
% ae_SLA_th = ae_extraction(path_SLA, csv_SLA_th);
%% Scatterplot
figure
hold on
scatter(af_HC(1,:),ar_HC(1,:),"red", "filled");
scatter(af_SLA(1,:), ar_SLA(1,:),"blue","filled");
title("Without treshold, PATAKA files")
xlabel("Activation Frequency")
ylabel("Activation Ratio")
legend("HC", "ALS")
hold off

saveas(gcf, fullfile(code_folder, "Figures\Scatterplot_AR_over_AF_PATAKA.png"));

figure
hold on
scatter(af_HC(2,:),ar_HC(2,:),"red", "filled");
scatter(af_SLA(2,:), ar_SLA(2,:),"blue","filled");
title("With treshold, PATAKA files")
xlabel("Activation Frequency")
ylabel("Activation Ratio")
legend("HC", "ALS")
hold off

saveas(gcf, fullfile(code_folder, "Figures\Scatterplot_AR_over_AF_PATAKA_w_th.png"));
%% Boxplots of Activation Ratio
figure('Position', [100, 100, 1200, 800])
boxplot([ar_HC(1,:)'; ar_SLA(1,:)'], [zeros(10,1); 1+zeros(8,1)]);
labels = {'HC', 'ALS'};
set(gca, 'XTickLabel', labels);
ylabel('Activation Ratio');
title("Activation Ratio of PATAKA files without threshold");
%signicance test
[p, h, stats] = ranksum(ar_HC(1,:), ar_SLA(1,:));
num_asterisks = '';
if h == 1
    if p < 0.001
        num_asterisks = '***';
    elseif p < 0.01
        num_asterisks = '**';
    elseif p < 0.05
        num_asterisks = '*';
    end
end
if ~isempty(num_asterisks)

    max_y = max([ar_HC(1,:)'; ar_SLA(1,:)']);
    line([1, 2], [max_y + 0.1, max_y + 0.1], 'Color', 'k', 'LineWidth', 0.7);
    line([1, 1], [max_y + 0.9, max_y + 0.1], 'Color', 'k', 'LineWidth', 0.7);
    line([2, 2], [max_y + 0.9, max_y + 0.1], 'Color', 'k', 'LineWidth', 0.7);

    text(1.5, max_y + 0.12, num_asterisks, 'HorizontalAlignment', 'center', 'FontSize', 16);
end
xlim([0, 3]);
ylim([min([ar_HC(1,:)'; ar_SLA(1,:)']) max([ar_HC(1,:)'; ar_SLA(1,:)']) + 0.15]);

saveas(gcf, fullfile(code_folder,"Figures\Activation_Ratio_PATAKA.png"));



figure('Position', [100, 100, 1200, 800])
boxplot([ar_HC(2,:)'; ar_SLA(2,:)'], [zeros(10,1); 1+zeros(8,1)]);
labels = {'HC', 'ALS'};
set(gca, 'XTickLabel', labels);
ylabel('Activation Ratio');
title("Activation Ratio of PATAKA files with threshold");
%signicance test
[p, h, stats] = ranksum(ar_HC(2,:), ar_SLA(2,:));
num_asterisks = '';
if h == 1
    if p < 0.001
        num_asterisks = '***';
    elseif p < 0.01
        num_asterisks = '**';
    elseif p < 0.05
        num_asterisks = '*';
    end
end
if ~isempty(num_asterisks)

    max_y = max([ar_HC(2,:)'; ar_SLA(2,:)']);
    line([1, 2], [max_y + 0.1, max_y + 0.1], 'Color', 'k', 'LineWidth', 0.7);
    line([1, 1], [max_y + 0.9, max_y + 0.1], 'Color', 'k', 'LineWidth', 0.7);
    line([2, 2], [max_y + 0.9, max_y + 0.1], 'Color', 'k', 'LineWidth', 0.7);


    text(1.5, max_y + 0.12, num_asterisks, 'HorizontalAlignment', 'center', 'FontSize', 16);
end
xlim([0, 3]);
ylim([min([ar_HC(2,:)'; ar_SLA(2,:)']) max([ar_HC(2,:)'; ar_SLA(2,:)']) + 0.15]);

saveas(gcf, fullfile(code_folder,"Figures\Activation_Ratio_PATAKA_w_th.png"));
%% Boxplot of Activation Frequency
figure('Position', [100, 100, 1200, 800])
boxplot([af_HC(1,:)'; af_SLA(1,:)'], [zeros(10,1); 1+zeros(8,1)]);
labels = {'HC', 'ALS'};
set(gca, 'XTickLabel', labels);
ylabel('Activation Frequency');
title("Activation Frequency of PATAKA files without threshold");
%signicance test
[p,h,stats] = ranksum(af_HC(1,:), af_SLA(1,:));
num_asterisks = '';
if h == 1
    if p < 0.001
        num_asterisks = '***';
    elseif p < 0.01
        num_asterisks = '**';
    elseif p < 0.05
        num_asterisks = '*';
    end
end
if ~isempty(num_asterisks)

    max_y = max([af_HC(1,:)'; af_SLA(1,:)']);
    line([1, 2], [max_y + 0.5, max_y + 0.5], 'Color', 'k', 'LineWidth', 0.7);
    line([1, 1], [max_y + 0.4, max_y + 0.5], 'Color', 'k', 'LineWidth', 0.7);
    line([2, 2], [max_y + 0.4, max_y + 0.5], 'Color', 'k', 'LineWidth', 0.7);


    text(1.5, max_y + 0.6, num_asterisks, 'HorizontalAlignment', 'center', 'FontSize', 16);
end
xlim([0, 3]);
ylim([min([af_HC(1,:)'; af_SLA(1,:)']) max([af_HC(1,:)'; af_SLA(1,:)']) + 0.8]);% annotation('textbox', [0.8, 0.1, 0.3, 0.1], 'String', ...

saveas(gcf, fullfile(code_folder,"Figures\Activation_Frequency_PATAKA.png"));



figure('Position', [100, 100, 1200, 800])
boxplot([af_HC(2,:)'; af_SLA(2,:)'], [zeros(10,1); 1+zeros(8,1)]);
labels = {'HC', 'ALS'};
set(gca, 'XTickLabel', labels);
ylabel('Activation Frequency');
title("Activation Frequency of PATAKA files with threshold");
%signicance test
[p,h,stats] = ranksum(af_HC(2,:), af_SLA(2,:));
num_asterisks = '';
if h == 1
    if p < 0.001
        num_asterisks = '***';
    elseif p < 0.01
        num_asterisks = '**';
    elseif p < 0.05
        num_asterisks = '*';
    end
end
if ~isempty(num_asterisks)

    max_y = max([af_HC(2,:)'; af_SLA(2,:)']);
    line([1, 2], [max_y + 0.5, max_y + 0.5], 'Color', 'k', 'LineWidth', 0.7);
    line([1, 1], [max_y + 0.4, max_y + 0.5], 'Color', 'k', 'LineWidth', 0.7);
    line([2, 2], [max_y + 0.4, max_y + 0.5], 'Color', 'k', 'LineWidth', 0.7);

    text(1.5, max_y + 0.6, num_asterisks, 'HorizontalAlignment', 'center', 'FontSize', 16);
end
xlim([0, 3]);
ylim([min([af_HC(2,:)'; af_SLA(2,:)']) max([af_HC(2,:)'; af_SLA(2,:)']) + 0.8]);% annotation('textbox', [0.8, 0.1, 0.3, 0.1], 'String', ...

saveas(gcf, fullfile(code_folder,"Figures\Activation_Frequency_PATAKA_w_th.png"));
