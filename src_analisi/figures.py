import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

'''
########################################### REGRESSION ##############################################################
general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(general_path, 'Results/First test')
results_LOO_path = os.path.join(results_path, 'LOO (F1, MFCCs)')

df_sep = pd.read_excel(os.path.join(results_path, 'results_regression_sep (MFCCs).xlsx'))
df_loo = pd.read_excel(os.path.join(results_LOO_path, 'results_regression_LOO.xlsx'))

# Grafico per sep
# Dataset complete
datasets = df_sep['Dataset'].unique()
technique = df_sep['Technique'].unique()

for dataset in datasets:
    for tech in technique:
        current_df = df_sep[(df_sep['Dataset'] == dataset) & (df_sep['Technique'] == tech)].reset_index(drop=True)

        true_labels = []
        pred_labels = []
        for s in current_df['True values']:
            # remove brackets, split by spaces, convert to int
            nums = list(map(int, s.strip("[]").split()))
            true_labels.extend(nums)

        for s in current_df['Predicted values']:
            nums = list(map(float, s.strip("[]").split()))
            pred_labels.extend(nums)
        
        print(true_labels)
            
        round_pred = [round(pred, 0) for pred in pred_labels]
        # compute RMSE
        rmse = np.sqrt(np.mean((np.array(true_labels) - np.array(round_pred)) ** 2))

        # Count how many true 0 are predicted as 1, how many as 2, how many as 3, how many as 4
        wrong_0_as_1 = sum(1 for t, p in zip(true_labels, round_pred) if t == 0 and p == 1)
        wrong_0_as_2 = sum(1 for t, p in zip(true_labels, round_pred) if t == 0 and p == 2)
        wrong_0_as_3 = sum(1 for t, p in zip(true_labels, round_pred) if t == 0 and p == 3)
        wrong_0_as_4 = sum(1 for t, p in zip(true_labels, round_pred) if t == 0 and p == 4)

        wrong_1_as_0 = sum(1 for t, p in zip(true_labels, round_pred) if t == 1 and p == 0)
        wrong_1_as_2 = sum(1 for t, p in zip(true_labels, round_pred) if t == 1 and p == 2)
        wrong_1_as_3 = sum(1 for t, p in zip(true_labels, round_pred) if t == 1 and p == 3)
        wrong_1_as_4 = sum(1 for t, p in zip(true_labels, round_pred) if t == 1 and p == 4)

        wrong_2_as_0 = sum(1 for t, p in zip(true_labels, round_pred) if t == 2 and p == 0)
        wrong_2_as_1 = sum(1 for t, p in zip(true_labels, round_pred) if t == 2 and p == 1)
        wrong_2_as_3 = sum(1 for t, p in zip(true_labels, round_pred) if t == 2 and p == 3)
        wrong_2_as_4 = sum(1 for t, p in zip(true_labels, round_pred) if t == 2 and p == 4)

        wrong_3_as_0 = sum(1 for t, p in zip(true_labels, round_pred) if t == 3 and p == 0)
        wrong_3_as_1 = sum(1 for t, p in zip(true_labels, round_pred) if t == 3 and p == 1)
        wrong_3_as_2 = sum(1 for t, p in zip(true_labels, round_pred) if t == 3 and p == 2)
        wrong_3_as_4 = sum(1 for t, p in zip(true_labels, round_pred) if t == 3 and p == 4)

        wrong_4_as_0 = sum(1 for t, p in zip(true_labels, round_pred) if t == 4 and p == 0)
        wrong_4_as_1 = sum(1 for t, p in zip(true_labels, round_pred) if t == 4 and p == 1)
        wrong_4_as_2 = sum(1 for t, p in zip(true_labels, round_pred) if t == 4 and p == 2)
        wrong_4_as_3 = sum(1 for t, p in zip(true_labels, round_pred) if t == 4 and p == 3)

        print(f"For dataset {dataset} and technique {tech}:")
        print(f"0 predicted as 1: {wrong_0_as_1}, as 2: {wrong_0_as_2}, as 3: {wrong_0_as_3}, as 4: {wrong_0_as_4}")
        print(f"1 predicted as 0: {wrong_1_as_0}, as 2: {wrong_1_as_2}, as 3: {wrong_1_as_3}, as 4: {wrong_1_as_4}")
        print(f"2 predicted as 0: {wrong_2_as_0}, as 1: {wrong_2_as_1}, as 3: {wrong_2_as_3}, as 4: {wrong_2_as_4}")
        print(f"3 predicted as 0: {wrong_3_as_0}, as 1: {wrong_3_as_1}, as 2: {wrong_3_as_2}, as 4: {wrong_3_as_4}")
        print(f"4 predicted as 0: {wrong_4_as_0}, as 1: {wrong_4_as_1}, as 2: {wrong_4_as_2}, as 3: {wrong_4_as_3}")

        # Plot the perfect prediction line in green and --
        # The only values that can be estimated are 0, 1, 2, 3, 4
        plt.figure()
        plt.scatter(true_labels, round_pred, color='blue', alpha=0.5, label='Predictions')       
        plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], color='green', linestyle='--', label='Perfect prediction')
        plt.xlabel('True values')
        plt.ylabel('Predicted values')
        plt.title(f'Scatter plot for {dataset} - {tech}, RMSE: {rmse}')
        plt.legend()
        plt.grid()
        plt.show()


# Grafico per sep
# Dataset complete
datasets = df_loo['Dataset'].unique()
technique = df_loo['Technique'].unique()

for dataset in datasets:
    for tech in ['5', '10%']:
        current_df = df_loo[(df_loo['Dataset'] == dataset) & (df_loo['Technique'] == tech)].reset_index(drop=True)

        true_labels = []
        pred_labels = []
        for i in range(len(current_df)):
            true_labels.append(int(current_df['True values'][i]))
            pred_labels.append(float(current_df['Predicted values'][i]))

        # print(len(true_labels))

        round_pred = [round(pred, 0) for pred in pred_labels]
        # compute RMSE
        rmse = np.sqrt(np.mean((np.array(true_labels) - np.array(round_pred)) ** 2))

        # Count how many true 0 are predicted as 1, how many as 2, how many as 3, how many as 4
        wrong_0_as_1 = sum(1 for t, p in zip(true_labels, round_pred) if t == 0 and p == 1)
        wrong_0_as_2 = sum(1 for t, p in zip(true_labels, round_pred) if t == 0 and p == 2)
        wrong_0_as_3 = sum(1 for t, p in zip(true_labels, round_pred) if t == 0 and p == 3)
        wrong_0_as_4 = sum(1 for t, p in zip(true_labels, round_pred) if t == 0 and p == 4)

        wrong_1_as_0 = sum(1 for t, p in zip(true_labels, round_pred) if t == 1 and p == 0)
        wrong_1_as_2 = sum(1 for t, p in zip(true_labels, round_pred) if t == 1 and p == 2)
        wrong_1_as_3 = sum(1 for t, p in zip(true_labels, round_pred) if t == 1 and p == 3)
        wrong_1_as_4 = sum(1 for t, p in zip(true_labels, round_pred) if t == 1 and p == 4)

        wrong_2_as_0 = sum(1 for t, p in zip(true_labels, round_pred) if t == 2 and p == 0)
        wrong_2_as_1 = sum(1 for t, p in zip(true_labels, round_pred) if t == 2 and p == 1)
        wrong_2_as_3 = sum(1 for t, p in zip(true_labels, round_pred) if t == 2 and p == 3)
        wrong_2_as_4 = sum(1 for t, p in zip(true_labels, round_pred) if t == 2 and p == 4)

        wrong_3_as_0 = sum(1 for t, p in zip(true_labels, round_pred) if t == 3 and p == 0)
        wrong_3_as_1 = sum(1 for t, p in zip(true_labels, round_pred) if t == 3 and p == 1)
        wrong_3_as_2 = sum(1 for t, p in zip(true_labels, round_pred) if t == 3 and p == 2)
        wrong_3_as_4 = sum(1 for t, p in zip(true_labels, round_pred) if t == 3 and p == 4)

        wrong_4_as_0 = sum(1 for t, p in zip(true_labels, round_pred) if t == 4 and p == 0)
        wrong_4_as_1 = sum(1 for t, p in zip(true_labels, round_pred) if t == 4 and p == 1)
        wrong_4_as_2 = sum(1 for t, p in zip(true_labels, round_pred) if t == 4 and p == 2)
        wrong_4_as_3 = sum(1 for t, p in zip(true_labels, round_pred) if t == 4 and p == 3)

        print(f"For dataset {dataset} and technique {tech}:")
        print(f"0 predicted as 1: {wrong_0_as_1}, as 2: {wrong_0_as_2}, as 3: {wrong_0_as_3}, as 4: {wrong_0_as_4}")
        print(f"1 predicted as 0: {wrong_1_as_0}, as 2: {wrong_1_as_2}, as 3: {wrong_1_as_3}, as 4: {wrong_1_as_4}")
        print(f"2 predicted as 0: {wrong_2_as_0}, as 1: {wrong_2_as_1}, as 3: {wrong_2_as_3}, as 4: {wrong_2_as_4}")
        print(f"3 predicted as 0: {wrong_3_as_0}, as 1: {wrong_3_as_1}, as 2: {wrong_3_as_2}, as 4: {wrong_3_as_4}")
        print(f"4 predicted as 0: {wrong_4_as_0}, as 1: {wrong_4_as_1}, as 2: {wrong_4_as_2}, as 3: {wrong_4_as_3}")

        # Plot the perfect prediction line in green and --
        # The only values that can be estimated are 0, 1, 2, 3, 4
        plt.figure()
        plt.scatter(true_labels, round_pred, color='blue', alpha=0.5, label='Predictions')       
        plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], color='green', linestyle='--', label='Perfect prediction')
        plt.xlabel('True values')
        plt.ylabel('Predicted values')
        plt.title(f'LOO Scatter plot for {dataset} - {tech}, RMSE: {rmse}')
        plt.legend()
        plt.grid()
        plt.show()
'''

########################################### CLASSIFICATION ##############################################################
general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(general_path, 'Results/First test')
results_LOO_path = os.path.join(results_path, 'LOO (F1, MFCCs)')

df_sep = pd.read_excel(os.path.join(results_path, 'results_classification_sep (f1 based. MFCCs).xlsx'))
df_loo = pd.read_excel(os.path.join(results_LOO_path, 'results_classification_LOO.xlsx'))

# Grafico per sep
# Dataset complete
target = df_sep['Target'].unique()
datasets = df_sep['Dataset'].unique()
technique = df_sep['Technique'].unique()

for t in target:
    for dataset in datasets:
        for tech in technique:
            current_df = df_loo[(df_loo['Dataset'] == dataset) & (df_loo['Technique'] == tech) & (df_loo['Target'] == t)].reset_index(drop=True)
            print(current_df)

            true_label = []
            pred_label = []
            for i in range(len(current_df)):
                true_label.append(current_df['True values'][i])
                pred_label.append(current_df['Predicted values'][i])

            # True positive quanti true_label =0 e pred_label=0 contemporaneamente
            tp = sum(1 for t, p in zip(true_label, pred_label) if t == 0 and p == 0)
            tn = sum(1 for t, p in zip(true_label, pred_label) if t == 1 and p == 1)
            fp = sum(1 for t, p in zip(true_label, pred_label) if t == 1 and p == 0)
            fn = sum(1 for t, p in zip(true_label, pred_label) if t == 0 and p == 1)

            print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
            # compute metrics (precision, recall, F1-score)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

            # Plot confusion matrix
            cm = np.array([[tn, fp],
                        [fn, tp]])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Normal','Predicted Impaired'],
                        yticklabels=['True Normal', 'True Impaired'])
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title(f'LOO {t} {dataset} - {tech}\n F1: {f1_score:.2f}, Accuracy: {accuracy:.2f}')
            plt.show()

# for t in target:
#     for dataset in datasets:
#         for tech in technique:
#             current_df = df_sep[(df_sep['Dataset'] == dataset) & (df_sep['Technique'] == tech) & (df_sep['Target'] == t)].reset_index(drop=True)

#             tp = 0
#             tn = 0
#             fp = 0
#             fn = 0
#             for i in range(len(current_df)):
#                 tp += int(current_df['TP'][i])
#                 tn += int(current_df['TN'][i])
#                 fp += int(current_df['FP'][i])
#                 fn += int(current_df['FN'][i])

#             print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
#             # compute metrics (precision, recall, F1-score)
#             precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#             f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#             accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

#             # Plot confusion matrix
#             cm = np.array([[tn, fp],
#                         [fn, tp]])
#             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Normal','Predicted Impaired'],
#                         yticklabels=['True Normal', 'True Impaired'])
#             plt.xlabel('Predicted labels')
#             plt.ylabel('True labels')
#             plt.title(f'{t} {dataset} - {tech}\n F1: {f1_score:.2f}, Accuracy: {accuracy:.2f}')
#             plt.show()