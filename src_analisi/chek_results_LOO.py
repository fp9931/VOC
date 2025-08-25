import os
import pandas as pd
import numpy as np

general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(general_path, 'Results/First test/LOO (F1, MFCCs)')

results = pd.read_excel(os.path.join(results_path, "results_classification_sep_LOO.xlsx"))

scores = results['Target'].unique()
datasets = results['Dataset'].unique()
techniques = results['Technique'].unique()

res = []

for score in scores:
    for dataset in datasets:
        # for technique in techniques:
            subset = results[(results['Target'] == score) & (results['Dataset'] == dataset)]
            # subset = results[(results['Target'] == score) & (results['Dataset'] == dataset) & (results['Technique'] == technique)]
            tp = 0
            tn = 0
            fp = 0
            fn = 0

            for i in range(0, len(subset), 3):
                triplet = subset.iloc[i:i+3]
                best_f1_validation = 0
                true_labels = None
                pred_labels = None
                best_technique = None

                 # Find max index in the triplet and normalize
                for index, row in triplet.iterrows():
                    if row['F1 validation'] > best_f1_validation:
                        best_f1_validation = row['F1 validation']
                        true_labels = row['True values']
                        pred_labels = row['Predicted values']
                        best_technique = row['Technique']
                if true_labels == 0 and pred_labels == 0:
                    tp += 1
                if true_labels == 1 and pred_labels == 1:
                    tn += 1
                if true_labels == 0 and pred_labels == 1:
                    fn += 1
                if true_labels == 1 and pred_labels == 0:
                    fp += 1

                print(f"Participant number: {i//3 + 1}")

            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

            res.append({
                'Target': score,
                'Dataset': dataset,
                'Technique': best_technique,
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'Accuracy': accuracy,
                'F1 Score': f1_score,
                'Precision': precision,
                'Recall': recall,
                'Specificity': specificity,
                'Sensitivity': sensitivity
            })
            results_df = pd.DataFrame(res)
            results_df.to_excel(os.path.join(results_path, f"results_aggregate_best_model.xlsx"), index=False)


results = pd.read_excel(os.path.join(results_path, "results_regression_sep_LOO.xlsx"))

scores = results['Target'].unique()
datasets = results['Dataset'].unique()
techniques = results['Technique'].unique()

res = []
for score in scores:
    for dataset in datasets:
        # for technique in techniques:
            subset = results[(results['Target'] == score) & (results['Dataset'] == dataset)]
            true_labels = []
            pred_labels = []

            for i in range(0, len(subset), 3):
                triplet = subset.iloc[i:i+3]
                best_rmse_validation = 0
                best_technique = None

                 # Find max index in the triplet and normalize
                for index, row in triplet.iterrows():
                    if row['RMSE validation'] > best_rmse_validation:
                        best_rmse_validation = row['RMSE validation']
                        true_labels.append(row['True values'])
                        pred_labels.append(row['Predicted values'])
                        best_technique = row['Technique']
                        pred_values_rounded = np.round(pred_labels)

            mae = np.mean([abs(t - p) for t, p in zip(true_labels, pred_labels)])
            mse = np.mean([(t - p) ** 2 for t, p in zip(true_labels, pred_labels)])
            rmse = np.sqrt(mse)

            mae_rounded = np.mean([abs(t - p) for t, p in zip(true_labels, pred_values_rounded)])
            mse_rounded = np.mean([(t - p) ** 2 for t, p in zip(true_labels, pred_values_rounded)])
            rmse_rounded = np.sqrt(mse_rounded)

            res.append({
                'Target': score,
                'Dataset': dataset,
                'Technique': best_technique,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                #'R2': r2,
                'MAE Rounded': mae_rounded,
                'MSE Rounded': mse_rounded,
                'RMSE Rounded': rmse_rounded,
                #'R2 Rounded': r2_rounded
            })
            results_df = pd.DataFrame(res)
            results_df.to_excel(os.path.join(results_path, f"results_aggregate_regression_best_model.xlsx"), index=False)
