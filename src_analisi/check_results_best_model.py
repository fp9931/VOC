import os
import pandas as pd
import numpy as np

general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(general_path, 'Results/First test')

results = pd.read_excel(os.path.join(results_path, "results_classification_sep (f1 based. MFCCs).xlsx"))

scores = results['Target'].unique()
datasets = results['Dataset'].unique()
techniques = results['Technique'].unique()

res = []

for score in scores:
    for dataset in datasets:
        # Seleziona tre righe consecutivi
        subset = results[(results['Target'] == score) & (results['Dataset'] == dataset)]
        for i in range(0, len(subset), 3):

            triplet = subset.iloc[i:i+3]
            best_f1_validation = 0
            best_model = None
            best_technique = None
            best_parameters = None
            best_features_set = None
            best_voted_features = None
            best_f1_train = None
            best_true_values = None
            best_predicted_values = None
            best_tp = None
            best_fp = None
            best_tn = None
            best_fn = None
            best_accuracy = None
            best_f1 = 0
            best_recall = 0
            best_precision = 0
            best_specificity = 0
            best_sensitivity = 0

            # Find max index in the triplet and normalize
            for index, row in triplet.iterrows():
                if row['F1 train'] > best_f1_validation:
                    best_f1_validation = row['F1 train']
                    best_model = row['Model']
                    best_technique = row['Technique']
                    best_parameters = row['Parameters']
                    best_features_set = row['Features set']
                    best_voted_features = row['Voted features']
                    best_f1_train = row['F1 train']
                    best_true_values = row['True values']
                    best_predicted_values = row['Predicted values']
                    best_tp = row['TP']
                    best_fp = row['FP']
                    best_tn = row['TN']
                    best_fn = row['FN']
                    best_accuracy = row['Accuracy']
                    best_f1 = row['F1-score']
                    best_recall = row['Recall']
                    best_precision = row['Precision']
                    best_specificity = row['Specificity']
                    best_sensitivity = row['Sensitivity']

            res.append({
                'Target': score,
                'Dataset': dataset,
                'Model': best_model,
                'Parameters': best_parameters,
                'Features set': best_features_set,
                'Technique': best_technique,
                'Voted features': best_voted_features,
                'F1 validation': best_f1_validation,
                'F1 train': best_f1_train,
                'True values': best_true_values,
                'Predicted values': best_predicted_values,
                'TP': best_tp,
                'FP': best_fp,
                'TN': best_tn,
                'FN': best_fn,
                'Accuracy': best_accuracy,
                'F1-score': best_f1,
                'Recall': best_recall,
                'Precision': best_precision,
                'Specificity': best_specificity,
                'Sensitivity': best_sensitivity,
            })
                
            results_df = pd.DataFrame(res)
            results_df.to_excel(os.path.join(results_path, f"F1_best_model_train.xlsx"), index=False)


results = pd.read_excel(os.path.join(results_path, "results_regression_sep (MFCCs).xlsx"))

scores = results['Target'].unique()
datasets = results['Dataset'].unique()
techniques = results['Technique'].unique()

res = []

for score in scores:
    for dataset in datasets:
        # Seleziona tre righe consecutivi
        subset = results[(results['Target'] == score) & (results['Dataset'] == dataset)]
        for i in range(0, len(subset), 3):
            triplet = subset.iloc[i:i+3]
            best_rmse_validation = np.inf
            best_model = None
            best_parameters = None
            best_features_set = None
            best_technique = None
            best_voted_features = None
            best_rmse_train = None
            best_rmse_test = None
            best_r2_train = None
            best_r2_test = None
            best_true_values = None
            best_predicted_values = None
            best_rmse_train_rounded = None
            best_rmse_test_rounded = None
            best_r2_train_rounded = None
            best_r2_test_rounded = None

            for index, row in triplet.iterrows():
                if row['RMSE train'] < best_rmse_validation:
                    best_rmse_validation = row['RMSE train']
                    best_model = row['Model']
                    best_parameters = row['Parameters']
                    best_features_set = row['Features set']
                    best_technique = row['Technique']
                    best_voted_features = row['Voted features']
                    best_rmse_train = row['RMSE train']
                    best_true_values = row['True values']
                    best_predicted_values = row['Predicted values']
                    best_rmse_train_rounded = row['RMSE train rounded']
                    best_rmse_test_rounded = row['RMSE test rounded']
                    best_r2_train_rounded = row['R2 train rounded']
                    best_r2_test_rounded = row['R2 test rounded']
                    best_rmse_test = row['RMSE test']
                    best_r2_train = row['R2 train']
                    best_r2_test = row['R2 test']

            res.append({
                'Target': score,
                'Dataset': dataset,
                'Model': best_model,
                'Parameters': best_parameters,
                'Features set': best_features_set,
                'Technique': best_technique,
                'Voted features': best_voted_features,
                'RMSE validation': best_rmse_validation,
                'RMSE train': best_rmse_train,
                'True values': best_true_values,
                'Predicted values': best_predicted_values,
                'RMSE test': best_rmse_test,
                'R2 train': best_r2_train,
                'R2 test': best_r2_test,
                'RMSE train rounded': best_rmse_train_rounded,
                'RMSE test rounded': best_rmse_test_rounded,
                'R2 train rounded': best_r2_train_rounded,
                'R2 test rounded': best_r2_test_rounded,
            })

            results_df = pd.DataFrame(res)
            results_df.to_excel(os.path.join(results_path, f"RMSE_best_model_train.xlsx"), index=False)
