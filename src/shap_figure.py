import os
import pandas as pd
import numpy as np
import re
import ast
import shap
from matplotlib import pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import random

seed = 42

def shap_analysis(name_results):

    random.seed(seed)
    np.random.seed(seed)

    general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(general_path, 'Results')
    features_path = os.path.join(general_path, 'Features')
    results = pd.read_excel(os.path.join(results_path, name_results))

    scores = results['Target'].unique()
    datasets = results['Dataset'].unique()
    techniques = results['Technique'].unique()
    scores_saved = results['Target'].unique()

    res = []

    count = 0
    for s, score in enumerate(scores):

        if isinstance(score, str) and score.startswith("["):
            score = ast.literal_eval(score)[0]

        for dataset in datasets:
            # Seleziona tre righe consecutivi
            subset = results[(results['Target'] == scores_saved[s]) & (results['Dataset'] == dataset)]
            data = pd.read_excel(os.path.join(features_path, f"Features_{dataset}.xlsx"))
            als_data = data

            y = als_data[score].values
            y = np.where(y == 4, 0, 1) # 0 normal, 1 impaired

            columns_to_drop = ['ID', 'ALSFRS-R_SpeechSubscore', 'ALSFRS-R_SwallowingSubscore', 'PUMNS_BulbarSubscore']
            als_data = als_data.drop(columns=columns_to_drop)
            X_df = als_data.copy()
            X = X_df.values
            shap_matrix_values = np.zeros((X_df.shape[0], X_df.shape[1]))
            shap_matrix = pd.DataFrame(shap_matrix_values, columns=X_df.columns)
            test_idx_all = []

            for i in range(0, len(subset), 3):

                triplet = subset.iloc[i:i+3]
                best_f1_validation = 0
                best_model = None
                best_technique = None
                best_parameters = None
                best_features_set = None
                best_idx_test = None
                best_f1_train = 0
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
                best_n_features = np.inf

                # Find max index in the triplet and normalize
                i = 0
                for index, row in triplet.iterrows():

                    features = triplet['Voted features'].values[i]
                    i += 1
                    features = re.sub(r'np\.float64\((.*?)\)', r'\1', features)
                    features = ast.literal_eval(features)

                    n_features = len(features)

                    if row['F1 validation'] >= best_f1_validation:
                        if n_features >= best_n_features:
                            continue
                        best_f1_validation = row['F1 validation']
                        best_n_features = n_features
                        best_model = row['Model']
                        best_technique = row['Technique']
                        best_parameters = row['Parameters']
                        best_features_set = row['Features set']
                        best_idx_test = row['idx test']
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
                    'idx test': best_idx_test,
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
                results_df.to_excel(os.path.join(results_path, f"Best_{name_results}"), index=False)
                '''
                # Plot shap
                parameters = results_df['Parameters'].values[count]
                parameters = re.sub(r'np\.float64\((.*?)\)', r'\1', parameters)
                parameters = ast.literal_eval(parameters)

                features = results_df['Features set'].values[count]
                features = re.sub(r'np\.float64\((.*?)\)', r'\1', features)
                features = ast.literal_eval(features)

                idx_test = results_df['idx test'].values[count]
                idx_test = [int(x) for x in idx_test.strip("[]").split()]

                X_test = als_data.iloc[idx_test][features]
                y_test = y[idx_test]

                mask = als_data.index.isin(idx_test)
                X_train = als_data[~mask][features]
                y_train = y[~mask]

                # Impute missing values
                imputer = IterativeImputer(max_iter=10, random_state=42)
                X_train = imputer.fit_transform(X_train)
                X_test = imputer.transform(X_test)

                # Standardize features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                if results_df['Model'].values[count] == 'XGB':
                    y_train_positive = len(np.where(y_train == 1)[0])
                    y_train_negative = len(np.where(y_train == 0)[0])
                    scale_pos_weight = y_train_negative / y_train_positive if y_train_positive > 0 else 1
                    model = XGBClassifier(random_state=42, n_jobs=-1)
                    model.set_params(scale_pos_weight=scale_pos_weight)

                elif results_df['Model'].values[count] == 'SVM':
                    model = SVC(class_weight='balanced')

                elif results_df['Model'].values[count] == 'RF':
                    model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

                elif results_df['Model'].values[count] == 'KNN':
                    model = KNeighborsClassifier(n_jobs=-1)

                else:
                    model = MLPClassifier(random_state=42, max_iter=1000, early_stopping=True, n_iter_no_change=10)
            
                model.set_params(**parameters)
                model.fit(X_train, y_train)
                # Calcola SHAP sul test set
                explainer = shap.Explainer(model.predict, X_train)
                shap_vals = explainer(X_test).values

                for j, col in enumerate(features):
                    for i, idx in enumerate(idx_test):
                        shap_matrix.loc[idx,col] = shap_vals[i, j]

                for j in range(len(idx_test)):
                    test_idx_all.append(idx_test[j])

                count += 1

            # reindexing using test_idx_all
            shap_matrix_df = pd.DataFrame(shap_matrix)
            shap_matrix_df.reindex(index=test_idx_all)
            shap_matrix_df.to_excel(os.path.join(results_path, f"shap_{score}_{dataset}.xlsx"), index=False)
            shap.summary_plot(shap_matrix_df.values, features=X_df.values, feature_names=shap_matrix_df.columns, plot_type="bar", color='#f5054f',max_display=10, show=False)
            plt.savefig(os.path.join(results_path, f"{score}_{dataset}.png"), bbox_inches='tight')
            shap.summary_plot(shap_matrix_df.values, features=X_df.values, feature_names=shap_matrix_df.columns, class_names=['Normal', 'Impaired'], class_inds=[0,1], plot_type="violin", max_display=10, show=False)
            plt.savefig(os.path.join(results_path, f"{score}_{dataset}_violin.png"), bbox_inches='tight', dpi=300)
            '''

# New MFCCs
if __name__ == "__main__":
    shap_analysis(name_results="Old_Age.xlsx")