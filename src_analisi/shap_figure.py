import os
import pandas as pd
import numpy as np
import re
import ast
import shap
from matplotlib import pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import LogisticRegression

import random

random.random(seed=42)
np.random.seed(42)

general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(general_path, 'Results\Final results')
features_path = os.path.join(general_path, 'Features\Old')
results = pd.read_excel(os.path.join(results_path, "SEP 1.xlsx"))

scores = results['Target'].unique()
datasets = results['Dataset'].unique()
techniques = results['Technique'].unique()
scores_saved = results['Target'].unique()

res = []


idx_speech = ([9,14,18,20,21,27,28,46,50,59,62,65,66,69,75,76,78,82,85,92],
                      [8,11,19,22,32,36,38,41,42,48,49,52,53,63,72,80,87,89,94,96],
                      [0, 1, 3, 5, 15, 17, 26, 35, 37, 39, 54, 55, 70, 79, 83, 84, 88, 91, 97, 99],
                      [6, 7, 13, 23, 25, 31, 33, 34, 44, 47, 56, 58, 60, 61, 64, 71, 74, 77, 81, 98],
                      [2, 4, 10, 12, 16, 24, 29, 30, 40, 43, 45, 51, 57, 67, 68, 73, 86, 90, 93, 95])

idx_swallowing = ([6, 8, 11, 12, 14, 22, 23, 34, 47, 52, 57, 63, 65, 72, 78, 80, 85, 89, 92, 99],
                           [7,24, 25, 27, 30, 32, 33, 36, 39, 41, 48, 49, 54, 59, 60, 71, 83, 84, 94, 95],
                           [1, 2, 9, 19, 21, 26, 31, 35, 40, 43, 44, 61, 70, 75, 82, 86, 87, 88, 91, 98],
                           [4, 10, 15, 16, 17, 18, 37, 38, 50, 51, 53, 55, 62, 64, 69, 74, 76, 77, 79, 96],
                           [0, 3, 5, 13, 20, 28, 29, 42, 45, 46, 56, 58, 66, 67, 68, 73, 81, 90, 93, 97])

count = 0
for s, score in enumerate(scores):

    if isinstance(score, str) and score.startswith("["):
        score = ast.literal_eval(score)[0]

    for dataset in datasets:
        # Seleziona tre righe consecutivi
        subset = results[(results['Target'] == scores_saved[s]) & (results['Dataset'] == dataset)]
        data = pd.read_excel(os.path.join(features_path, f"{dataset}_clean.xlsx"))
        als_data = data[data['category'] == 'ALS']

        y = als_data[score].values
        y = np.where(y == 4, 0, 1) # 0 normal, 1 impaired

        columns_to_drop = ['subjid', 'category', 'sex', 'ALSFRS-R_SpeechSubscore', 'ALSFRS-R_SwallowingSubscore', 'PUMNS_BulbarSubscore']
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
            # best_voted_features = None
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

            # Find max index in the triplet and normalize
            for index, row in triplet.iterrows():
                if row['F1 validation'] > best_f1_validation:
                    best_f1_validation = row['F1 validation']
                    best_model = row['Model']
                    best_technique = row['Technique']
                    best_parameters = row['Parameters']
                    best_features_set = row['Features set']
                    # best_voted_features = row['Voted features']
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
                # 'Voted features': best_voted_features,
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
            results_df.to_excel(os.path.join(results_path, f"SEP final.xlsx"), index=False)
        
            # Plot shap
            parameters = results_df['Parameters'].values[count]
            parameters = re.sub(r'np\.float64\((.*?)\)', r'\1', parameters)
            parameters = ast.literal_eval(parameters)

            features = results_df['Features set'].values[count]
            features = re.sub(r'np\.float64\((.*?)\)', r'\1', features)
            features = ast.literal_eval(features)

            if score == 'ALSFRS-R_SpeechSubscore':
                idx_test = idx_speech[count % 5]
            else:
                idx_test = idx_swallowing[count % 5]

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
            shap_vals = explainer(X_test)

            shap_matrix.loc[idx_test, features] = shap_vals.values

            for j in range(len(idx_test)):
                test_idx_all.append(idx_test[j])

            shap_matrix.loc[idx_test, features] = shap_vals.values

            for j in range(len(idx_test)):
                test_idx_all.append(idx_test[j])

            count += 1

        # reindexing using test_idx_all
        shap.summary_plot(shap_matrix.values, features=X_df.values, feature_names=shap_matrix.columns, plot_type="bar", color='#f5054f',max_display=10, show=False)
        plt.savefig(os.path.join(results_path, f"{score}_{dataset}.png"), bbox_inches='tight')
        shap.summary_plot(shap_matrix.values, features=X_df.values, feature_names=shap_matrix.columns, class_names=['Normal', 'Impaired'], class_inds=[0,1], plot_type="violin", max_display=10, show=False)
        plt.savefig(os.path.join(results_path, f"{score}_{dataset}_violin.png"), bbox_inches='tight', dpi=300)
