import os
import sys
import numpy as np
import pandas as pd
import re
import ast

from mrmr import mrmr_classif
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

from collections import Counter
import shap
from matplotlib import pyplot as plt

general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
features_path = os.path.join(general_path, 'Features')
results_path = os.path.join(general_path, 'Results')

# Load the cleaned dataframes
df = pd.read_excel(os.path.join(features_path, 'complete_clean.xlsx'))
results_dataset = pd.read_excel(os.path.join(results_path, 'MRMR', 'Swallowing', 'MFCCs', 'swallowing.xlsx'))


# Filter ALS patients and drop unnecessary columns
als_df_complete = df[df['category'] == 'ALS']
y = als_df_complete['ALSFRS-R_SwallowingSubscore'].values
id = als_df_complete['subjid'].values

columns_to_drop = ['subjid', 'category', 'sex', 'ALSFRS-R_SpeechSubscore', 'ALSFRS-R_SwallowingSubscore', 'PUMNS_BulbarSubscore', 'SML11_t', 'SML12_t', 'SML13_t', 'SML21_t', 'SML22_t', 'SML23_t', 'SML31_t', 'SML32_t', 'SML33_t', 'SML41_t', 'SML42_t', 'SML43_t', 'x2D_DCT1_t', 'x2D_DCT2_t', 'x2D_DCT3_t', 'x2D_DCT4_t', 'x2D_DCT5_t', 'x2D_DCT6_t', 'x2D_DCT7_t', 'x2D_DCT8_t', 'x2D_DCT9_t',
                        'SML11_k', 'SML12_k', 'SML13_k', 'SML21_k', 'SML22_k', 'SML23_k', 'SML31_k', 'SML32_k', 'SML33_k', 'SML41_k', 'SML42_k', 'SML43_k', 'x2D_DCT1_k', 'x2D_DCT2_k', 'x2D_DCT3_k', 'x2D_DCT4_k', 'x2D_DCT5_k', 'x2D_DCT6_k', 'x2D_DCT7_k', 'x2D_DCT8_k', 'x2D_DCT9_k',
                        'SML11_p', 'SML12_p', 'SML13_p', 'SML21_p', 'SML22_p', 'SML23_p', 'SML31_p', 'SML32_p', 'SML33_p', 'SML41_p', 'SML42_p', 'SML43_p', 'x2D_DCT1_p', 'x2D_DCT2_p', 'x2D_DCT3_p', 'x2D_DCT4_p', 'x2D_DCT5_p', 'x2D_DCT6_p', 'x2D_DCT7_p', 'x2D_DCT8_p', 'x2D_DCT9_p']
# columns_to_drop = ['subjid', 'category', 'sex', 'ALSFRS-R_SpeechSubscore', 'ALSFRS-R_SwallowingSubscore', 'PUMNS_BulbarSubscore']

als_df = als_df_complete.drop(columns=columns_to_drop)

# Ordinare i risultati in base a f1_train
results_dataset = results_dataset.sort_values(by='F1 train', ascending=False)
# Take only the first row
results_dataset = results_dataset.iloc[0:3, :]

X_df_tot = als_df.copy()
y = np.where(y == 4, 0, 1)  # 0 normal, 1 impaired

metrics = {
    # 'Feature Selection': [],
    'Model': [],
    'Technique': [],
    'F1 train': [],
    'F1 test': [],
    'Features set': [],
    'Parameters': [],
    'Mean F1-score': [],
    'Std F1-score': [],
    'Mean Accuracy': [],
    'Std Accuracy': [],
    'Mean Recall': [],
    'Std Recall': [],
    'Mean Precision': [],
    'Std Precision': [],
    'Mean Specificity': [],
    'Std Specificity': [],
    'Mean Sensitivity': [],
    'Std Sensitivity': []
}

for i in range(len(results_dataset)):

    results = {
        'TN': [],
        'FP': [],
        'FN': [],
        'TP': [],
        'Accuracy': [],
        'F1-score': [],
        'Recall': [],
        'Precision': [],
        'Specificity': [],
        'Sensitivity': []
    }

    parameters = results_dataset['Parameters'].values[i]
    parameters = re.sub(r'np\.float64\((.*?)\)', r'\1', parameters)
    parameters = ast.literal_eval(parameters)

    features = results_dataset['Features set'].values[0]
    features = re.sub(r'np\.float64\((.*?)\)', r'\1', features)
    features = ast.literal_eval(features)

    X_df = X_df_tot[features]
    X = X_df.values

    shap_matrix_values = np.zeros((X_df_tot.shape[0], X_df_tot.shape[1]))
    shap_matrix = pd.DataFrame(shap_matrix_values, columns=X_df_tot.columns)
    test_idx_all = []

    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(cross_validation.split(X, y)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Impute missing values
        imputer = IterativeImputer(max_iter=10, random_state=42)
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if results_dataset['Model'].values[i] == 'XGB':
            y_train_positive = len(np.where(y[train_index] == 1)[0])
            y_train_negative = len(np.where(y[train_index] == 0)[0])
            scale_pos_weight = y_train_negative / y_train_positive if y_train_positive > 0 else 1
            model = XGBClassifier(random_state=42, n_jobs=-1)
            model.set_params(scale_pos_weight=scale_pos_weight)

        elif results_dataset['Model'].values[i] == 'SVM':
            model = SVC(class_weight='balanced')

        elif results_dataset['Model'].values[i] == 'RF':
            model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

        elif results_dataset['Model'].values[i] == 'KNN':
            model = KNeighborsClassifier(n_jobs=-1)

        else:
            model = MLPClassifier(random_state=42, max_iter=1000, early_stopping=True, n_iter_no_change=10)

        model.set_params(**parameters)
        model.fit(X_train, y_train)

        # Calcola SHAP sul test set
        explainer = shap.Explainer(model.predict, X_train)
        shap_vals = explainer(X_test)

        shap_matrix.loc[test_index, features] = shap_vals.values  # Assuming binary classification, we take the second class's SHAP values

        for j in range(len(test_index)):
            test_idx_all.append(test_index[j])

        # Evaluate on the test set
        test_predictions = model.predict(X_test)
        test_f1 = f1_score(y_test, test_predictions)
        tn, fp, fn, tp = confusion_matrix(y_test, test_predictions).ravel()
        accuracy = accuracy_score(y_test, test_predictions)
        precision = precision_score(y_test, test_predictions)
        recall = recall_score(y_test, test_predictions)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        results['TN'].append(tn)
        results['FP'].append(fp)
        results['FN'].append(fn)
        results['TP'].append(tp)
        results['Accuracy'].append(accuracy)
        results['F1-score'].append(test_f1)
        results['Recall'].append(recall)
        results['Precision'].append(precision)
        results['Specificity'].append(specificity)
        results['Sensitivity'].append(sensitivity)
    
    # metrics['Feature Selection'].append(results_dataset['Feature Selection'].values[i])
    metrics['Model'].append(results_dataset['Model'].values[i])
    metrics['Technique'].append(results_dataset['Technique'].values[i])
    metrics['F1 train'].append(results_dataset['F1 train'].values[i])
    metrics['F1 test'].append(results_dataset['F1-score'].values[i])
    metrics['Features set'].append(results_dataset['Features set'].values[i])
    metrics['Parameters'].append(results_dataset['Parameters'].values[i])
    metrics['Mean F1-score'].append(f"{np.mean(results['F1-score']):.2f}")
    metrics['Std F1-score'].append(f"{np.std(results['F1-score']):.2f}")
    metrics['Mean Accuracy'].append(f"{np.mean(results['Accuracy']):.2f}")
    metrics['Std Accuracy'].append(f"{np.std(results['Accuracy']):.2f}")
    metrics['Mean Recall'].append(f"{np.mean(results['Recall']):.2f}")
    metrics['Std Recall'].append(f"{np.std(results['Recall']):.2f}")
    metrics['Mean Precision'].append(f"{np.mean(results['Precision']):.2f}")
    metrics['Std Precision'].append(f"{np.std(results['Precision']):.2f}")
    metrics['Mean Specificity'].append(f"{np.mean(results['Specificity']):.2f}")
    metrics['Std Specificity'].append(f"{np.std(results['Specificity']):.2f}")
    metrics['Mean Sensitivity'].append(f"{np.mean(results['Sensitivity']):.2f}")
    metrics['Std Sensitivity'].append(f"{np.std(results['Sensitivity']):.2f}")

    # # Convert metrics to DataFrame
    # metrics_df = pd.DataFrame(metrics)
    # metrics_df.to_excel(os.path.join(results_path, 'MRMR', 'Swallowing', 'MFCCs', 'cross_validation_MRMR.xlsx'), index=False)

    shap_matrix_df = shap_matrix.copy()
    shap_matrix_df.to_excel(os.path.join(results_path, 'MRMR', 'Swallowing', 'MFCCs', f"Swallowing_{results_dataset['Model'].values[i]}.xlsx"), index=False)
    shap.summary_plot(shap_matrix_df.values, features=X_df_tot.values, feature_names=shap_matrix_df.columns, plot_type="bar", color='#f5054f',max_display=10, show=False)
    plt.savefig(os.path.join(results_path, 'MRMR', 'Swallowing', 'MFCCs', f"Swallowing_{results_dataset['Model'].values[i]}.png"), bbox_inches='tight')
    shap.summary_plot(shap_matrix_df.values, features=X_df_tot.values, feature_names=shap_matrix_df.columns, class_names=['0', '1'], class_inds=[0,1], plot_type="violin", max_display=10, show=False)
    plt.savefig(os.path.join(results_path, 'MRMR', 'Swallowing', 'MFCCs', f"Swallowing_violin_{results_dataset['Model'].values[i]}.png"), bbox_inches='tight', dpi=300)
    break
