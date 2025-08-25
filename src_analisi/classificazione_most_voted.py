import os
import numpy as np
import pandas as pd
import itertools
from collections import Counter

from mrmr import mrmr_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, accuracy_score, root_mean_squared_error, root_mean_squared_log_error, r2_score, roc_auc_score
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import LogisticRegression, LinearRegression


import re
import ast

def main_classification(df, y, score, name_dataset):

    global results_classification, df_results

    X_df = df.copy()
    X = X_df.values
    y = np.where(y == 4, 0, 1) # 0 normal, 1 impaired

    out_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(out_cv.split(X, y)):

        # Seleziona solo le righe nel results_df che nella colonna 'Target' hanno il valore uguale a score, dove target è ['ALSFRS-R_SpeechSubscore']
        df_results_filtered = df_results[df_results['Target'] == f"['{score}']"]
        print(len(df_results_filtered))
        # Estrai solo le righe con dataset uguale a name_dataset
        df_results_filtered = df_results_filtered[df_results_filtered['Dataset'] == name_dataset]
        print(len(df_results_filtered))

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        print(f"Fold {fold_idx}:")
        # Verify proportions in train and test sets
        count_normal_train = (y_train == 0).sum()
        count_impaired_train = (y_train == 1).sum()
        count_normal_test = (y_test == 0).sum()
        count_impaired_test = (y_test == 1).sum()
        print(f"Train set: {count_normal_train} normal, {count_impaired_train} impaired")
        print(f"Test set: {count_normal_test} normal, {count_impaired_test} impaired")

        #Impute missing values
        imputer = IterativeImputer(max_iter=10, random_state=42)
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Initial feature selection (exploit the controlled randomness of the dataset)
        features_technique = ['5', '10%', 'Free']
        for technique in features_technique:

            # Estrai le righe del dataframe df_results_filtered che nella colonna 'Technique' hanno il valore uguale a technique
            df_results_filtered_tech = df_results_filtered[df_results_filtered['Technique'] == technique]
            # Estrai solo la riga per fold_idx (è un indice, non un valore della colonna)
            df_results_filtered_fold = df_results_filtered_tech.iloc[fold_idx:fold_idx+1]

            # Feature selected
            selected_features_list = df_results_filtered_fold['Voted features'].values[0]
            selected_features = re.sub(r'np\.int64\((.*?)\)', r'\1', selected_features_list)
            selected_features = ast.literal_eval(selected_features)

            # Parameters
            best_params_list = df_results_filtered_fold['Parameters'].values[0]
            best_params_list = re.sub(r'np\.int64\((.*?)\)', r'\1', best_params_list)
            best_params = ast.literal_eval(best_params_list)
            # Find the indeces of the selected features in the original columns
            selected_features_indices = [X_df.columns.get_loc(col) for col in selected_features if col in X_df.columns]

            # Model
            model_name = df_results_filtered_fold['Model'].values[0]
            if model_name == 'SVM':
                best_model = SVC(class_weight='balanced')
            elif model_name == 'RF':
                best_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
            elif model_name == 'XGB':
                best_model = XGBClassifier(random_state=42, n_jobs=-1)
            elif model_name == 'KNN':
                best_model = KNeighborsClassifier(n_jobs=-1)
            elif model_name == 'MLP':
                best_model = MLPClassifier(random_state=42, max_iter=1000, early_stopping=True, n_iter_no_change=10)
            else:
                raise ValueError(f"Model {model_name} not recognized.")
            best_name = model_name
            best_model.set_params(**best_params)

            # Train the model
            X_train_selected = X_train[:, selected_features_indices]
            X_test_selected = X_test[:, selected_features_indices]
            final_model = best_model.fit(X_train_selected, y_train)

            # Evaluate on the test set
            train_predictions = final_model.predict(X_train_selected)
            test_predictions = final_model.predict(X_test_selected)
            train_f1 = f1_score(y_train, train_predictions)
            train_roc = roc_auc_score(y_train, train_predictions)
            test_f1 = f1_score(y_test, test_predictions)
            test_roc = roc_auc_score(y_test, test_predictions)
            tn, fp, fn, tp = confusion_matrix(y_test, test_predictions).ravel()
            accuracy = accuracy_score(y_test, test_predictions)
            precision = precision_score(y_test, test_predictions)
            recall = recall_score(y_test, test_predictions)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

            results_classification['Target'].append([score])
            results_classification['Technique'].append(technique)
            results_classification['Voted features'].append(selected_features)
            results_classification['Dataset'].append(name_dataset)
            results_classification['Model'].append(best_name)
            results_classification['Parameters'].append(best_params)
            results_classification['F1 train'].append(train_f1)
            results_classification['ROC train'].append(train_roc)
            results_classification['True values'].append(y_test)
            results_classification['Predicted values'].append(test_predictions)
            results_classification['TN'].append(tn)
            results_classification['FP'].append(fp)
            results_classification['FN'].append(fn)
            results_classification['TP'].append(tp)
            results_classification['Accuracy'].append(accuracy)
            results_classification['F1-score'].append(test_f1)
            results_classification['ROC test'].append(test_roc)
            results_classification['Recall'].append(recall)
            results_classification['Precision'].append(precision)
            results_classification['Specificity'].append(specificity)
            results_classification['Sensitivity'].append(sensitivity)

            results_df = pd.DataFrame(results_classification)
            results_df.to_excel(os.path.join(results_path, 'classification_voted.xlsx'), index=False)

def main_regression(df, y, score, name_dataset):

    global results_regression, df_results

    X_df = df.copy()
    X = X_df.values

    out_cv = StratifiedShuffleSplit(n_splits=5, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(out_cv.split(X, y)):

        # Seleziona solo le righe nel results_df che nella colonna 'Target' hanno il valore uguale a score, dove target è ['ALSFRS-R_SpeechSubscore']
        df_results_filtered = df_results[df_results['Target'] == f"['{score}']"]
        print(len(df_results_filtered))
        # Estrai solo le righe con dataset uguale a name_dataset
        df_results_filtered = df_results_filtered[df_results_filtered['Dataset'] == name_dataset]
        print(len(df_results_filtered))

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        print(f"Fold {fold_idx}:")

        #Impute missing values
        imputer = IterativeImputer(max_iter=10, random_state=42)
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Initial feature selection (exploit the controlled randomness of the dataset)
        features_technique = ['5', '10%']
        for technique in features_technique:

            # Estrai le righe del dataframe df_results_filtered che nella colonna 'Technique' hanno il valore uguale a technique
            df_results_filtered_tech = df_results_filtered[df_results_filtered['Technique'] == technique]
            # Estrai solo la riga per fold_idx (è un indice, non un valore della colonna)
            df_results_filtered_fold = df_results_filtered_tech.iloc[fold_idx:fold_idx+1]

            # Feature selected
            selected_features_list = df_results_filtered_fold['Voted features'].values[0]
            selected_features = re.sub(r'np\.int64\((.*?)\)', r'\1', selected_features_list)
            selected_features = ast.literal_eval(selected_features)

            # Parameters
            best_params_list = df_results_filtered_fold['Parameters'].values[0]
            best_params_list = re.sub(r'np\.int64\((.*?)\)', r'\1', best_params_list)
            best_params = ast.literal_eval(best_params_list)
            # Find the indeces of the selected features in the original columns
            selected_features_indices = [X_df.columns.get_loc(col) for col in selected_features if col in X_df.columns]

            # Model
            model_name = df_results_filtered_fold['Model'].values[0]
            if model_name == 'SVM':
                best_model = SVR()
            elif model_name == 'RF':
                best_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            elif model_name == 'XGB':
                best_model = XGBRegressor(random_state=42, n_jobs=-1)
            elif model_name == 'KNN':
                best_model = KNeighborsRegressor(n_jobs=-1)
            elif model_name == 'MLP':
                best_model = MLPRegressor(random_state=42, max_iter=1000, early_stopping=True, n_iter_no_change=10)
            else:
                raise ValueError(f"Model {model_name} not recognized.")
            best_name = model_name
            best_model.set_params(**best_params)

            # Train the model
            X_train_selected = X_train[:, selected_features_indices]
            X_test_selected = X_test[:, selected_features_indices]
            final_model = best_model.fit(X_train_selected, y_train)

            # Evaluate on the test set
            train_predictions = final_model.predict(X_train_selected)
            test_predictions = final_model.predict(X_test_selected)
            train_rmse = root_mean_squared_error(y_train, train_predictions)
            test_rmse = root_mean_squared_error(y_test, test_predictions)
            train_r2 = r2_score(y_train, train_predictions)
            test_r2 = r2_score(y_test, test_predictions)
            train_round = [round(pred, 0) for pred in train_predictions]
            test_round = [round(pred, 0) for pred in test_predictions]
            train_rmse_round = root_mean_squared_error(y_train, train_round)
            test_rmse_round = root_mean_squared_error(y_test, test_round)
            train_r2_round = r2_score(y_train, train_round)
            test_r2_round = r2_score(y_test, test_round)

            results_regression['Target'].append([score])
            results_regression['Dataset'].append(name_dataset)
            results_regression['Technique'].append(technique)
            results_regression['Model'].append(best_name)
            results_regression['Parameters'].append(best_params)
            results_regression['True values'].append(y_test)
            results_regression['Predicted values'].append(test_predictions)
            results_regression['RMSE train'].append(train_rmse)
            results_regression['RMSE test'].append(test_rmse)
            results_regression['R2 train'].append(train_r2)
            results_regression['R2 test'].append(test_r2)
            results_regression['RMSE train rounded'].append(train_rmse_round)
            results_regression['RMSE test rounded'].append(test_rmse_round)
            results_regression['R2 train rounded'].append(train_r2_round)
            results_regression['R2 test rounded'].append(test_r2_round)

            results_df = pd.DataFrame(results_regression)
            results_df.to_excel(os.path.join(results_path, 'regression_voted.xlsx'), index=False)

# Main
if __name__ == "__main__":

    general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(general_path, 'Features/Old')
    results_path = os.path.join(general_path, 'Results/First test')

    # Load the cleaned dataframes
    df_complete = pd.read_excel(os.path.join(features_path, 'complete_clean.xlsx'))
    df_syllable = pd.read_excel(os.path.join(features_path, 'syllables_clean.xlsx'))
    df_vowels = pd.read_excel(os.path.join(features_path, 'vowels_clean.xlsx'))

    # Filter ALS patients and drop unnecessary columns
    als_df_complete = df_complete[df_complete['category'] == 'ALS']
    als_df_syllable = df_syllable[df_syllable['category'] == 'ALS']
    als_df_vowels = df_vowels[df_vowels['category'] == 'ALS']

    y_speech = als_df_vowels['ALSFRS-R_SpeechSubscore'].values
    y_swallowing = als_df_vowels['ALSFRS-R_SwallowingSubscore'].values
    y_bulbar = als_df_vowels['PUMNS_BulbarSubscore'].values

    id = als_df_vowels['subjid'].values

    df_results = pd.read_excel(os.path.join(results_path, 'results_regression_sep (MFCCs).xlsx'))

    results_regression = {
        'Target': [],
        'Dataset': [],
        'Model': [],
        'Technique': [],    
        'Parameters': [],
        'True values': [],
        'Predicted values': [],
        'RMSE train': [],
        'RMSE test': [],
        'R2 train': [],
        'R2 test': [],
        'RMSE train rounded': [],
        'RMSE test rounded': [],
        'R2 train rounded': [],
        'R2 test rounded': [],
    }

    results_classification = {
        'Target': [],
        'Dataset': [],
        'Model': [],
        'Parameters': [],
        'Technique': [],
        'Voted features': [],
        'F1 train': [],
        'ROC train': [],
        'True values': [],
        'Predicted values': [],
        'TP': [],
        'FP': [],
        'TN': [],
        'FN': [],
        'Accuracy': [],
        'F1-score': [],
        'ROC test': [],
        'Recall': [],
        'Precision': [],
        'Specificity': [],
        'Sensitivity': [],
    }

    # Speech
    columns_to_drop_complete = ['subjid', 'category', 'sex', 'ALSFRS-R_SpeechSubscore', 'ALSFRS-R_SwallowingSubscore', 'PUMNS_BulbarSubscore',] 
                    # 'SML11_t', 'SML12_t', 'SML13_t', 'SML21_t', 'SML22_t', 'SML23_t', 'SML31_t', 'SML32_t', 'SML33_t', 'SML41_t', 'SML42_t', 'SML43_t', 'x2D_DCT1_t', 'x2D_DCT2_t', 'x2D_DCT3_t', 'x2D_DCT4_t', 'x2D_DCT5_t', 'x2D_DCT6_t', 'x2D_DCT7_t', 'x2D_DCT8_t', 'x2D_DCT9_t',
                    # 'SML11_k', 'SML12_k', 'SML13_k', 'SML21_k', 'SML22_k', 'SML23_k', 'SML31_k', 'SML32_k', 'SML33_k', 'SML41_k', 'SML42_k', 'SML43_k', 'x2D_DCT1_k', 'x2D_DCT2_k', 'x2D_DCT3_k', 'x2D_DCT4_k', 'x2D_DCT5_k', 'x2D_DCT6_k', 'x2D_DCT7_k', 'x2D_DCT8_k', 'x2D_DCT9_k',
                    # 'SML11_p', 'SML12_p', 'SML13_p', 'SML21_p', 'SML22_p', 'SML23_p', 'SML31_p', 'SML32_p', 'SML33_p', 'SML41_p', 'SML42_p', 'SML43_p', 'x2D_DCT1_p', 'x2D_DCT2_p', 'x2D_DCT3_p', 'x2D_DCT4_p', 'x2D_DCT5_p', 'x2D_DCT6_p', 'x2D_DCT7_p', 'x2D_DCT8_p', 'x2D_DCT9_p']
    columns_to_drop_syllable = columns_to_drop_complete
    columns_to_drop_vowels = ['subjid', 'category', 'sex', 'ALSFRS-R_SpeechSubscore', 'ALSFRS-R_SwallowingSubscore', 'PUMNS_BulbarSubscore']
    
    # als_df = als_df_complete.drop(columns=columns_to_drop_complete)
    # main_classification(als_df, y_speech, 'ALSFRS-R_SpeechSubscore', 'complete')
    # als_df = als_df_syllable.drop(columns=columns_to_drop_syllable)
    # main_classification(als_df, y_speech, 'ALSFRS-R_SpeechSubscore', 'syllable')
    # als_df = als_df_vowels.drop(columns=columns_to_drop_vowels)
    # main_classification(als_df, y_speech, 'ALSFRS-R_SpeechSubscore', 'vowels')

    # als_df = als_df_complete.drop(columns=columns_to_drop_complete)
    # main_classification(als_df, y_swallowing, 'ALSFRS-R_SwallowingSubscore', 'complete')
    # als_df = als_df_syllable.drop(columns=columns_to_drop_syllable)
    # main_classification(als_df, y_swallowing, 'ALSFRS-R_SwallowingSubscore', 'syllable')
    # als_df = als_df_vowels.drop(columns=columns_to_drop_vowels)
    # main_classification(als_df, y_swallowing, 'ALSFRS-R_SwallowingSubscore', 'vowels')

    als_df = als_df_complete.drop(columns=columns_to_drop_complete)
    main_regression(als_df, y_bulbar, 'PUMNS_BulbarSubscore', 'complete')
    als_df = als_df_syllable.drop(columns=columns_to_drop_syllable)
    main_regression(als_df, y_bulbar, 'PUMNS_BulbarSubscore', 'syllable')
    als_df = als_df_vowels.drop(columns=columns_to_drop_vowels)
    main_regression(als_df, y_bulbar, 'PUMNS_BulbarSubscore', 'vowels')