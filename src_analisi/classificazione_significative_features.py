import os
import numpy as np
import pandas as pd
import itertools
from collections import Counter

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

def remove_columns(df, columns_to_remove):
    df_clean = df.drop(columns=columns_to_remove)
    return df_clean

def main_classification(df, y, score, features_dataset, name_dataset):

    global results_classification, results_regression

    X_df = df.copy()
    X_df_copy = X_df[features_dataset]
    X = X_df_copy.values

    # Normal (4) vs Impaired (0, 1, 2, 3)
    count_normal = (y == 4).sum()
    count_impaired = (y < 4).sum()
    proportion_impaired = count_impaired / (count_normal + count_impaired) if (count_normal + count_impaired) > 0 else 0
    print(f"Score {score}: {count_normal} normal, {count_impaired} impaired, proportion impaired {proportion_impaired:.2f}")

    if score != 'PUMNS_BulbarSubscore':
        y = np.where(y == 4, 0, 1) # 0 normal, 1 impaired

    if score != 'PUMNS_BulbarSubscore':
        models = [
            {"name": "SVM", "model": SVC(class_weight='balanced'), "parameters": {
                'C': [0.0001, 0.01, 0.02, 0.1, 0.2, 1, 2, 10, 20, 100, 1000],
                'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
                'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
                'degree': [2, 3, 4],
            }},
            {"name": "RF", "model": RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'), "parameters": {
                'n_estimators': [10, 20, 30, 40, 50, 60, 70, 100, 150, 200, 300],
                'max_depth': [None, 2, 5, 7, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }},
            {"name": "XGB", "model": XGBClassifier(random_state=42, n_jobs=-1), "parameters": {
                'n_estimators': [10, 20, 30, 40, 50, 100, 200],
                'max_depth': [2, 3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2, 0.5, 0.7, 1.0],
                'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
            }},
            {"name": "KNN", "model": KNeighborsClassifier(n_jobs=-1), "parameters": {
                'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
            }},
            {"name": "MLP", "model": MLPClassifier(random_state=42, max_iter=1000, early_stopping=True, n_iter_no_change=10), "parameters": {
                'hidden_layer_sizes': [(64,), (32,), (16,), (8,), (64,32), (32, 16), (16, 8)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001],
            }},
        ]
    else:
        models = [
            {"name": "SVM", "model": SVR(), "parameters": {
                'C': [0.0001, 0.01, 0.02, 0.1, 0.2, 1, 2, 10, 20, 50],
                'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
                'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
                'degree': [2, 3, 4],
            }},
            {"name": "RF", "model": RandomForestRegressor(random_state=42, n_jobs=-1), "parameters": {
                'n_estimators': [10, 20, 30, 40, 50, 60, 70, 100, 150, 200, 300],
                'max_depth': [None, 2, 5, 7, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }},
            {"name": "XGB", "model": XGBRegressor(random_state=42, n_jobs=-1), "parameters": {
                'n_estimators': [10, 20, 30, 40, 50, 100, 200],
                'max_depth': [2, 3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2, 0.5, 0.7, 1.0],
                'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
            }},
            {"name": "KNN", "model": KNeighborsRegressor(n_jobs=-1), "parameters": {
                'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
            }},
            {"name": "MLP", "model": MLPRegressor(random_state=42, max_iter=1000, early_stopping=True, n_iter_no_change=10), "parameters": {
                'hidden_layer_sizes': [(64,), (32,), (16,), (8,), (64,32), (32, 16), (16, 8)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001],
            }},
        ]

        # Split the data into training and test sets
    if score != 'PUMNS_BulbarSubscore':
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        out_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        inner_cv = StratifiedShuffleSplit(n_splits=5, random_state=42)
        out_cv = StratifiedShuffleSplit(n_splits=5, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(out_cv.split(X, y)):
        print(test_idx)
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

        if score != 'PUMNS_BulbarSubscore':
            best_f1_validation = -np.inf
            best_model = None
            best_params = None
            best_name = None
            best_technique = None
        else:
            best_rmse_validation = np.inf
            best_model = None
            best_params = None
            best_name = None
            best_technique = None

        # Train and evaluate each model
        for model_info in models:
            model_name = model_info['name']
            model_class = model_info['model'].__class__
            model_parameters = model_info['parameters']

            for params in itertools.product(*model_parameters.values()):
                params = dict(zip(model_parameters.keys(), params))

                if score != 'PUMNS_BulbarSubscore':
                    f1_validation = []
                    roc_validation = []
                else:
                    rmse_validation = []
                
                for _, (train_idx, val_idx) in enumerate(inner_cv.split(X_train, y_train)):

                    X_inner_train, y_inner_train = X_train[train_idx], y_train[train_idx]
                    X_inner_val, y_inner_val = X_train[val_idx], y_train[val_idx]

                    model = model_class(**params)
                    model.fit(X_inner_train, y_inner_train)
                    y_inner_pred = model.predict(X_inner_val)

                    if score != 'PUMNS_BulbarSubscore':
                        validation_f1 = f1_score(y_inner_val, y_inner_pred)
                        validation_roc = roc_auc_score(y_inner_val, y_inner_pred)
                        f1_validation.append(validation_f1)
                        roc_validation.append(validation_roc)
                    else:
                        y_inner_pred = [round(pred, 0) for pred in y_inner_pred]
                        validation_rmse = root_mean_squared_error(y_inner_val, y_inner_pred)
                        rmse_validation.append(validation_rmse)

                if score != 'PUMNS_BulbarSubscore':      
                    mean_f1_validation = np.median(f1_validation)
                else:
                    mean_rmse_validation = np.median(rmse_validation)
                
                if score != 'PUMNS_BulbarSubscore':
                    if mean_f1_validation > best_f1_validation:
                        best_f1_validation = mean_f1_validation
                        best_model = model_class(**params)
                        best_params = params
                        best_name = model_name
                else:
                    if mean_rmse_validation < best_rmse_validation:
                        best_rmse_validation = mean_rmse_validation
                        best_model = model_class(**params)
                        best_params = params
                        best_name = model_name

        if score != 'PUMNS_BulbarSubscore':
            print(f"Best model for fold {fold_idx} features technique: {best_technique}: {best_model.__class__.__name__} with params: {best_params}, F1 validation: {best_f1_validation:.4f}")
        else:
            print(f"Best model for fold {fold_idx} features technique: {best_technique}: {best_model.__class__.__name__} with params: {best_params}, RMSE validation: {best_rmse_validation:.4f}")

        # Select features based on the best technique
        final_model = best_model.__class__(**best_params)
        final_model.fit(X_train, y_train)

        # Evaluate on the test set
        if score != 'PUMNS_BulbarSubscore':
            train_predictions = final_model.predict(X_train)
            test_predictions = final_model.predict(X_test)
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
            results_classification['Dataset'].append(name_dataset)
            results_classification['Model'].append(best_name)
            results_classification['Parameters'].append(best_params)
            results_classification['F1 validation'].append(best_f1_validation)
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
            results_df.to_excel(os.path.join(results_path, 'results_classification_sep (significant).xlsx'), index=False)

        else:
            train_predictions = final_model.predict(X_train)
            test_predictions = final_model.predict(X_test)
            train_rmse = root_mean_squared_log_error(y_train, train_predictions)
            test_rmse = root_mean_squared_error(y_test, test_predictions)
            train_r2 = r2_score(y_train, train_predictions)
            test_r2 = r2_score(y_test, test_predictions)
            train_round = [round(pred, 0) for pred in train_predictions]
            test_round = [round(pred, 0) for pred in test_predictions]
            train_rmse_round = root_mean_squared_log_error(y_train, train_round)
            test_rmse_round = root_mean_squared_error(y_test, test_round)
            train_r2_round = r2_score(y_train, train_round)
            test_r2_round = r2_score(y_test, test_round)

            results_regression['Target'].append([score])
            results_regression['Dataset'].append(name_dataset)
            results_regression['Model'].append(best_name)
            results_regression['Parameters'].append(best_params)
            results_regression['True values'].append(y_test)
            results_regression['Predicted values'].append(test_predictions)
            results_regression['RMSE validation'].append(best_rmse_validation)
            results_regression['RMSE train'].append(train_rmse)
            results_regression['RMSE test'].append(test_rmse)
            results_regression['R2 train'].append(train_r2)
            results_regression['R2 test'].append(test_r2)
            results_regression['RMSE train rounded'].append(train_rmse_round)
            results_regression['RMSE test rounded'].append(test_rmse_round)
            results_regression['R2 train rounded'].append(train_r2_round)
            results_regression['R2 test rounded'].append(test_r2_round)

            results_df = pd.DataFrame(results_regression)
            results_df.to_excel(os.path.join(results_path, 'results_regression_sep (significant).xlsx'), index=False)


general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
features_path = os.path.join(general_path, 'Features')
results_path = os.path.join(general_path, 'Results/First test')
statistical_path = os.path.join(results_path, 'Statistics')

df_statistics = pd.read_excel(os.path.join(statistical_path, 'statistical_analysis (MFCCs).xlsx'))
df_significative_speech = df_statistics[df_statistics['p-value Speech'] < 0.05]
features_speech = df_significative_speech['Feature'].tolist()
df_significative_swallowing = df_statistics[df_statistics['p-value Swallowing'] < 0.05]
features_swallowing = df_significative_swallowing['Feature'].tolist()
df_significative_bulbar = df_statistics[df_statistics['p-value'] < 0.05]
features_bulbar = df_significative_bulbar['Feature'].tolist()

df_statistics_noMFCCs = pd.read_excel(os.path.join(statistical_path, 'statistical_analysis.xlsx'))
df_significative_speech_noMFCCs = df_statistics_noMFCCs[df_statistics_noMFCCs['p-value Speech'] < 0.05]
features_speech_noMFCCs = df_significative_speech_noMFCCs['Feature'].tolist()
df_significative_swallowing_noMFCCs = df_statistics_noMFCCs[df_statistics_noMFCCs['p-value Swallowing'] < 0.05]
features_swallowing_noMFCCs = df_significative_swallowing_noMFCCs['Feature'].tolist()
df_significative_bulbar_noMFCCs = df_statistics_noMFCCs[df_statistics_noMFCCs['p-value'] < 0.05]
features_bulbar_noMFCCs = df_significative_bulbar_noMFCCs['Feature'].tolist()

# Load the cleaned dataframes
df_complete = pd.read_excel(os.path.join(features_path, 'complete_clean.xlsx'))
df_syllable = pd.read_excel(os.path.join(features_path, 'syllables_clean.xlsx'))
df_vowels = pd.read_excel(os.path.join(features_path, 'vowels_clean.xlsx'))

df_complete_noMFCCs = pd.read_excel(os.path.join(features_path, 'Old/complete_clean.xlsx'))
df_syllable_noMFCCs = pd.read_excel(os.path.join(features_path, 'Old/syllables_clean.xlsx'))
df_vowels_noMFCCs = pd.read_excel(os.path.join(features_path, 'Old/vowels_clean.xlsx'))

columns_to_drop = ['subjid', 'category', 'sex', 'ALSFRS-R_SpeechSubscore', 'ALSFRS-R_SwallowingSubscore', 'PUMNS_BulbarSubscore']

# Filter ALS patients and drop unnecessary columns
als_df_complete = df_complete[df_complete['category'] == 'ALS']
als_df_syllable = df_syllable[df_syllable['category'] == 'ALS']
als_df_vowels = df_vowels[df_vowels['category'] == 'ALS']

als_df_complete_noMFCCs = df_complete_noMFCCs[df_complete_noMFCCs['category'] == 'ALS']
als_df_syllable_noMFCCs = df_syllable_noMFCCs[df_syllable_noMFCCs['category'] == 'ALS']
als_df_vowels_noMFCCs = df_vowels_noMFCCs[df_vowels_noMFCCs['category'] == 'ALS']

y_speech = als_df_vowels['ALSFRS-R_SpeechSubscore'].values
y_swallowing = als_df_vowels['ALSFRS-R_SwallowingSubscore'].values
y_bulbar = als_df_vowels['PUMNS_BulbarSubscore'].values

id = als_df_vowels['subjid'].values

results_classification = {
    'Target': [],
    'Dataset': [],
    'Model': [],
    'Parameters': [],
    'F1 validation': [],
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

results_regression = {
    'Target': [],
    'Dataset': [],
    'Model': [],
    'Parameters': [],
    'RMSE validation': [],
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

# als_df = remove_columns(als_df_complete, columns_to_drop)
# main_classification(als_df, y_speech, 'ALSFRS-R_SpeechSubscore', features_speech, 'complete')
als_df = remove_columns(als_df_syllable, columns_to_drop)
features_syllables_speech = [feat for feat in features_speech if feat.endswith('_t') or feat.endswith('_k') or feat.endswith('_p')]
main_classification(als_df, y_speech, 'ALSFRS-R_SpeechSubscore', features_syllables_speech, 'syllable')
als_df = remove_columns(als_df_vowels, columns_to_drop)
features_vowels = [feat for feat in features_speech if feat.endswith('_a') or feat.endswith('_e') or feat.endswith('_i') or feat.endswith('_o') or feat.endswith('_u')]
main_classification(als_df, y_speech, 'ALSFRS-R_SpeechSubscore', features_vowels, 'vowels')


als_df = remove_columns(als_df_complete, columns_to_drop)
main_classification(als_df, y_swallowing, 'ALSFRS-R_SwallowingSubscore', features_swallowing, 'complete')
als_df = remove_columns(als_df_syllable, columns_to_drop)
features_syllables_swallowing = [feat for feat in features_swallowing if feat.endswith('_t') or feat.endswith('_k') or feat.endswith('_p')]
main_classification(als_df, y_swallowing, 'ALSFRS-R_SwallowingSubscore', features_syllables_swallowing, 'syllable')
als_df = remove_columns(als_df_vowels, columns_to_drop)
features_vowels_swallowing = [feat for feat in features_swallowing if feat.endswith('_a') or feat.endswith('_e') or feat.endswith('_i') or feat.endswith('_o') or feat.endswith('_u')]
main_classification(als_df, y_swallowing, 'ALSFRS-R_SwallowingSubscore', features_vowels_swallowing, 'vowels')


als_df = remove_columns(als_df_complete, columns_to_drop)
main_classification(als_df, y_bulbar, 'PUMNS_BulbarSubscore', features_bulbar, 'complete')
als_df = remove_columns(als_df_syllable, columns_to_drop)
features_syllables_bulbar = [feat for feat in features_bulbar if feat.endswith('_t') or feat.endswith('_k') or feat.endswith('_p')]
main_classification(als_df, y_bulbar, 'PUMNS_BulbarSubscore', features_syllables_bulbar, 'syllable')
als_df = remove_columns(als_df_vowels, columns_to_drop)
features_vowels_bulbar = [feat for feat in features_bulbar if feat.endswith('_a') or feat.endswith('_e') or feat.endswith('_i') or feat.endswith('_o') or feat.endswith('_u')]
main_classification(als_df, y_bulbar, 'PUMNS_BulbarSubscore', features_vowels_bulbar, 'vowels')



als_df = remove_columns(als_df_complete_noMFCCs, columns_to_drop)
main_classification(als_df, y_speech, 'ALSFRS-R_SpeechSubscore', features_speech_noMFCCs, 'complete_noMFCCs')
als_df = remove_columns(als_df_syllable_noMFCCs, columns_to_drop)
features_syllables_speech = [feat for feat in features_speech_noMFCCs if feat.endswith('_t') or feat.endswith('_k') or feat.endswith('_p')]
main_classification(als_df, y_speech, 'ALSFRS-R_SpeechSubscore', features_syllables_speech, 'syllable')
als_df = remove_columns(als_df_vowels_noMFCCs, columns_to_drop)
features_vowels = [feat for feat in features_speech_noMFCCs if feat.endswith('_a') or feat.endswith('_e') or feat.endswith('_i') or feat.endswith('_o') or feat.endswith('_u')]
main_classification(als_df, y_speech, 'ALSFRS-R_SpeechSubscore', features_vowels, 'vowels')


als_df = remove_columns(als_df_complete_noMFCCs, columns_to_drop)
main_classification(als_df, y_swallowing, 'ALSFRS-R_SwallowingSubscore', features_swallowing_noMFCCs, 'complete')
als_df = remove_columns(als_df_syllable_noMFCCs, columns_to_drop)
features_syllables_swallowing = [feat for feat in features_swallowing_noMFCCs if feat.endswith('_t') or feat.endswith('_k') or feat.endswith('_p')]
main_classification(als_df, y_swallowing, 'ALSFRS-R_SwallowingSubscore', features_syllables_swallowing, 'syllable')
als_df = remove_columns(als_df_vowels_noMFCCs, columns_to_drop)
features_vowels_swallowing = [feat for feat in features_swallowing_noMFCCs if feat.endswith('_a') or feat.endswith('_e') or feat.endswith('_i') or feat.endswith('_o') or feat.endswith('_u')]
main_classification(als_df, y_swallowing, 'ALSFRS-R_SwallowingSubscore', features_vowels_swallowing, 'vowels')


als_df = remove_columns(als_df_complete_noMFCCs, columns_to_drop)
main_classification(als_df, y_bulbar, 'PUMNS_BulbarSubscore', features_bulbar_noMFCCs, 'complete')
als_df = remove_columns(als_df_syllable_noMFCCs, columns_to_drop)
features_syllables_bulbar = [feat for feat in features_bulbar_noMFCCs if feat.endswith('_t') or feat.endswith('_k') or feat.endswith('_p')]
main_classification(als_df, y_bulbar, 'PUMNS_BulbarSubscore', features_syllables_bulbar, 'syllable')
als_df = remove_columns(als_df_vowels_noMFCCs, columns_to_drop)
features_vowels_bulbar = [feat for feat in features_bulbar_noMFCCs if feat.endswith('_a') or feat.endswith('_e') or feat.endswith('_i') or feat.endswith('_o') or feat.endswith('_u')]
main_classification(als_df, y_bulbar, 'PUMNS_BulbarSubscore', features_vowels_bulbar, 'vowels')
