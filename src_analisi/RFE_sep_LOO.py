import os
import numpy as np
import pandas as pd
import itertools
from collections import Counter

# from mrmr import mrmr_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit, LeaveOneOut
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, accuracy_score, root_mean_squared_error, root_mean_squared_log_error, r2_score
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import LogisticRegression, LinearRegression


def remove_columns(df, columns_to_remove):
    df_clean = df.drop(columns=columns_to_remove)
    return df_clean

def feature_selection(X_df, X_train, y_train, X_test, task, score, technique):
    selected_features = []

    if score != 'PUMNS_BulbarSubscore':
        estimator = LogisticRegression(max_iter=1000)
    else:
        estimator = LinearRegression()

    if technique == '5':
        X_task_df = pd.DataFrame(X_train, columns=X_df.columns)
        for id_task in task:
            features = [col for col in X_df.columns if col.endswith(id_task)]
            X_task = X_task_df[features]
            y_task = pd.Series(y_train, name=score)

            selector = RFE(estimator=estimator, step=1, n_features_to_select=5)
            selected_mask = selector.fit(X_task.values, y_task.values).support_
            selected_features.extend([features[i] for i in range(len(features)) if selected_mask[i]])

    elif technique == '10%':
        X_task_df = pd.DataFrame(X_train, columns=X_df.columns)
        y_task = pd.Series(y_train, name=score)
        n_features = int(len(X_task_df.columns) * 0.1)
        selector = RFE(estimator=estimator, step=1, n_features_to_select=n_features)
        selected_mask = selector.fit(X_task_df, y_task).support_
        selected_features = [X_task_df.columns[i] for i in range(len(X_task_df.columns)) if selected_mask[i]]

    elif technique == 'Free':
        X_train_df = pd.DataFrame(X_train, columns=X_df.columns)
        y_train_df = pd.Series(y_train, name=score)
        selector = RFECV(estimator=estimator, step=1,
                         cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
                         min_features_to_select=5)
        selected_mask = selector.fit(X_train_df, y_train_df).support_
        selected_features = [X_train_df.columns[i] for i in range(len(X_train_df.columns)) if selected_mask[i]]

    selected_indices = [X_df.columns.get_loc(col) for col in selected_features]
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    return X_train_selected, X_test_selected, selected_features

def main_classification(df, y, score, task, name_dataset):

    global results_classification, results_regression

    X_df = df.copy()
    X = X_df.values

    # Normal (4) vs Impaired (0, 1, 2, 3)
    count_normal = (y == 4).sum()
    count_impaired = (y < 4).sum()
    proportion_impaired = count_impaired / (count_normal + count_impaired) if (count_normal + count_impaired) > 0 else 0
    print(f"Score {score}: {count_normal} normal, {count_impaired} impaired, proportion impaired {proportion_impaired:.2f}")
    if score != 'PUMNS_BulbarSubscore':
        y = np.where(y == 4, 0, 1)  # 0 normal, 1 impaired

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

    features_technique = ['5', '10%', 'Free']

    # Split the data into training and test sets
    out_cv = LeaveOneOut()
    if score != 'PUMNS_BulbarSubscore':
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        inner_cv = StratifiedShuffleSplit(n_splits=5, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(out_cv.split(X, y)):
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
        feature_selected = {}
        for technique in features_technique:
            feature_selected[technique] = []
            for inner_folder_idx, (train_idx, val_idx) in enumerate(inner_cv.split(X_train, y_train)):
                X_inner_train, y_inner_train = X_train[train_idx], y_train[train_idx]
                X_inner_val, y_inner_val = X_train[val_idx], y_train[val_idx]

                _, _, selected_features = feature_selection(X_df, X_inner_train, y_inner_train, X_inner_val, task, score, technique)
                feature_selected[technique].append(selected_features)

        for technique in features_technique:

            if score != 'PUMNS_BulbarSubscore':
                best_f1_validation = -np.inf
                best_model = None
                best_params = None
                best_name = None
                best_technique = None
                best_features_voted = None
            else:
                best_rmse_validation = np.inf
                best_model = None
                best_params = None
                best_name = None
                best_technique = None
                best_features_voted = None

            # Train and evaluate each model
            for model_info in models:
                model_name = model_info['name']
                model_class = model_info['model'].__class__
                model_parameters = model_info['parameters']

                for params in itertools.product(*model_parameters.values()):
                    params = dict(zip(model_parameters.keys(), params))

                    if score != 'PUMNS_BulbarSubscore':
                        f1_validation = []
                    else:
                        rmse_validation = []
                    
                    feature_selections = []

                    for inner_folder_idx, (train_idx, val_idx) in enumerate(inner_cv.split(X_train, y_train)):

                        X_inner_train, y_inner_train = X_train[train_idx], y_train[train_idx]
                        X_inner_val, y_inner_val = X_train[val_idx], y_train[val_idx]

                        selected_features = feature_selected[technique][inner_folder_idx]
                        selected_indices = [X_df.columns.get_loc(col) for col in selected_features]
                        X_inner_train = X_inner_train[:, selected_indices]
                        X_inner_val = X_inner_val[:, selected_indices]

                        feature_selections.append(selected_features)

                        model = model_class(**params)
                        model.fit(X_inner_train, y_inner_train)
                        y_inner_pred = model.predict(X_inner_val)

                        if score != 'PUMNS_BulbarSubscore':
                            validation_f1 = f1_score(y_inner_val, y_inner_pred)
                            f1_validation.append(validation_f1)
                        else:
                            y_inner_pred = [round(pred, 0) for pred in y_inner_pred]
                            validation_rmse = root_mean_squared_error(y_inner_val, y_inner_pred)
                            rmse_validation.append(validation_rmse)

                    if score != 'PUMNS_BulbarSubscore':      
                        # mean_f1_validation = np.mean(f1_validation)
                        mean_f1_validation = np.median(f1_validation)
                    else:
                        # mean_rmse_validation = np.mean(rmse_validation)
                        mean_rmse_validation = np.median(rmse_validation)

                    all_features_selected = [feat for fold_feats in feature_selections for feat in fold_feats]
                    features_count = Counter(all_features_selected)
                    n_folds = len(feature_selections)
                    voted_features = [feat for feat, count in features_count.items() if count >= n_folds / 2]

                    if score != 'PUMNS_BulbarSubscore':
                        if mean_f1_validation > best_f1_validation:
                            best_f1_validation = mean_f1_validation
                            best_technique = technique
                            best_model = model_class(**params)
                            best_params = params
                            best_features_voted = voted_features
                            best_name = model_name
                    else:
                        if mean_rmse_validation < best_rmse_validation:
                            best_rmse_validation = mean_rmse_validation
                            best_technique = technique
                            best_model = model_class(**params)
                            best_params = params
                            best_features_voted = voted_features
                            best_name = model_name

            if score != 'PUMNS_BulbarSubscore':
                print(f"Best model for fold {fold_idx} features technique: {best_technique}: {best_model.__class__.__name__} with params: {best_params}, F1 validation: {best_f1_validation:.4f}")
            else:
                print(f"Best model for fold {fold_idx} features technique: {best_technique}: {best_model.__class__.__name__} with params: {best_params}, RMSE validation: {best_rmse_validation:.4f}")

            # Select features based on the best technique
            X_train_selected, X_test_selected, best_features = feature_selection(X_df, X_train, y_train, X_test, task, score, best_technique)
            final_model = best_model.__class__(**best_params)
            final_model.fit(X_train_selected, y_train)

            # Evaluate on the test set
            if score != 'PUMNS_BulbarSubscore':
                train_predictions = final_model.predict(X_train_selected)
                test_predictions = final_model.predict(X_test_selected)
                train_f1 = f1_score(y_train, train_predictions)

                results_classification['Target'].append([score])
                results_classification['Dataset'].append(name_dataset)
                results_classification['Model'].append(best_name)
                results_classification['Parameters'].append(best_params)
                results_classification['Features set'].append(best_features)
                results_classification['Technique'].append(best_technique)
                results_classification['Voted features'].append(best_features_voted)
                results_classification['F1 validation'].append(best_f1_validation)
                results_classification['F1 train'].append(train_f1)
                results_classification['True values'].append(y_test)
                results_classification['Predicted values'].append(test_predictions)

                results_df = pd.DataFrame(results_classification)
                results_df.to_excel(os.path.join(results_path, 'results_classification_sep_LOO.xlsx'), index=False)

            else:
                train_predictions = final_model.predict(X_train_selected)
                test_predictions = final_model.predict(X_test_selected)
                train_rmse = root_mean_squared_log_error(y_train, train_predictions)
                train_r2 = r2_score(y_train, train_predictions)
                train_round = [round(pred, 0) for pred in train_predictions]
                test_round = [round(pred, 0) for pred in test_predictions]
                train_rmse_round = root_mean_squared_log_error(y_train, train_round)
                train_r2_round = r2_score(y_train, train_round)

                results_regression['Target'].append([score])
                results_regression['Dataset'].append(name_dataset)
                results_regression['Model'].append(best_name)
                results_regression['Parameters'].append(best_params)
                results_regression['Features set'].append(best_features)
                results_regression['Technique'].append(best_technique)
                results_regression['Voted features'].append(best_features_voted)
                results_regression['RMSE validation'].append(best_rmse_validation)
                results_regression['RMSE train'].append(train_rmse)
                results_regression['R2 train'].append(train_r2)
                results_regression['RMSE train rounded'].append(train_rmse_round)
                results_regression['R2 train rounded'].append(train_r2_round)
                results_regression['True values'].append(y_test)
                results_regression['Predicted values'].append(test_predictions)
                results_regression['Predicted rounded'].append(test_round)

                results_df = pd.DataFrame(results_regression)
                results_df.to_excel(os.path.join(results_path, 'results_regression_sep_LOO.xlsx'), index=False)

# Main
if __name__ == "__main__":

    general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(general_path, 'Features')
    results_path = os.path.join(general_path, 'Results')

    # Load the cleaned dataframes
    df_complete = pd.read_excel(os.path.join(features_path, 'complete_clean.xlsx'))
    df_syllable = pd.read_excel(os.path.join(features_path, 'syllables_clean.xlsx'))
    df_vowels = pd.read_excel(os.path.join(features_path, 'vowels_clean.xlsx'))

    columns_to_drop_complete = ['subjid', 'category', 'sex', 'ALSFRS-R_SpeechSubscore', 'ALSFRS-R_SwallowingSubscore', 'PUMNS_BulbarSubscore', 'SML11_t', 'SML12_t', 'SML13_t', 'SML21_t', 'SML22_t', 'SML23_t', 'SML31_t', 'SML32_t', 'SML33_t', 'SML41_t', 'SML42_t', 'SML43_t', 'x2D_DCT1_t', 'x2D_DCT2_t', 'x2D_DCT3_t', 'x2D_DCT4_t', 'x2D_DCT5_t', 'x2D_DCT6_t', 'x2D_DCT7_t', 'x2D_DCT8_t', 'x2D_DCT9_t',
                        'SML11_k', 'SML12_k', 'SML13_k', 'SML21_k', 'SML22_k', 'SML23_k', 'SML31_k', 'SML32_k', 'SML33_k', 'SML41_k', 'SML42_k', 'SML43_k', 'x2D_DCT1_k', 'x2D_DCT2_k', 'x2D_DCT3_k', 'x2D_DCT4_k', 'x2D_DCT5_k', 'x2D_DCT6_k', 'x2D_DCT7_k', 'x2D_DCT8_k', 'x2D_DCT9_k',
                        'SML11_p', 'SML12_p', 'SML13_p', 'SML21_p', 'SML22_p', 'SML23_p', 'SML31_p', 'SML32_p', 'SML33_p', 'SML41_p', 'SML42_p', 'SML43_p', 'x2D_DCT1_p', 'x2D_DCT2_p', 'x2D_DCT3_p', 'x2D_DCT4_p', 'x2D_DCT5_p', 'x2D_DCT6_p', 'x2D_DCT7_p', 'x2D_DCT8_p', 'x2D_DCT9_p']
    columns_to_drop_syllable = columns_to_drop_complete 
    columns_to_drop_vowels = ['subjid', 'category', 'sex', 'ALSFRS-R_SpeechSubscore', 'ALSFRS-R_SwallowingSubscore', 'PUMNS_BulbarSubscore']

    task_complete = ['_a','_e', '_i', '_o', '_u', '_k', '_p', '_t']
    task_syllable = ['_k', '_p', '_t']
    task_vowels = ['_a','_e', '_i', '_o', '_u']

    # Filter ALS patients and drop unnecessary columns
    als_df_complete = df_complete[df_complete['category'] == 'ALS']
    als_df_syllable = df_syllable[df_syllable['category'] == 'ALS']
    als_df_vowels = df_vowels[df_vowels['category'] == 'ALS']

    y_speech = als_df_complete['ALSFRS-R_SpeechSubscore'].values
    y_swallowing = als_df_complete['ALSFRS-R_SwallowingSubscore'].values
    y_bulbar = als_df_complete['PUMNS_BulbarSubscore'].values

    id = als_df_complete['subjid'].values

    # Compute chance level
    chance_level_speech = max(Counter(y_speech).values()) / len(y_speech)
    print(f"Chance level: {chance_level_speech:.2f}")
    chance_level_swallowing = max(Counter(y_swallowing).values()) / len(y_swallowing)
    print(f"Chance level: {chance_level_swallowing:.2f}")

    results_classification = {
        'Target': [],
        'Dataset': [],
        'Model': [],
        'Parameters': [],
        'Features set': [],
        'Technique': [],
        'Voted features': [],
        'F1 validation': [],
        'F1 train': [],
        'True values': [],
        'Predicted values': [],
    }

    results_regression = {
        'Target': [],
        'Dataset': [],
        'Model': [],
        'Parameters': [],
        'Features set': [],
        'Technique': [],
        'Voted features': [],
        'RMSE validation': [],
        'RMSE train': [],
        'R2 train': [],
        'RMSE train rounded': [],
        'R2 train rounded': [],
        'True values': [],
        'Predicted values': [],
        'Predicted rounded': [],
    }

    # Complete dataset speech
    als_df = remove_columns(als_df_complete, columns_to_drop_complete)
    main_classification(als_df, y_speech, 'ALSFRS-R_SpeechSubscore', task_complete, 'complete')
    main_classification(als_df, y_swallowing, 'ALSFRS-R_SwallowingSubscore', task_complete, 'complete')
    main_classification(als_df, y_bulbar, 'PUMNS_BulbarSubscore', task_complete, 'complete')

    # Syllables dataset speech
    als_df = remove_columns(als_df_syllable, columns_to_drop_syllable)
    main_classification(als_df, y_speech, 'ALSFRS-R_SpeechSubscore', task_syllable, 'syllable')
    main_classification(als_df, y_swallowing, 'ALSFRS-R_SwallowingSubscore', task_syllable, 'syllable')
    main_classification(als_df, y_bulbar, 'PUMNS_BulbarSubscore', task_syllable, 'syllable')

    # Vowels dataset speech
    als_df = remove_columns(als_df_vowels, columns_to_drop_vowels)
    main_classification(als_df, y_speech, 'ALSFRS-R_SpeechSubscore', task_vowels, 'vowels')
    main_classification(als_df, y_swallowing, 'ALSFRS-R_SwallowingSubscore', task_vowels, 'vowels')
    main_classification(als_df, y_bulbar, 'PUMNS_BulbarSubscore', task_vowels, 'vowels')