import os
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import random

from mrmr import mrmr_classif
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GroupKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, accuracy_score, root_mean_squared_error, root_mean_squared_log_error, r2_score, roc_auc_score
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import LogisticRegression, LinearRegression

from shap_figure import shap_analysis

seed = 42
np.random.seed(seed)
random.seed(seed)

def remove_columns(df, columns_to_remove):
    df_clean = df.drop(columns=columns_to_remove)
    return df_clean

def fisher_score(X, y):
    scores = []
    classes = np.unique(y)
    n = X.shape[0]

    for i in range(X.shape[1]):
        feature = X[:, i]
        overall_mean = np.mean(feature)
        
        num = 0
        denom = 0
        for c in classes:
            idx = np.where(y == c)[0]
            n_c = len(idx)
            class_mean = np.mean(feature[idx])
            class_var = np.var(feature[idx])
            num += n_c * (class_mean - overall_mean) ** 2
            denom += n_c * class_var
        score = num / denom if denom != 0 else 0
        scores.append(score)
    return np.array(scores)

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
        
        X_train_df_copy = X_train_df_copy.drop(columns=['Onset', 'Age', 'Disease duration'])
        selector = RFE(estimator=estimator, step=1, n_features_to_select=n_features)
        selected_mask = selector.fit(X_train_df_copy, y_task).support_
        selected_features = [X_train_df_copy.columns[i] for i in range(len(X_train_df_copy.columns)) if selected_mask[i]]

    elif technique == 'Free':
        X_train_df_copy = pd.DataFrame(X_train, columns=X_df.columns)

        X_train_df_copy = X_train_df_copy.drop(columns=['Onset', 'Age', 'Disease duration'])
        y_train_df = pd.Series(y_train, name=score)
        selector = RFECV(estimator=estimator, step=1,
                         cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
                         min_features_to_select=5)
        selected_mask = selector.fit(X_train_df_copy, y_train_df).support_
        selected_features = [X_train_df_copy.columns[i] for i in range(len(X_train_df_copy.columns)) if selected_mask[i]]

    elif technique == '10% mRMR':
        X_train_df_copy = pd.DataFrame(X_train, columns=X_df.columns)
        X_train_df_copy = X_train_df_copy.drop(columns=['Onset', 'Age', 'Disease duration'])

        y_task = pd.Series(y_train, name=score)
        n_features = int(len(X_train_df_copy.columns) * 0.1)

        selected_features = mrmr_classif(X=X_train_df_copy, y=y_task, K=n_features)
    
    elif technique == '5 mRMR':
        X_task_df = pd.DataFrame(X_train, columns=X_df.columns)
        y_task = pd.Series(y_train, name=score)

        for id_task in task:
            features = [col for col in X_df.columns if col.endswith(id_task)]
            X_task = X_task_df[features]
            y_task = pd.Series(y_train, name=score)

            selected_mask = mrmr_classif(X=X_task, y=y_task, K=5)
            selected_features.extend(selected_mask)

    elif technique == 'Fisher':
        best_score = -np.inf
        best_features = None
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        X_train_df_copy = pd.DataFrame(X_train, columns=X_df.columns)
        X_train_df_copy = X_train_df_copy.drop(columns=['Onset', 'Age', 'Disease duration'])
        X_train_values = X_train_df_copy.values
        y_train_df = pd.Series(y_train, name=score)

        for train_idx, val_idx in cv.split(X_train_values, y_train):

            X_tr, y_tr = X_train_values[train_idx], y_train[train_idx]
            X_val, y_val = X_train_values[val_idx], y_train[val_idx]

            y = LabelEncoder().fit_transform(y_tr)
            scores = fisher_score(X_tr, y)
            feature_ranking = np.argsort(scores)[::-1]
            percentages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
            for pct in percentages:
                n_features = int(len(scores) * pct)
                top_features = feature_ranking[:n_features]

                estimator.fit(X_tr[:, top_features], y_tr)
                y_pred = estimator.predict(X_val[:, top_features])
                score = f1_score(y_pred, y_val, average='weighted')
                if score > best_score:
                    best_score = score
                    best_features = top_features

                    selected_features = [X_train_df_copy.columns[i] for i in best_features]


    #selected_features.append("Age")

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
        y = np.where(y == 4, 0, 1) # 0 normal, 1 impaired

    if score != 'PUMNS_BulbarSubscore':
        models = [
            {"name": "SVM", "model": SVC(class_weight='balanced'), "parameters": {
                'C': [0.0001, 0.01, 0.02, 0.1, 0.2, 1, 2, 10, 20, 100, 1000],
                'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
                'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
                'degree': [2, 3, 4],
            }},
            {"name": "RF", "model": RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight='balanced'), "parameters": {
                'n_estimators': [10, 20, 30, 40, 50, 60, 60, 70, 100, 150],
                'max_depth': [None, 2, 5, 7, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }},
            {"name": "XGB", "model": XGBClassifier(random_state=seed, n_jobs=-1), "parameters": {
                'n_estimators': [10, 20, 30, 40, 50, 100],
                'max_depth': [2, 3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2, 0.5, 0.7, 1.0],
                'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
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
            {"name": "RF", "model": RandomForestRegressor(random_state=seed, n_jobs=-1), "parameters": {
                'n_estimators': [10, 20, 30, 40, 50, 60, 60, 70, 100, 150, 200, 300],
                'max_depth': [None, 2, 5, 7, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }},
            {"name": "XGB", "model": XGBRegressor(random_state=seed, n_jobs=-1), "parameters": {
                'n_estimators': [10, 20, 30, 40, 50, 100, 200],
                'max_depth': [2, 3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2, 0.5, 0.7, 1.0],
                'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
            }},
        ]

    features_technique = ['5 mRMR', '10% mRMR', '5', '10%', 'Free']
    model_shuffled = models

    # Split the data into training and test sets
    if score != 'PUMNS_BulbarSubscore':
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        out_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    else:
        inner_cv = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
        out_cv = GroupKFold(n_splits=5, shuffle=True, random_state=seed)

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
        imputer = IterativeImputer(max_iter=10, random_state=seed)
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
            for model_info in model_shuffled:
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
                        mean_f1_validation = np.median(f1_validation)
                    else:
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
                results_classification['Features set'].append(best_features)
                results_classification['Technique'].append(best_technique)
                results_classification['idx test'].append(test_idx)
                results_classification['Voted features'].append(best_features_voted)
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
                results_df.to_excel(os.path.join(results_path, 'classification.xlsx'), index=False)

            else:
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
                results_regression['Model'].append(best_name)
                results_regression['Parameters'].append(best_params)
                results_regression['Features set'].append(best_features)
                results_regression['Technique'].append(best_technique)
                results_regression['idx test'].append(test_idx)
                results_regression['Voted features'].append(best_features_voted)
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
                results_df.to_excel(os.path.join(results_path, 'regression.xlsx'), index=False)


# Main
if __name__ == "__main__":

    general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(general_path, 'Features')
    results_path = os.path.join(general_path, 'Results')

    # Load the cleaned dataframes
    df_complete = pd.read_excel(os.path.join(features_path, 'Features_complete.xlsx'))
    df_syllable = pd.read_excel(os.path.join(features_path, 'Features_syllable.xlsx'))
    df_vowels = pd.read_excel(os.path.join(features_path, 'Features_vowels.xlsx'))

    columns_to_drop = ['subjid', 'sex', 'category', 'ALSFRS-R_SpeechSubscore', 'ALSFRS-R_SwallowingSubscore', 'PUMNS_BulbarSubscore']

    task_complete = ['_a','_e', '_i', '_o', '_u', '_k', '_p', '_t']
    task_syllable = ['_k', '_p', '_t']
    task_vowels = ['_a','_e', '_i', '_o', '_u']

    y_speech = df_complete['ALSFRS-R_SpeechSubscore'].values
    y_swallowing = df_syllable['ALSFRS-R_SwallowingSubscore'].values
    y_bulbar = df_vowels['PUMNS_BulbarSubscore'].values

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
        'idx test': [],
        'Voted features': [],
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
        'Features set': [],
        'Technique': [],
        'idx test': [],
        'Voted features': [],
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

    # # Rimuovere tutte le colonne che iniziano con 'mfcc'
    # df_complete = df_complete.loc[:, ~df_complete.columns.str.startswith('mfcc')]
    # df_syllable = df_syllable.loc[:, ~df_syllable.columns.str.startswith('mfcc')]
    # df_vowels = df_vowels.loc[:, ~df_vowels.columns.str.startswith('mfcc')]

    # Bulbar
    df = remove_columns(df_complete, columns_to_drop)
    main_classification(df, y_bulbar, 'PUMNS_BulbarSubscore', task_complete, 'complete')
    df = remove_columns(df_syllable, columns_to_drop)
    main_classification(df, y_bulbar, 'PUMNS_BulbarSubscore', task_syllable, 'syllable')
    df = remove_columns(df_vowels, columns_to_drop)
    main_classification(df, y_bulbar, 'PUMNS_BulbarSubscore', task_vowels, 'vowels')

    # Speech
    df = remove_columns(df_complete, columns_to_drop)
    main_classification(df, y_speech, 'ALSFRS-R_SpeechSubscore', task_complete, 'complete')
    df = remove_columns(df_syllable, columns_to_drop)
    main_classification(df, y_speech, 'ALSFRS-R_SpeechSubscore', task_syllable, 'syllable')
    df = remove_columns(df_vowels, columns_to_drop)
    main_classification(df, y_speech, 'ALSFRS-R_SpeechSubscore', task_vowels, 'vowels')

    # Swallowing
    df = remove_columns(df_complete, columns_to_drop)
    main_classification(df, y_swallowing, 'ALSFRS-R_SwallowingSubscore', task_complete, 'complete')
    df = remove_columns(df_syllable, columns_to_drop)
    main_classification(df, y_swallowing, 'ALSFRS-R_SwallowingSubscore', task_syllable, 'syllable')
    df = remove_columns(df_vowels, columns_to_drop)
    main_classification(df, y_swallowing, 'ALSFRS-R_SwallowingSubscore', task_vowels, 'vowels')