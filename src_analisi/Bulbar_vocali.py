import os
import sys
import numpy as np
import pandas as pd

from mrmr import mrmr_regression
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from collections import Counter

def remove_columns(df, columns_to_remove):
    df_clean = df.drop(columns=columns_to_remove)
    return df_clean

def prepare_data(df, y):
    X_df = df.copy()
    X = X_df.values

    # Split the data into training and test sets
    train_test_split = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in train_test_split.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

    #Impute missing values
    imputer = IterativeImputer(max_iter=10, random_state=42)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_df, X_train, X_test, y_train, y_test

def regression_function(model, parameters, X_train, X_test, y_train, y_test, model_name, feature_selection, features, name_file):

    global results, results_path

    # Train-validation split and hyperparameter tuning
    train_test_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid_search = GridSearchCV(model, parameters, cv=train_test_split, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    # Evaluate on the test set for regression rmse, etc..
    test_predictions = best_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)

    results['Model'].append(model_name)
    results['Technique'].append(feature_selection)
    results['Parameters'].append(best_params)
    results['Features set'].append(features)
    results['RMSE'].append(rmse)
    results['R2'].append(r2)
    results['Validation'].append(best_score)

    results_df = pd.DataFrame(results)
    results_df.to_excel(os.path.join(results_path, name_file), index=False)

def regression(X_train_selected, X_test_selected, y_train, y_test, feature_selection, features, name_file):
    # SVM Classifier

    model = SVR()
    model_name = "SVM"
    parameters = {
        'C': [0.0001, 0.01, 0.02, 0.1, 0.2, 1, 2, 10, 20, 100, 1000],
        'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
        'degree': [2, 3, 4],
    }
    regression_function(model, parameters, X_train_selected, X_test_selected, y_train, y_test, model_name, feature_selection, features, name_file)

    # Random Forest Classifier

    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    model_name = "RF"
    parameters = {
        'n_estimators': [10, 20, 30, 40, 50, 60, 70, 100],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    regression_function(model, parameters, X_train_selected, X_test_selected, y_train, y_test, model_name, feature_selection, features, name_file)


    # XGBoost Classifier

    model = XGBRegressor(random_state=42, n_jobs=-1)
    model_name = "XGB"
    parameters = {
        'n_estimators': [10, 20, 30, 40, 50, 100, 200],
        'max_depth': [2, 3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2, 0.5, 0.7, 1.0],
        'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    }
    regression_function(model, parameters, X_train_selected, X_test_selected, y_train, y_test, model_name, feature_selection, features, name_file)

    # KNN classifier

    model = KNeighborsRegressor(n_jobs=-1)
    model_name = "KNN"
    parameters = {
        'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
    }
    regression_function(model, parameters, X_train_selected, X_test_selected, y_train, y_test, model_name, feature_selection, features, name_file)

    # MLP Classifier

    model = MLPRegressor(random_state=42, max_iter=1000, early_stopping=True, n_iter_no_change=10)
    model_name = "MLP"
    parameters = {
        'hidden_layer_sizes': [(32,), (16,), (8,), (32, 16), (16, 8)],
        'activation': ['relu', 'tanh', 'sigmoid'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive'],
    }
    regression_function(model, parameters, X_train_selected, X_test_selected, y_train, y_test, model_name, feature_selection, features, name_file)

def main_regression(X_df, X_train, X_test, y_train, y_test, name_file):

    ##################################################### 5 features per syllable/vowel ##########################################################

    features_to_select = []
    task = ['_a','_e', '_i', '_o', '_u', '_k', '_p', '_t']
    for i, id_task in enumerate(task):
        # Keep only features whose name ends with the current task
        features = [col for k, col in enumerate(X_df.columns) if col.endswith(id_task)]
        X_task_df = pd.DataFrame(X_train, columns=X_df.columns)
        X_task = X_task_df[features]
        # Select only the 5 most importat features using mRMR
        y_task = pd.Series(y_train, name='PUMNS_BulbarSubscore').astype(float)

        # Feature selection using mRMR
        selected_features = mrmr_regression(X=X_task, y=y_task, K=5)
        features_to_select.extend(selected_features)

    # Select the features in the training and test sets  --> 5 per syllable/vowel
    X_train_selected = X_train[:, [X_df.columns.get_loc(col) for col in features_to_select]]
    X_test_selected = X_test[:, [X_df.columns.get_loc(col) for col in features_to_select]]
    
    feature_selection = "5"
    regression(X_train_selected, X_test_selected, y_train, y_test, feature_selection, features_to_select, name_file)

    # #################################################### 10% features per syllable/vowel ##########################################################

    X_task_df = pd.DataFrame(X_train, columns=X_df.columns)
    y_task = pd.Series(y_train, name='PUMNS_BulbarSubscore').astype(float)
    selected_features = mrmr_regression(X=X_task_df, y=y_task, K=int(len(X_task_df.columns) * 0.1))

    X_train_selected = X_train[:, [X_df.columns.get_loc(col) for col in selected_features]]
    X_test_selected = X_test[:, [X_df.columns.get_loc(col) for col in selected_features]]

    feature_selection = "10%"
    regression(X_train_selected, X_test_selected, y_train, y_test, feature_selection, selected_features, name_file)

    #################################################### Free features per syllable/vowel ##########################################################

    num_features = X_train.shape[1]
    X_train_df = pd.DataFrame(X_train, columns=X_df.columns)
    y_train_df = pd.Series(y_train, name='PUMNS_BulbarSubscore').astype(float)
    selected_features = mrmr_regression(X_train_df, y_train_df, K=num_features)

    feature_sets = [selected_features[:i] for i in range(1, len(selected_features) + 1)]

    for feature_set in feature_sets:
        X_train_selected = X_train[:, [X_df.columns.get_loc(col) for col in feature_set]]
        X_test_selected = X_test[:, [X_df.columns.get_loc(col) for col in feature_set]]
        
        feature_selection = "Free"
        regression(X_train_selected, X_test_selected, y_train, y_test, feature_selection, feature_set, name_file)

# Main
if __name__ == "__main__":

    general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(general_path, 'Features')
    results_path = os.path.join(general_path, 'Results')

    # Load the cleaned dataframes
    df = pd.read_excel(os.path.join(features_path, 'vowels_clean.xlsx'))

    # Filter ALS patients and drop unnecessary columns
    als_df_complete = df[df['category'] == 'ALS']
    y = als_df_complete['PUMNS_BulbarSubscore'].values
    id = als_df_complete['subjid'].values

    # Compute chance level
    chance_level = max(Counter(y).values()) / len(y)

    # With MFCCs
    results = {
        'Technique': [],
        'Model': [],
        'Parameters': [],
        'Features set': [],
        'RMSE': [],
        'R2': [],
        'Validation': []
    }

    columns_to_drop = ['subjid', 'category', 'sex', 'ALSFRS-R_SpeechSubscore', 'ALSFRS-R_SwallowingSubscore', 'PUMNS_BulbarSubscore']

    als_df = remove_columns(als_df_complete, columns_to_drop)
    X_df, X_train, X_test, y_train, y_test = prepare_data(als_df, y)
    main_regression(X_df, X_train, X_test, y_train, y_test, 'bulbar_vowels.xlsx')

    # Without MFCCs
    results = {
        'Technique': [],
        'Model': [],
        'Parameters': [],
        'Features set': [],
        'RMSE': [],
        'R2': [],
        'Validation': []
    }

    columns_to_drop = ['subjid', 'category', 'sex', 'ALSFRS-R_SpeechSubscore', 'ALSFRS-R_SwallowingSubscore', 'PUMNS_BulbarSubscore',
                            'mfcc_0_a', 'mfcc_1_a', 'mfcc_2_a', 'mfcc_3_a', 'mfcc_4_a', 'mfcc_5_a', 'mfcc_6_a', 'mfcc_7_a', 'mfcc_8_a', 'mfcc_9_a', 'mfcc_10_a', 'mfcc_11_a',
                            'mfcc_0_e', 'mfcc_1_e', 'mfcc_2_e', 'mfcc_3_e', 'mfcc_4_e', 'mfcc_5_e', 'mfcc_6_e', 'mfcc_7_e', 'mfcc_8_e', 'mfcc_9_e', 'mfcc_10_e', 'mfcc_11_e',
                            'mfcc_0_i', 'mfcc_1_i', 'mfcc_2_i', 'mfcc_3_i', 'mfcc_4_i', 'mfcc_5_i', 'mfcc_6_i', 'mfcc_7_i', 'mfcc_8_i', 'mfcc_9_i', 'mfcc_10_i', 'mfcc_11_i',
                            'mfcc_0_o', 'mfcc_1_o', 'mfcc_2_o', 'mfcc_3_o', 'mfcc_4_o', 'mfcc_5_o', 'mfcc_6_o', 'mfcc_7_o', 'mfcc_8_o', 'mfcc_9_o', 'mfcc_10_o', 'mfcc_11_o',
                            'mfcc_0_u', 'mfcc_1_u', 'mfcc_2_u', 'mfcc_3_u', 'mfcc_4_u', 'mfcc_5_u', 'mfcc_6_u', 'mfcc_7_u', 'mfcc_8_u', 'mfcc_9_u', 'mfcc_10_u', 'mfcc_11_u'
]

    als_df = remove_columns(als_df_complete, columns_to_drop)
    X_df, X_train, X_test, y_train, y_test = prepare_data(als_df, y)
    main_regression(X_df, X_train, X_test, y_train, y_test, 'bulbar_noMFCCs_vowels.xlsx')