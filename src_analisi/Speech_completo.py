import os
import sys
import numpy as np
import pandas as pd

from mrmr import mrmr_classif
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

from collections import Counter

def remove_columns(df, columns_to_remove):
    df_clean = df.drop(columns=columns_to_remove)
    return df_clean

def prepare_data(df, y):
    X_df = df.copy()
    X = X_df.values
        
    # Binarize the target variable
    # Normal (4) vs Impaired (0, 1, 2, 3)
    count_normal = (y == 4).sum()
    count_impaired = (y < 4).sum()
    proportion_impaired = count_impaired / (count_normal + count_impaired) if (count_normal + count_impaired) > 0 else 0
    print(f"Score ALSFRS-R_SpeechSubscore: {count_normal} normal, {count_impaired} impaired, proportion impaired {proportion_impaired:.2f}")
    y = np.where(y == 4, 0, 1)  # 0 normal, 1 impaired

    # Split the data into training and test sets
    train_test_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
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

def classification_function(model, parameters, X_train, X_test, y_train, y_test, model_name, feature_selection, features, name_file):

    global results, results_path

    # Train-validation split and hyperparameter tuning
    train_test_split = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid_search = GridSearchCV(model, parameters, cv=train_test_split, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    if model_name == "MLP":
        best_model.fit(X_train, y_train)

    # Evaluate on the test set
    best_score = f1_score(y_train, best_model.predict(X_train))
    test_predictions = best_model.predict(X_test)
    test_f1 = f1_score(y_test, test_predictions)
    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions).ravel()
    accuracy = accuracy_score(y_test, test_predictions)
    precision = precision_score(y_test, test_predictions)
    recall = recall_score(y_test, test_predictions)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    results['Model'].append(model_name)
    results['Technique'].append(feature_selection)
    results['Parameters'].append(best_params)
    results['Features set'].append(features)
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
    results['F1 train'].append(best_score)

    results_df = pd.DataFrame(results)
    results_df.to_excel(os.path.join(results_path, name_file), index=False)

def classification(X_train_selected, X_test_selected, y_train, y_test, feature_selection, features, name_file):
    # SVM Classifier

    model = SVC(class_weight='balanced')
    model_name = "SVM"
    parameters = {
        'C': [0.0001, 0.01, 0.02, 0.1, 0.2, 1, 2, 10, 20, 100, 1000],
        'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
        'degree': [2, 3, 4],
    }
    classification_function(model, parameters, X_train_selected, X_test_selected, y_train, y_test, model_name, feature_selection, features, name_file)

    # Random Forest Classifier

    model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    model_name = "RF"
    parameters = {
        'n_estimators': [10, 20, 30, 40, 50, 60, 70, 100],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    classification_function(model, parameters, X_train_selected, X_test_selected, y_train, y_test, model_name, feature_selection, features, name_file)


    # XGBoost Classifier

    model = XGBClassifier(random_state=42, n_jobs=-1, )
    model_name = "XGB"
    y_train_positive = len(np.where(y_train == 1)[0])
    y_train_negative = len(np.where(y_train == 0)[0])
    scale_pos_weight = y_train_negative / y_train_positive if y_train_positive > 0 else 1
    model.set_params(scale_pos_weight=scale_pos_weight)
    parameters = {
        'n_estimators': [10, 20, 30, 40, 50, 100, 200],
        'max_depth': [2, 3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2, 0.5, 0.7, 1.0],
        'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    }
    classification_function(model, parameters, X_train_selected, X_test_selected, y_train, y_test, model_name, feature_selection, features, name_file)

    # KNN classifier

    model = KNeighborsClassifier(n_jobs=-1)
    model_name = "KNN"
    parameters = {
        'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
    }
    classification_function(model, parameters, X_train_selected, X_test_selected, y_train, y_test, model_name, feature_selection, features, name_file)

    # MLP Classifier

    model = MLPClassifier(random_state=42, max_iter=1000, early_stopping=True, n_iter_no_change=10)
    model_name = "MLP"
    parameters = {
        'hidden_layer_sizes': [(64,), (32,), (16,), (8,), (64,32), (32, 16), (16, 8)],
        'activation': ['relu', 'tanh', 'sigmoid'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive'],
    }
    classification_function(model, parameters, X_train_selected, X_test_selected, y_train, y_test, model_name, feature_selection, features, name_file)

def main_classification(X_df, X_train, X_test, y_train, y_test, name_file):

    ##################################################### 5 features per syllable/vowel ##########################################################

    features_to_select = []
    task = ['_a','_e', '_i', '_o', '_u', '_k', '_p', '_t']
    for i, id_task in enumerate(task):
        # Keep only features whose name ends with the current task
        features = [col for k, col in enumerate(X_df.columns) if col.endswith(id_task)]
        X_task_df = pd.DataFrame(X_train, columns=X_df.columns)
        X_task = X_task_df[features]

        # Select only the 5 most importat features using mRMR
        y_task = pd.Series(y_train, name='ALSFRS-R_SpeechSubscore')
        # Feature selection using mRMR
        selected_features = mrmr_classif(X=X_task, y=y_task, K=5)

        features_to_select.extend(selected_features)

    # Select the features in the training and test sets  --> 5 per syllable/vowel
    X_train_selected = X_train[:, [X_df.columns.get_loc(col) for col in features_to_select]]
    X_test_selected = X_test[:, [X_df.columns.get_loc(col) for col in features_to_select]]
    
    feature_selection = "5"
    classification(X_train_selected, X_test_selected, y_train, y_test, feature_selection, features_to_select, name_file)

    # #################################################### 10% features per syllable/vowel ##########################################################

    X_task_df = pd.DataFrame(X_train, columns=X_df.columns)
    y_task = pd.Series(y_train, name='ALSFRS-R_SpeechSubscore')
    selected_features = mrmr_classif(X=X_task_df, y=y_task, K=int(len(X_task_df.columns) * 0.1))

    X_train_selected = X_train[:, [X_df.columns.get_loc(col) for col in selected_features]]
    X_test_selected = X_test[:, [X_df.columns.get_loc(col) for col in selected_features]]

    feature_selection = "10%"
    classification(X_train_selected, X_test_selected, y_train, y_test, feature_selection, selected_features, name_file)

    #################################################### Free features per syllable/vowel ##########################################################

    # num_features = X_train.shape[1]
    # X_train_df = pd.DataFrame(X_train, columns=X_df.columns)
    # y_train_df = pd.Series(y_train, name='ALSFRS-R_SpeechSubscore')
    # selected_features = mrmr_classif(X_train_df, y_train_df, K=num_features)

    # feature_sets = [selected_features[:i] for i in range(5, len(selected_features)//2, 5)]
    # print(len(feature_sets))

    # for feature_set in feature_sets:
    #     X_train_selected = X_train[:, [X_df.columns.get_loc(col) for col in feature_set]]
    #     X_test_selected = X_test[:, [X_df.columns.get_loc(col) for col in feature_set]]
        
    #     feature_selection = "Free"
    #     classification(X_train_selected, X_test_selected, y_train, y_test, feature_selection, feature_set, name_file)

# Main
if __name__ == "__main__":

    general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(general_path, 'Features')
    results_path = os.path.join(general_path, 'Results')

    # Load the cleaned dataframes
    df = pd.read_excel(os.path.join(features_path, 'complete_clean.xlsx'))

    # Filter ALS patients and drop unnecessary columns
    als_df_complete = df[df['category'] == 'ALS']
    y = als_df_complete['ALSFRS-R_SpeechSubscore'].values
    id = als_df_complete['subjid'].values

    # Compute chance level
    chance_level = max(Counter(y).values()) / len(y)
    print(f"Chance level: {chance_level:.2f}")

    # With MFCCs
    results = {
        'Technique': [],
        'Model': [],
        'Parameters': [],
        'Features set': [],
        'TP': [],
        'FP': [],
        'TN': [],
        'FN': [],
        'Accuracy': [],
        'F1-score': [],
        'Recall': [],
        'Precision': [],
        'Specificity': [],
        'Sensitivity': [],
        'F1 train': []
    }

    columns_to_drop = ['subjid', 'category', 'sex', 'ALSFRS-R_SpeechSubscore', 'ALSFRS-R_SwallowingSubscore', 'PUMNS_BulbarSubscore', 'SML11_t', 'SML12_t', 'SML13_t', 'SML21_t', 'SML22_t', 'SML23_t', 'SML31_t', 'SML32_t', 'SML33_t', 'SML41_t', 'SML42_t', 'SML43_t', 'x2D_DCT1_t', 'x2D_DCT2_t', 'x2D_DCT3_t', 'x2D_DCT4_t', 'x2D_DCT5_t', 'x2D_DCT6_t', 'x2D_DCT7_t', 'x2D_DCT8_t', 'x2D_DCT9_t',
                            'SML11_k', 'SML12_k', 'SML13_k', 'SML21_k', 'SML22_k', 'SML23_k', 'SML31_k', 'SML32_k', 'SML33_k', 'SML41_k', 'SML42_k', 'SML43_k', 'x2D_DCT1_k', 'x2D_DCT2_k', 'x2D_DCT3_k', 'x2D_DCT4_k', 'x2D_DCT5_k', 'x2D_DCT6_k', 'x2D_DCT7_k', 'x2D_DCT8_k', 'x2D_DCT9_k',
                            'SML11_p', 'SML12_p', 'SML13_p', 'SML21_p', 'SML22_p', 'SML23_p', 'SML31_p', 'SML32_p', 'SML33_p', 'SML41_p', 'SML42_p', 'SML43_p', 'x2D_DCT1_p', 'x2D_DCT2_p', 'x2D_DCT3_p', 'x2D_DCT4_p', 'x2D_DCT5_p', 'x2D_DCT6_p', 'x2D_DCT7_p', 'x2D_DCT8_p', 'x2D_DCT9_p']

    als_df = remove_columns(als_df_complete, columns_to_drop)
    X_df, X_train, X_test, y_train, y_test = prepare_data(als_df, y)
    main_classification(X_df, X_train, X_test, y_train, y_test, 'speech.xlsx')

    # Without MFCCs
    results = {
        'Technique': [],
        'Model': [],
        'Parameters': [],
        'Features set': [],
        'TP': [],
        'FP': [],
        'TN': [],
        'FN': [],
        'Accuracy': [],
        'F1-score': [],
        'Recall': [],
        'Precision': [],
        'Specificity': [],
        'Sensitivity': [],
        'F1 train': []
    }

    columns_to_drop = ['subjid', 'category', 'sex', 'ALSFRS-R_SpeechSubscore', 'ALSFRS-R_SwallowingSubscore', 'PUMNS_BulbarSubscore', 'SML11_t', 'SML12_t', 'SML13_t', 'SML21_t', 'SML22_t', 'SML23_t', 'SML31_t', 'SML32_t', 'SML33_t', 'SML41_t', 'SML42_t', 'SML43_t', 'x2D_DCT1_t', 'x2D_DCT2_t', 'x2D_DCT3_t', 'x2D_DCT4_t', 'x2D_DCT5_t', 'x2D_DCT6_t', 'x2D_DCT7_t', 'x2D_DCT8_t', 'x2D_DCT9_t',
                            'SML11_k', 'SML12_k', 'SML13_k', 'SML21_k', 'SML22_k', 'SML23_k', 'SML31_k', 'SML32_k', 'SML33_k', 'SML41_k', 'SML42_k', 'SML43_k', 'x2D_DCT1_k', 'x2D_DCT2_k', 'x2D_DCT3_k', 'x2D_DCT4_k', 'x2D_DCT5_k', 'x2D_DCT6_k', 'x2D_DCT7_k', 'x2D_DCT8_k', 'x2D_DCT9_k',
                            'SML11_p', 'SML12_p', 'SML13_p', 'SML21_p', 'SML22_p', 'SML23_p', 'SML31_p', 'SML32_p', 'SML33_p', 'SML41_p', 'SML42_p', 'SML43_p', 'x2D_DCT1_p', 'x2D_DCT2_p', 'x2D_DCT3_p', 'x2D_DCT4_p', 'x2D_DCT5_p', 'x2D_DCT6_p', 'x2D_DCT7_p', 'x2D_DCT8_p', 'x2D_DCT9_p',
                            'mfcc_0_t', 'mfcc_1_t', 'mfcc_2_t', 'mfcc_3_t', 'mfcc_4_t', 'mfcc_5_t', 'mfcc_6_t', 'mfcc_7_t', 'mfcc_8_t', 'mfcc_9_t', 'mfcc_10_t', 'mfcc_11_t',
                            'mfcc_0_k', 'mfcc_1_k', 'mfcc_2_k', 'mfcc_3_k', 'mfcc_4_k', 'mfcc_5_k', 'mfcc_6_k', 'mfcc_7_k', 'mfcc_8_k', 'mfcc_9_k', 'mfcc_10_k', 'mfcc_11_k',
                            'mfcc_0_p', 'mfcc_1_p', 'mfcc_2_p', 'mfcc_3_p', 'mfcc_4_p', 'mfcc_5_p', 'mfcc_6_p', 'mfcc_7_p', 'mfcc_8_p', 'mfcc_9_p', 'mfcc_10_p', 'mfcc_11_p',
                            'mfcc_0_a', 'mfcc_1_a', 'mfcc_2_a', 'mfcc_3_a', 'mfcc_4_a', 'mfcc_5_a', 'mfcc_6_a', 'mfcc_7_a', 'mfcc_8_a', 'mfcc_9_a', 'mfcc_10_a', 'mfcc_11_a',
                            'mfcc_0_e', 'mfcc_1_e', 'mfcc_2_e', 'mfcc_3_e', 'mfcc_4_e', 'mfcc_5_e', 'mfcc_6_e', 'mfcc_7_e', 'mfcc_8_e', 'mfcc_9_e', 'mfcc_10_e', 'mfcc_11_e',
                            'mfcc_0_i', 'mfcc_1_i', 'mfcc_2_i', 'mfcc_3_i', 'mfcc_4_i', 'mfcc_5_i', 'mfcc_6_i', 'mfcc_7_i', 'mfcc_8_i', 'mfcc_9_i', 'mfcc_10_i', 'mfcc_11_i',
                            'mfcc_0_o', 'mfcc_1_o', 'mfcc_2_o', 'mfcc_3_o', 'mfcc_4_o', 'mfcc_5_o', 'mfcc_6_o', 'mfcc_7_o', 'mfcc_8_o', 'mfcc_9_o', 'mfcc_10_o', 'mfcc_11_o',
                            'mfcc_0_u', 'mfcc_1_u', 'mfcc_2_u', 'mfcc_3_u', 'mfcc_4_u', 'mfcc_5_u', 'mfcc_6_u', 'mfcc_7_u', 'mfcc_8_u', 'mfcc_9_u', 'mfcc_10_u', 'mfcc_11_u'
]

    als_df = remove_columns(als_df_complete, columns_to_drop)
    X_df, X_train, X_test, y_train, y_test = prepare_data(als_df, y)
    main_classification(X_df, X_train, X_test, y_train, y_test, 'speech_noMFCCs.xlsx')