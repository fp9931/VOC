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


general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
features_path = os.path.join(general_path, 'Features')
results_path = os.path.join(general_path, 'Results')

# Load the cleaned dataframes
df = pd.read_excel(os.path.join(features_path, 'complete_clean.xlsx'))
results_dataset = pd.read_excel(os.path.join(results_path, 'RFE', 'Swallowing', 'MFCCs', 'swallowing_rfe.xlsx'))
# Prendere solo i risultati che hanno Technique == 5
results_dataset = results_dataset[results_dataset['Technique'] == '5']

# Filter ALS patients and drop unnecessary columns
als_df_complete = df[df['category'] == 'ALS']
y = als_df_complete['ALSFRS-R_SwallowingSubscore'].values
id = als_df_complete['subjid'].values

# Compute chance level
chance_level = max(Counter(y).values()) / len(y)
print(f"Chance level: {chance_level:.2f}")

columns_to_drop = ['subjid', 'category', 'sex', 'ALSFRS-R_SpeechSubscore', 'ALSFRS-R_SwallowingSubscore', 'PUMNS_BulbarSubscore', 'SML11_t', 'SML12_t', 'SML13_t', 'SML21_t', 'SML22_t', 'SML23_t', 'SML31_t', 'SML32_t', 'SML33_t', 'SML41_t', 'SML42_t', 'SML43_t', 'x2D_DCT1_t', 'x2D_DCT2_t', 'x2D_DCT3_t', 'x2D_DCT4_t', 'x2D_DCT5_t', 'x2D_DCT6_t', 'x2D_DCT7_t', 'x2D_DCT8_t', 'x2D_DCT9_t',
                        'SML11_k', 'SML12_k', 'SML13_k', 'SML21_k', 'SML22_k', 'SML23_k', 'SML31_k', 'SML32_k', 'SML33_k', 'SML41_k', 'SML42_k', 'SML43_k', 'x2D_DCT1_k', 'x2D_DCT2_k', 'x2D_DCT3_k', 'x2D_DCT4_k', 'x2D_DCT5_k', 'x2D_DCT6_k', 'x2D_DCT7_k', 'x2D_DCT8_k', 'x2D_DCT9_k',
                        'SML11_p', 'SML12_p', 'SML13_p', 'SML21_p', 'SML22_p', 'SML23_p', 'SML31_p', 'SML32_p', 'SML33_p', 'SML41_p', 'SML42_p', 'SML43_p', 'x2D_DCT1_p', 'x2D_DCT2_p', 'x2D_DCT3_p', 'x2D_DCT4_p', 'x2D_DCT5_p', 'x2D_DCT6_p', 'x2D_DCT7_p', 'x2D_DCT8_p', 'x2D_DCT9_p']

als_df = als_df_complete.drop(columns=columns_to_drop)
y = np.where(y == 4, 0, 1)  # 0 normal, 1 impaired

parameters_svm = results_dataset['Parameters'].values[0]
parameters_svm = re.sub(r'np\.float64\((.*?)\)', r'\1', parameters_svm)
parameters_svm = ast.literal_eval(parameters_svm)

parameters_rf = results_dataset['Parameters'].values[1]
parameters_rf = re.sub(r'np\.float64\((.*?)\)', r'\1', parameters_rf)
parameters_rf = ast.literal_eval(parameters_rf)

parameters_xgb = results_dataset['Parameters'].values[2]
parameters_xgb = re.sub(r'np\.float64\((.*?)\)', r'\1', parameters_xgb)
parameters_xgb = ast.literal_eval(parameters_xgb)

parameters_knn = results_dataset['Parameters'].values[3]
parameters_knn = re.sub(r'np\.float64\((.*?)\)', r'\1', parameters_knn)
parameters_knn = ast.literal_eval(parameters_knn)

parameters_mlp = results_dataset['Parameters'].values[4]
parameters_mlp = re.sub(r'np\.float64\((.*?)\)', r'\1', parameters_mlp)
parameters_mlp = ast.literal_eval(parameters_mlp)

features = results_dataset['Features set'].values[0]
features = re.sub(r'np\.float64\((.*?)\)', r'\1', features)
features = ast.literal_eval(features)

X_df = als_df.copy()
X_df = X_df[features]
X = X_df.values

unimodal_models = {
    'SVM': SVC(**parameters_svm, class_weight='balanced', probability=True, random_state=42),
    'RF': RandomForestClassifier(**parameters_rf, random_state=42, n_jobs=-1, class_weight='balanced'),
    'XGB': XGBClassifier(**parameters_xgb, random_state=42, n_jobs=-1),
    'KNN': KNeighborsClassifier(**parameters_knn, n_jobs=-1),
    'MLP': MLPClassifier(**parameters_mlp, random_state=42, max_iter=1000, early_stopping=True, n_iter_no_change=10)
}

fusion_classifier = RandomForestClassifier(**parameters_rf, random_state=42, n_jobs=-1, class_weight='balanced')
train_test_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

meta_features = []
meta_labels = []

for train_idx, test_idx in train_test_split.split(X, y):
    X_train_ext, y_train_ext = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

model_outputs = []
test_outputs = []

#Impute missing values
imputer = IterativeImputer(max_iter=10, random_state=42)
X_train_ext = imputer.fit_transform(X_train_ext)
X_test = imputer.transform(X_test)

# Scale the data
scaler = StandardScaler()
X_train_ext = scaler.fit_transform(X_train_ext)
X_test = scaler.transform(X_test)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_features = []
meta_labels = []

meta_feature_matrix = np.zeros((X_train_ext.shape[0], len(unimodal_models)))

for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(skf.split(X_train_ext, y_train_ext)):
    X_train_fold, y_train_fold = X_train_ext[train_fold_idx], y_train_ext[train_fold_idx]
    X_val_fold, y_val_fold = X_train_ext[val_fold_idx], y_train_ext[val_fold_idx]

    for i, (name, model) in enumerate(unimodal_models.items()):
            model.fit(X_train_fold, y_train_fold)
            probs = model.predict_proba(X_val_fold)[:, 1]
            meta_feature_matrix[val_fold_idx, i] = probs

meta_features = meta_feature_matrix
meta_labels = y_train_ext      
# Addestra i modelli base sull'intero training esteso per test set
test_feature_matrix = []
for name, model in unimodal_models.items():
    model.fit(X_train_ext, y_train_ext)
    probs_test = model.predict_proba(X_test)[:, 1]
    test_feature_matrix.append(probs_test.reshape(-1, 1))

test_features = np.hstack(test_feature_matrix)
test_labels = y_test

# Addestra il classificatore di fusione
fusion_classifier.fit(meta_features, meta_labels)
final_preds = fusion_classifier.predict(test_features)

# Valutazione finale
print("Accuracy:", accuracy_score(test_labels, final_preds))
print("F1-score:", f1_score(test_labels, final_preds))
print("Recall:", recall_score(test_labels, final_preds))
print("Precision:", precision_score(test_labels, final_preds))