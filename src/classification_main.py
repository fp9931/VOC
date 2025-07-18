import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from mrmr import mrmr_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
features_path = os.path.join(general_path, 'Features')
results_path = os.path.join(general_path, 'Results')

vowels_df = pd.read_excel(os.path.join(features_path, 'vowels.xlsx'))

ID_column = 'subjid'
label_column = 'category'
sex_column = 'sex'
clinical_scores = ["ALSFRS-R_SpeechSubscore", "ALSFRS-R_SwallowingSubscore", "PUMNS_BulbarSubscore", "PUMNS_UpperLimbsSubscore", "PUMNS_LowerLimbsSubscore"]

# Compute proprortion of each classes based on the clinical scores in ALS patients
als_df = vowels_df[vowels_df[label_column] == 'ALS']
data = als_df.drop(columns=[ID_column, label_column, sex_column, *clinical_scores]).values
labels = als_df[label_column].values
id = als_df[ID_column].values

models = ["SVM", "KNN", "RandomForest", "XGBoost", "MLP"]
parameters = {
    "SVM": {
        'C': [0.01, 0.01, 0.1, 1, 10, 100, 1000],
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'gamma': [0.01, 0.1, 1],
        'class_weight': [None, 'balanced']
    },
    # "KNN": {
    #     'n_neighbors': [3, 5, 7, 9],
    #     'weights': ['uniform', 'distance'],
    #     'metric': ['euclidean', 'manhattan']
    # },
    # "RandomForest": {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [None, 10, 20],
    #     'min_samples_split': [2, 5],
    #     'class_weight': [None, 'balanced']
    # },
    # "XGBoost": {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [3, 6, 9],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'subsample': [0.8, 1.0],
    #     'colsample_bytree': [0.8, 1.0],
    # },
    # "MLP": {
    #     'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    #     'activation': ['relu', 'tanh'],
    #     'solver': ['adam', 'sgd'],
    #     'alpha': [0.0001, 0.001],
    #     'learning_rate': ['constant', 'adaptive'],
    #     'max_iter': [200, 500]
    # }
}

# Classify ALS patients based on clinical scores
for score in clinical_scores:
    count_normal = 0
    count_impaired = 0
    proportion_impaired = 0
    if score.startswith("ALSFRS-R"):
        count_normal += (als_df[score] == 4).sum()
        count_impaired += (als_df[score] < 4).sum()
        proportion_impaired = count_impaired / (count_normal + count_impaired) if (count_normal + count_impaired) > 0 else 0
        print(f"Score {score}: {count_normal} normal, {count_impaired} impaired, proportion impaired {proportion_impaired:.2f}")
    else:
        break

    # Data 
    X = data
    y = als_df[score].values
    # Make normal, impaired binary classification
    y = np.where(y == 4, 0, 1)  # 0 normal, 1 impaired

    # Split the data into training, validation and test sets mantaining the proportion of impaired ALS patients
    X_train_ext, X_test, y_train_ext, y_test = None, None, None, None

    train_test_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx_ext, test_idx in train_test_split.split(X, y):
        X_train_ext, y_train_ext = X[train_idx_ext], y[train_idx_ext]
        X_test, y_test = X[test_idx], y[test_idx]

    # Compute proprortion of each classes in the training set
    count_normal = np.sum(y_train_ext == 0)
    count_impaired = np.sum(y_train_ext == 1)
    proportion_impaired = count_impaired / (count_normal + count_impaired) if (count_normal + count_impaired) > 0 else 0
    print(f"Score {score}: {count_normal} normal, {count_impaired} impaired, proportion impaired {proportion_impaired:.2f}")

    count_normal = np.sum(y_test == 0)
    count_impaired = np.sum(y_test == 1)
    proportion_impaired = count_impaired / (count_normal + count_impaired) if (count_normal + count_impaired) > 0 else 0
    print(f"Score {score}: {count_normal} normal, {count_impaired} impaired, proportion impaired {proportion_impaired:.2f}")

    # Convert to DataFrame for mRMR
    X_train_df = pd.DataFrame(X_train_ext, columns=als_df.columns.drop([ID_column, label_column, sex_column, *clinical_scores]))
    print(f"Training set shape: {X_train_df.shape}")
    y_train_series = pd.Series(y_train_ext, name=label_column)
    # Feature selection using mRMR
    selected_features = mrmr_classif(X=X_train_df, y=y_train_series, K=len(X_train_df.columns))

    pipeline = Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=42)),
        ('scaler', StandardScaler()),
    ])

    # Fit the pipeline on the training data
    X_train_tot = pipeline.fit_transform(X_train_ext, y_train_ext)
    X_test_tot = pipeline.transform(X_test)

    best_score = 0
    best_model = None
    best_params = None
    best_features = None
    best_features_indices = None

    number_of_features = len(selected_features)
    for i in range(5, number_of_features+1, 5):
        selected_features_subset = selected_features[0:i]
        selected_features_indices = [X_train_df.columns.get_loc(feature) for feature in selected_features_subset]

        # Reduce the training, validation and test sets to the selected features
        X_train = X_train_tot[:, selected_features_indices]

        train_validation_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        # Grid search for hyperparameter tuning
        for model in models:
            print(f"Training model: {model}")
            if model == "SVM":
                clf = SVC()
            # elif model == "KNN":
            #     clf = KNeighborsClassifier()
            # elif model == "RandomForest":
            #     clf = RandomForestClassifier(random_state=42)
            # elif model == "XGBoost":
            #     clf = XGBClassifier(random_state=42, n_jobs=1)
            # elif model == "MLP":
            #     clf = MLPClassifier(max_iter=1000, random_state=42)
            else:
                continue
            
            grid_search = GridSearchCV(clf, parameters[model], cv=train_validation_split, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train, y_train_ext)

            # Choose the model with the best score on the training set (F1 score)
            train_predictions = grid_search.predict(X_train)
            train_f1 = f1_score(y_train_ext, train_predictions, average='weighted')
            if train_f1 > best_score:
                best_score = train_f1
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_features = selected_features_subset
                best_features_indices = selected_features_indices

    print(f"Best model: {best_model} with params: {best_params}, features: {best_features} and validation score: {best_score:.2f}")
    # Evaluate on test set
    test_predictions = best_model.predict(X_test_tot[:,best_features_indices])
    test_f1 = f1_score(y_test, test_predictions, average='weighted')
    print(f"Test F1 score for {score}: {test_f1:.2f}")





