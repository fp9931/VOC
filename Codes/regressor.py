from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut, GridSearchCV, KFold
import scipy.io
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split

f = scipy.io.loadmat("C:\\Users\\mimos\\Projects\\VOC\\Data\\Features.mat")

ID_pre = f["id"].T
label_pre = f["label"].T
features_pre = f["features"].T
alsfrs_pre = f["alsfrs"].T

[m, n] = ID_pre.shape

ID = np.reshape(ID_pre.T,(m*n,-1))
label = np.reshape(label_pre.T,(m*n,-1))
y = np.reshape(alsfrs_pre.T,(m*n,-1))
features = np.reshape(features_pre.T,(m*n,-1))
np.random.shuffle(features)

X = features.reshape(len(features), -1)
y = y.ravel()

y_true = []
y_pred = []
sample_ID = []
sample_label = []

subjects = np.unique(ID)
model = SVR()
param_grid = {
    'kernel' : ['linear'],
    'C': 1,
    'epsilon': 0.1
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

model.fit(X_train, y_train)

y_pred.append(np.mean(model.predict(X_test)))
y_true.append(np.mean(y_test))

y_t = np.array([yt.item() if isinstance(yt, np.ndarray) else yt for yt in y_true])
y_p = np.array([yp.item() if isinstance(yp, np.ndarray) else yp for yp in y_pred])

final_rmse = np.sqrt(mean_squared_error(y_t, y_p))

print(f'Nested LOSOCV RMSE: {final_rmse:.2f}')
