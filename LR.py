"""
This script is on learning a Linear Regression model.
Before writing up our own algorithms, it made sense to use the pre-existing algorithms from libraries such as sklearn.
This provides a baseline for the performance of LR on our dataset to match.

Preliminary Considerations
There were many considerations to be made. The first regarding hyper-parameters and high-dimensional data.
It was vital to not overthink the first few steps.
"""

from sklearn.model_selection import train_test_split
import pandas as pd
from auxiliary.data_clean2 import clean_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Open Dataset
data = pd.read_csv('dataset/GSMArena_dataset_2020.csv', index_col=0)

# Some Insight
# data.info()
# data.head()

# NOTE: conflicting features 'main_camera_dual', 'comms_nfc', 'battery_charging', 'selfie_camera_video' resulting in
# many null cols.
data_features = data[
    ["oem", "launch_announced", "launch_status", "body_dimensions", "display_size", "comms_wlan", "comms_usb",
     "features_sensors", "platform_os", "platform_cpu", "platform_gpu", "memory_internal",
     "main_camera_single", "main_camera_video", "misc_price",
     "selfie_camera_video",
     "selfie_camera_single", "battery"]]

df = clean_data(data_features)

df.dropna(inplace=True)
df.reset_index(drop=True)

# Now its time to split the data

y = df["misc_price"]
X = df.drop(["key_index", "misc_price", "rom", "selfie_camera_video"], axis=1)

# Train & test split. 70-30 split for the preliminary split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=120, test_size=.3)

""" 
Baseline performance of sklearn algorithms.
"""

lr_model = LinearRegression()

# Batch-train LR
lr_model.fit(X_train, y_train)

# Test the model & retrieve predictions
y_pred = lr_model.predict(X_test)

print("r2 score: ", r2_score(y_test, y_pred))
print("MSE: ", mean_squared_error(y_test, y_pred))
print("\n")

# Test categorical data only
X = X.drop(["body_dimensions", "screen_size", "scn_bdy_ratio", "clock_speed", "battery"], axis=1)

# Categorical input/ouput
X_trainC, X_testC, y_trainC, y_testC = train_test_split(
    X, y, random_state=120, test_size=.3)

lr_model.fit(X_trainC, y_trainC)
y_predC = lr_model.predict(X_testC)
print("r2 score (Categorical Input): ", r2_score(y_testC, y_predC))
print("MSE (Categorical Input): ", mean_squared_error(y_testC, y_predC))
print("\n")

"""
Investigating Linear Regression in more detail.
Now we investigate LR in more depth by learning our own models and regularizing.
"""


# Set up class & method defs for LR batch

class LinReg:
    """
    A streamlined linear regression object for batch learning.
    """

    def __init__(self, epochs=1000, n_features=20):
        self.theta_pred = 0
        self.epochs = epochs
        self.n_features = n_features
        self.t0 = 5
        self.t1 = 50
        self.weights = []

    def learn_rate(self, t):
        return self.t0 / (t + self.t1)

    def fit_batch(self, X, y):
        """
        Use the normal eq. to find the weights. Note high computational complexity so not optimal
        for use on complete dataset.
        """
        self.theta_pred = \
            np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def fit_stochastic(self, X, y):
        """
        Stochastic gradient descent.
        NOTE1. a bit of 'boiler-plate' code in gradient descent, should probably specify a 'gradient_desc' method
        and have 'L1' & 'L2' options instead.
        """
        # initialize random weights according to gaussian distr.
        self.weights = np.random.randn(self.n_features, 1)
        prev_weights = self.weights
        n = X.shape[0]

        # Ref: 'Hands on Machine Learning ..' Gueron. (p 127) for general stochastic grad. descent algorithm.
        for epoch in range(self.epochs):
            for i in range(n):
                rand_index = np.random.randint(n)
                x_i = X[rand_index:rand_index + 1]
                y_i = y[rand_index:rand_index + 1]
                grad = 2 * x_i.T.dot(x_i.dot(self.weights) - y_i)
                self.weights += -self.learn_rate(epoch * n + i) * grad

                # conditional end
                if epoch > 1 and np.linalg.norm(np.abs(self.weights - prev_weights)) < 10:
                    return

                prev_weights = self.weights

    def L1_fit(self, X, y, lmb=1, cond_end=10):
        """
        Fit according to Lasso regression. NOTE: uses self.weights.
        """
        # new objective function -> argmin(y-Xw)^T(y-Xw) + lmb*L1Norm(w)
        self.weights = np.random.randn(self.n_features, 1)
        prev_weights = self.weights
        n = X.shape[0]

        for epoch in range(self.epochs):
            for i in range(n):
                rand_index = np.random.randint(n)
                x_i = X[rand_index:rand_index + 1]
                y_i = y[rand_index:rand_index + 1]
                grad = 2 * x_i.T.dot(x_i.dot(self.theta_pred) - y_i)
                penalty = lmb * np.linalg.norm(self.theta_pred, ord=1)
                self.weights += -self.learn_rate(epoch * n + i) * grad + [self.weights.shape[0]*[penalty]]

                # conditional end
                if epoch > 1 and np.linalg.norm(np.abs(self.weights - prev_weights)) < cond_end:
                    return

                prev_weights = self.weights

    def L2_fit(self, X, y, lmb=1, closed_form=True, cond_end=10):
        """
        Fit according to Ridge regression.
        """
        if closed_form:
            S1 = np.linalg.inv(X.T.dot(X) + lmb * np.identity())
            S2 = X.T.dot(y)
            self.theta_pred = S1.dot(S2)
        else:
            # objective function -> argmin(y-Xw)^T(y-Xw) + lmb*L2Norm(w)**2. ref slides (1).
            self.weights = np.random.randn(self.n_features, 1)
            prev_weights = self.weights
            n = X.shape[0]

            for epoch in range(self.epochs):
                for i in range(n):
                    rand_index = np.random.randint(n)
                    x_i = X[rand_index:rand_index + 1]
                    y_i = y[rand_index:rand_index + 1]
                    grad = 2 * x_i.T.dot(x_i.dot(self.weights) - y_i)
                    penalty = lmb * (np.linalg.norm(self.weights, ord=2) ** 2)
                    self.weights += -self.learn_rate(epoch * n + i) * grad + [self.weights.shape[0]*[penalty]]

                    # conditional end if |w*(i+1) - w*(i)| changes less than an arbitary value, say 10.
                    if epoch > 1 and np.linalg.norm(np.abs(self.weights - prev_weights)) < cond_end:
                        return

                    prev_weights = self.weights

    def predict_batch(self, X):
        """
        For batch fit & closed form L2.
        """
        return X.dot(self.theta_pred)

    def predict_stochastic(self, X):
        """
        For gradient descent, stochastic, L1, L2.
        """
        return X.dot(self.weights)

    def performance(self, y_test, y_pred, batch=True):
        print('Coefficients: \n', self.theta_pred if batch else self.weights)

        print('Mean squared error: %.2f'
              % mean_squared_error(y_test, y_pred))

        print('Coefficient of determination: %.2f'
              % r2_score(y_test, y_pred))

    def plot(self, X, y, batch=True):
        plt.figure()
        plt.plot(X, y)

        plt.figure()
        plt.plot(self.theta_pred if batch else self.weights)


# Train LinReg Batch
lin_reg = LinReg(n_features=X_train.shape[1])

lin_reg.fit_batch(X_train, y_train)
y_pred = lin_reg.predict_batch(X_test)
lin_reg.performance(y_test, y_pred)


# # Perform 4-fold cross-validation on the datasets
# kf_4 = KFold(n_splits=4, shuffle=True)
# kf_4.get_n_splits(X)

# for train, test in kf_4.split(X):
#     lin_reg.fit_batch(X[train], y[train])
#     y_pred = lin_reg.predict(X[test])
#     print(lin_reg.performance(y[test], y_pred))

# # Perform 10-fold cross-validation on the datasets
# kf_10 = KFold(n_splits=10, shuffle=True)
# kf_10.get_n_splits(X)

# for train, test in kf_10.split(X):
#     lin_reg.fit_stochastic(X[train], y[train])
#     y_pred = lin_reg.predict(X[test])
#     print(lin_reg.performance(y[test], y_pred))


# Regularize with L1:
# lin_reg.L1_fit(X_train, y_train)
# y_pred = lin_reg.predict_stochastic(X_test)
# lin_reg.performance(y_test, y_pred)
#
# # Regularize with L2
# lin_reg.L2_fit(X_train, y_train, closed_form=False)
# y_pred = lin_reg.predict_stochastic(X_test)
# lin_reg.performance(y_test, y_pred)
#
# Plot the coefficients (vector), and plot each L1, L2, & batch-normal equation accuracy.
# Plot accuracy of L1 & L2 over epochs = 100,200,300,400,500...1000,5000.
#
# plt.plot(lin_reg.theta_pred)
# plt.plot(lin_reg.weights)

# train l1_reg over 100,200...5000 epochs & store performance1

# train l1_reg over 100,200...5000 epochs & store performance2

# plot performance1 & performance2 on same figure
