# -*- coding: utf-8 -*-
"""
Script for a Decision Tree classifier on the GSM Phone Arena Dataset
"""

from auxiliary.data_clean2 import clean_data
import pandas as pd
import numpy as np
import graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from sklearn.tree import DecisionTreeClassifier, plot_tree
from feature_selection import y_classify_five, y_classify

# Load up dataset 1: gsmarena
data = pd.read_csv('dataset/GSMArena_dataset_2020.csv', index_col=0)

data_features = data[["oem", "launch_announced", "launch_status", "body_dimensions", "display_size", "comms_wlan", "comms_usb",
                "features_sensors", "platform_os", "platform_cpu", "platform_gpu", "memory_internal",
                "main_camera_single", "main_camera_video", "misc_price",
                "selfie_camera_video",
                "selfie_camera_single", "battery"]]

# Clean up the data into a trainable form.
df = clean_data(data_features)

y = df["misc_price"]
X = df.drop(["misc_price"], axis=1)

# convert to categorical data
lab_enc = preprocessing.LabelEncoder()
# y = lab_enc.fit_transform(y)
y = y.apply(y_classify)

# Split data into train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=0)

dtClf = tree.DecisionTreeClassifier()
dtClf = dtClf.fit(X_train, y_train)
y_pred = dtClf.predict(X_test)
print(r"Decision tree score for is {}".format(accuracy_score(y_pred, y_test)))

# Results is about 17.6%, which is far below satisfactory. Probably means that the tree has overfit the training data
# Try to tune hyperparameters with Grid Search
param_grid = {'min_samples_leaf': np.arange(2,30,2), 'criterion': ['gini', 'entropy']}
grid_tree = GridSearchCV(tree.DecisionTreeClassifier(random_state=0), param_grid,cv=10,return_train_score=True)
grid_tree.fit(X_train, y_train)

estimator = grid_tree.best_estimator_
y_pred = grid_tree.predict(X_test)
tree_performance = accuracy_score(y_test, y_pred)
print(r"New decision tree score for is {} (Grid search should be the same as before)".format(tree_performance))

# Maybe Grid Search isnt the best, try randomized search next
grid_tree = RandomizedSearchCV(tree.DecisionTreeClassifier(random_state=0), param_grid)
grid_tree.fit(X_train, y_train)

estimator = grid_tree.best_estimator_
y_pred = grid_tree.predict(X_test)
tree_performance = accuracy_score(y_test, y_pred)
print(r"New decision tree score for is {} (Randomized Search should be 78%)".format(tree_performance))
