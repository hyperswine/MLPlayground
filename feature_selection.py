"""
This script allows us to gain insights on what features matter the most.
It uses a random forest classifier to do this.
"""
from auxiliary.data_clean2 import clean_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from threading import Thread

def y_classify(y):
    if y > 700:
        return 2
    elif y >= 300 and y <= 700:
        return 1

    return 0


def y_classify_five(y):
    if y > 1000:
        return 4
    elif y > 700 and y <= 1000:
        return 3
    elif y > 450 and y <= 700:
        return 2
    elif y > 200 and y <= 450:
        return 1

    return 0


def feature_selection(df, expressiveness='F'):
    """
    Output the features that are the most important in the feature dataframe
    """
    y = df["misc_price"]
    # NOTE: 3 classes default. Switch this to 'y_classify+_five' for 5 classes.
    # 3 classes seems to result in a higher performance with both classifiers
    y5 = y.apply(y_classify_five)
    y = y.apply(y_classify)

    # plot numbers of class labels
    if expressiveness != 'P':
        print("Number of Labels for 3-class\n\tLabel\tNumber")
        for i in range(3):
            print(f"\t{i}\t{np.sum(y.apply(lambda x: x==i))}")

        print("Number of Labels for 5-class\n\tLabel\tNumber")
        for i in range(5):
            print(f"\t{i}\t{np.sum(y5.apply(lambda x: x == i))}")
    if expressiveness != 'P':
        y.to_csv('output_csv')

    X = df.drop(["key_index", "misc_price", "rom", "selfie_camera_video"], axis=1)
    rand_forest = RandomForestClassifier(n_estimators=500, n_jobs=-1)

    rand_forest.fit(X, y)

    if expressiveness != 'P':
        for feature, score in zip(X, rand_forest.feature_importances_):
            print(feature, score)

    # use the random forest to predict
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=120, test_size=.3)
    X_train5, X_test5, y_train5, y_test5 = train_test_split(X, y5, random_state=120, test_size=.3)

    rand_forest.fit(X_train, y_train)
    y_pred = rand_forest.predict(X_test)
    print("Accuracy of RF classifier", accuracy_score(y_test, y_pred))

    # use a neural net (note numeric input)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=10)

    clf.fit(X_train5, y_train5)
    y_pred5 = clf.predict(X_test5)
    print("Accuracy of Multiple Layer Perceptron (Categorical Only)", accuracy_score(y_test5, y_pred5))

    # MLP without numeric input
    X_cat = X.drop(["body_dimensions", "screen_size", "scn_bdy_ratio", "clock_speed", "battery"], axis=1)
    X_trainC, X_testC, y_trainC, y_testC = train_test_split(X_cat, y, random_state=120, test_size=.3)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=5)

    clf.fit(X_trainC, y_trainC)
    y_predC = clf.predict(X_testC)
    print("Accuracy of Multiple Layer Perceptron (Numeric In)", accuracy_score(y_testC, y_predC))

    # k-NN with k = 1...10
    for i in range(1, 11):
        clf = KNeighborsClassifier(n_neighbors=i, weights='distance')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Accuracy of NN with k = {i}", accuracy_score(y_test, y_pred))

    # Naive Bayes
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy of Gassian Naive Bayes @ default settings", accuracy_score(y_test, y_pred))

    clf = MultinomialNB(alpha=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy of Multinomial NB @ alpha = 1 (laplace)", accuracy_score(y_test, y_pred))

    clf = ComplementNB(alpha=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy of Complement NB @ alpha = 1 (laplace)", accuracy_score(y_test, y_pred))


def plot_pairs(df):
    """
    Plots of pair points between features.
    """
    for featureX in df.columns:
        for featureY in df.columns:
            if featureX == 'key_index' or featureY == 'key_index':
                continue

            if featureX != featureY:
                plt.figure()
                plt.plot(df[featureX], df[featureY])
                plt.savefig(f'plots/pair_plot_{featureX}_{featureY}.png')


if __name__ == "__main__":
    data = pd.read_csv('dataset/GSMArena_dataset_2020.csv',
                       index_col=0)

    data_features = data[
        ["oem", "launch_announced", "launch_status", "body_dimensions", "display_size", "comms_wlan", "comms_usb",
         "features_sensors", "platform_os", "platform_cpu", "platform_gpu", "memory_internal",
         "main_camera_single", "main_camera_video", "misc_price",
         "selfie_camera_video",
         "selfie_camera_single", "battery"]]

    df = clean_data(data_features)

    expressiveness = ""
    #
    # expressiveness = input("Turn off expressiveness? y/n : ")
    # if expressiveness.lower() == 'y':
    #     expressiveness = 'P'

    feature_selection(df, expressiveness)
    # plot_pairs(df) # creates quite a lot of plots (a few hundred) in /plots
