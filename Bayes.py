import seaborn as sns

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from auxiliary.data_clean2 import clean_data
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import preprocessing
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

df.dropna(inplace=True)
df.reset_index(drop=True)

y = df["misc_price"]
X = df.drop(["key_index", "misc_price"], axis=1)

# convert to categorical data
lab_enc = preprocessing.LabelEncoder()
# y = lab_enc.fit_transform(y)

sns.set_style("whitegrid")
sns.boxplot(x=y) #Box plot
plt.show()

y = y.apply(y_classify)#e.g. y > 700: return 2; 700 >= y >= 300: return 1; y < 300: return 0


# Split data into train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)


clf = BernoulliNB()
model = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('\n- - - BernoulliNB')
print(classification_report(y_test, y_pred))
print('0: price < 300, 1:700 >= price >= 300, 2: price > 700')
print('precise accuracy = ',accuracy_score(y_pred, y_test))

clf = GaussianNB()
model = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('\n- - - GaussianNB')
print(classification_report(y_test, y_pred))
print('0: price < 300, 1:700 >= price >= 300, 2: price > 700')
print('precise accuracy = ',accuracy_score(y_pred, y_test))

clf = MultinomialNB()
model = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('\n- - - MultinomialNB')
print(classification_report(y_test, y_pred))
print('0: price < 300, 1:700 >= price >= 300, 2: price > 700')
print('precise accuracy = ',accuracy_score(y_pred, y_test))

clf = ComplementNB()
model = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('\n- - - ComplementNB')
print(classification_report(y_test, y_pred))
print('0: price < 300, 1:700 >= price >= 300, 2: price > 700')
print('precise accuracy = ',accuracy_score(y_pred, y_test))