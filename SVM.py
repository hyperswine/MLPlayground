# Support Vector Machines

# Load scripts to clean and generate data
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, plot_roc_curve, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from feature_selection import y_classify, y_classify_five
from time import time

from auxiliary.data_clean2 import clean_data
import pandas as pd
import numpy as np

data = pd.read_csv('dataset/GSMArena_dataset_2020.csv', index_col=0)

data_features = data[
    ["oem", "launch_announced", "launch_status", "body_dimensions", "display_size", "comms_wlan", "comms_usb",
     "features_sensors", "platform_os", "platform_cpu", "platform_gpu", "memory_internal",
     "main_camera_single", "main_camera_video", "misc_price",
     "selfie_camera_video",
     "selfie_camera_single", "battery"]]

# Clean up the data into a trainable form.
df = clean_data(data_features)

# Load helper functions

# Now its time to split the data

y = df["misc_price"]
X = df.drop(["key_index", "misc_price"], axis=1)
# one numeric variable = screen size.
X = X.drop(["screen_size", "scn_bdy_ratio", "clock_speed", "battery"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=120, test_size=.3)

y5 = y.apply(y_classify_five)
X_train5, X_test5, y_train5, y_test5 = train_test_split(
    X, y5, random_state=100, test_size=.3)

y3 = y.apply(y_classify)
X_train3, X_test3, y_train3, y_test3 = train_test_split(
    X, y3, random_state=80, test_size=.3)

"""
Prelininary investigation into SVM performance. Here, we establish a baseline for how well an SVM should perform
in practice.
"""


# NOTE: default radial basis kernel
t0 = time()

svm_clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svm_clf.fit(X_train3, y_train3)

print(f"SVM-3 class classification finished in {time()-t0} seconds")

y_pred3 = svm_clf.predict(X_test3)
print("3 class accuracy: ", accuracy_score(y_test3, y_pred3))
print("Classification Report,\n", classification_report(y_test3, y_pred3))
print("Confusion Matrix\n", confusion_matrix(y_test3, y_pred3))
# plot_roc_curve(svm_clf, X_test, y_test) - Unfortunately only works for binary pipelines

print("\n\n========NEXT==================================\n\n")

t0 = time()
svm_clf.fit(X_train5, y_train5)
print(f"SVM-5 class classification finished in {time()-t0} seconds")

y_pred5 = svm_clf.predict(X_test5)
print("5 class accuracy: ", accuracy_score(y_test5, y_pred5))
print("Classification Report,\n", classification_report(y_test5, y_pred5))
print("Confusion Matrix\n", confusion_matrix(y_test5, y_pred5))

print("\n\n========NEXT==================================\n\n")

svr_clf = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=.2))
svr_clf.fit(X_train, y_train)
print("Support Vector Regression score: ", svr_clf.score(X_test, y_test))

# plot svm's
# plt.scatter(svm_clf.support_vectors_)


"""
It appears that 3-class classification would work the best. Accurate SV regression seems highly unlikely.
Hence, in the next stage we will build our own multiclass SVM classifier & utilize various nonlinear kernels.
NOTE: incomplete.
"""


class HyperSVM:
  """
  A support-vector machine with multiple kernel mappings for high
  dimensions & hinge loss. Uses a One-vs-One strategy for multiclass classification.
  """

  def __init__(self, dual=True, C=1, kernel="gaussian"):
    self.dual = dual
    self.svm_models = []
    self.C = C
    self.kernel = kernel

  def fit(self, X, y, y_class):
    """
    Fit m(m-1)/2 models in 'ovo' manner, given m classes
    NOTE: Maximum 30 classes = 435 models. Assumes that y_class contains all possible classes.
    """
    self.svm_models = []
    for ci in X.y_class:
      self.svm_models.append([self.fit_model(ci, cj, X, y)
                             for cj in X.y_class if ci != cj])

  def fit_model(self, i, j, X, y):
    """
    i - class 1
    j - class 2
    X - input data
    y - output classes
    """
    # filter the values from X, y that only contain classes i, j
    y = np.array(filter(lambda x: x == i or x == j, y))
    X = pd.DataFrame(X.iloc[y.index])

    # fit an svm
    svm_mod = SvmMod(i, j, kernel=self.kernel)
    svm_mod.fit(X, y, C=self.C)

    return svm_mod

  def predict(self, X):
    """
    Input data into all models & retreive an output and its associated 'score'.
    The class with the highest total score is the predicted class.

    Return - 1xm list of predictions.
    """
    # append the score for each output feature
    # scores_pair are stored as "(feature 1, feature 2)": "(score 1, score 2)" mappings
    scores_pair = {}
    prediction = []
    # main loop to predict all examples
    for index, example in X.iterrows():
      for svm_mod in self.svm_models:
        # each SvmMod should also have 2 feature names to take the 2 feature values of that row
        scores_pair[svm_mod.feature1] = svm_mod.predict(example)

      # retreive the feature with the highest total score & append to prediction
      prediction.append(np.argmax(scores_pair))
      # reset scores pair to predict next example
      scores_pair = {}

    return prediction

  def performance(self, y_test, y_pred):
    """
    Output accuracy score amongst other metrics
    """
    print("Classification scores:", classification_report(y_test, y_pred))


class SvmMod:
  """
  Representation of a binary, (currently) linear SVM.
  """

  def __init__(self, class1, class2, kernel="gaussian", iterations=100):
    self.classes = (class1, class2)
    self.a_star = []
    self.kernels = {"gaussian": gaussian_kern, "linear": lin_kern,
                    "poly": poly_kern, "sigmoid": hyperbolictan_kern}
    self.kern = kernel
    self.lagrag_multipliers = []
    self.y = []
    self.iterations = iterations

  def fit(self, X, y, C=1):
    """
    Expect - dataframe of two feature columns.
    """
    # convert y to +1/-1
    enc = OneHotEncoder()
    y = enc.fit_transform(y)
    def f(x): return -1 if x == 0 else 1
    y = f(y)

    # a* = argmax<1..n>(-1/2 * sum<i=1..n>( sum<j=1..n>( a[i]*a[j]*y[i]*y[j]*(K(X[i], X[j]) )) + sum<i=1..n>(a[i]))
    a = [1] * X.shape[0]
    converged = False

    # algorithm for iterative multiplier updates, (Ref: Kernel_Methods_handout: slide 56)
    # Could someone should change the 'a' updates & use gradients?
    # while not converged or self.iterations < 1000:
    #     converged = True
    #     sum_dotp = 0
    #
    #     # calculate sums of dot products, multipliers & class labels
    #     for i in range(X.shape[0]):
    #         for j in range(X.shape[0]):
    #             # could prob use gram matrix y[i]y[j]X[i]X[j]
    #             sum_dotp += a[i]*a[j]*y[i]*y[j]*(X[i].dot(X[j]))
    #         # check for convergence
    #         if np.sum(a * y) == 0 and sum_dotp <= 0:
    #             a[i] = a[i]+1
    #             converged = False
    #
    #     # multiply by -.5
    #     sum *= -.5
    #
    #     # add sum of multipliers to sum_dotp
    #     sum_dotp += np.sum(a)
    #
    #     self.iterations += 1

  def kernel_function(self, x_i, x_j):
    return self.kern(x_i, x_j)

  def partial_lagrangian(self, L, var):
    """
    Calculates the partial lagriangian derivative with respect to var.
    Minimizes hinge loss -> maximizes dual.

    Return - A binary SVM that outputs a score for each class according to its euclidean distance from the maximum-separating-hyperplane.
    NOTE: positive score for the side of the 'positive' (first) class, negative otherwise.
    """
    pass

  def max_margin(self):
    # a = X.shape[0] * [1]
    # gram_X = X.dot(X.T)
    # gram_X_unlabeled = X.T.dot(X.T)
    #
    # # NOTE: y = X.shape[0] * [+/-1]
    # a_mult = a * y
    #
    # remove a multiplier given constraint
    # a1 = a_mult[0]
    # a_mult[1:] *= -1;
    # a1 should be positive with respect to rest of the multipliers
    # if a1 < 0:
    #     a1 *= -1
    #     a_mult[1:] *= -1
    #
    # expand gram matrix & add multipliers
    # 0 <= a_i <= C for all i
    # Solve dual problem via gram matrix
    pass

  def predict(self, X):
    """
    Return - tuple containing scores for class1 & class2
    """
    for i in range(X.shape[0]):
      sum += self.lagrag_multipliers[i] * \
          self.y[i] * self.kernel_function(X[i], X)

    return sum


def gaussian_kern(x_i, y_i, sigma=.8, rbf=False, gamma=.5):
  return np.exp(gamma * np.linalg.norm((x_i - y_i) ** 2) if rbf else -np.linalg.norm((x_i - y_i) ** 2) / (2 * sigma))


def lin_kern(x_i, y_i, k=0):
  return x_i.dot(y_i) + k


def poly_kern(x_i, y_i, degree=3, k=0):
  return (x_i.T.dot(y_i) + k) ** degree


def hyperbolictan_kern(x_i, y_i, k=0):
  alpha = 1 / x_i.shape[1]  # 1/N features
  return np.tanh(alpha * x_i.T.dot(y_i) + k)
