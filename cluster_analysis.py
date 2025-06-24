"""
Model & analyze potential clusters and 'Nearest-Neighbors'.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from auxiliary.data_clean2 import clean_data
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Load up Data
data = pd.read_csv('dataset/GSMArena_dataset_2020.csv', index_col=0)

data_features = data[["oem", "launch_announced", "launch_status", "body_dimensions", "display_size", "comms_wlan", "comms_usb",
                "features_sensors", "platform_os", "platform_cpu", "platform_gpu", "memory_internal",
                "main_camera_single", "main_camera_video", "misc_price",
                "selfie_camera_video",
                "selfie_camera_single", "battery"]]

df = clean_data(data_features)

# dataset without labels
X1 = df.drop(["key_index", "misc_price"], axis=1)
# dataset without labels and inexpressive features
X2 = df.drop(["key_index", "misc_price", "rom", "selfie_camera_video"], axis=1)
# dataset with output classes though unlabelled for clustering
X3 = df.drop(["key_index"], axis=1)

"""
K-Means for cluster analysis. Note relatively high dimensional data may result in some extraordinarily high
distance measures. PCA & dimensionality reduction should be done to produce more meaningful results.
"""

from sklearn.cluster import KMeans

for i in [3,5,10]:
    # fit the model with i clusters
    clf = KMeans(n_clusters=i, random_state=0).fit(X1)
    # get the centres of each cluster
    print(f"=========================== K:{i} ===========================")
    print("Cluster centres:", clf.cluster_centers_, "\n")
    print("----------------------------------------------------------")
    print(f"Squared distances of samples to centres (inertia) of k={i}\n", clf.inertia_)
    print("----------------------------------------------------------")
    print("Iterations ran\n", clf.n_iter_)

# PCA for numeric (quantitative data = dimensions, display size, cpu clock, ram, battery)
pca = PCA() # keep all components for now
pca.fit(X1[["body_dimensions", "screen_size", "scn_bdy_ratio", "clock_speed", "battery"]])

print("Variance explained by each of the components [vector]:", pca.explained_variance_ratio_)
print("Euclidean norms of each component (projection)", pca.singular_values_)

"""
The results show that 'body dimensions' may be a replacement for all numeric 'size-based' features.
It does make sense that the larger in physical size the device, the more 'battery' or 'screen' it would have.
With a ratio var of 0.99, it seems reasonable to simply use 'body_dimensions' as the only numeric feature.
"""

# KD-Tree
from sklearn.neighbors import KDTree

X4 = X1.drop(["body_dimensions", "screen_size", "scn_bdy_ratio", "clock_speed", "battery"], axis=1)

# chebyshev distance for categorical features (integer valued, e.g. 'canberra distance' unavailable for kd-tree)
kd_tree = KDTree(X4, leaf_size=2, metric='chebyshev')
# kernel density (gaussian). Note: BFS might be slower if very dense
gaussian_density = kd_tree.kernel_density(X4[:100], h=.1, breadth_first=False)
# kernel density (exponential).
exp_density = kd_tree.kernel_density(X4[:100], h=.1, kernel='exponential')

print("Gaussian density\n")
print(gaussian_density)
print(f"======================================================")
print("Exponential density\n")
print(exp_density)

# distance to 5 closest neighbors for first 50 instances
d, _ = kd_tree.query(X4[:50], k=5)

print(f"======================================================")
print("Distance of 5 closest neighbors for first 50 instances\n")
print(d)

"""
It appears the distances aren't too large as expected with qualitative data.
However, the kernel density suggests that []
"""
