# -*- coding: utf-8 -*-
"""
Plot and explore dataset

"""

from auxiliary.data_clean2 import clean_data
from feature_selection import y_classify_five, y_classify

import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt



# Load up dataset 1: gsmarena
data = pd.read_csv('dataset/GSMArena_dataset_2020.csv', index_col=0)



data_features = data[["oem", "launch_announced", "launch_status", "body_dimensions", "display_size", "comms_wlan", "comms_usb",
                "features_sensors", "platform_os", "platform_cpu", "platform_gpu", "memory_internal",
                "main_camera_single", "main_camera_video", "misc_price",
                "selfie_camera_video",
                "selfie_camera_single", "battery"]]

new_df = pd.DataFrame(data_features)
#check how m,any missing values there are
print(new_df.isnull().sum())
msno.matrix(new_df)
plt.savefig('./figs/original')
fig = msno.heatmap(new_df)
plt.tight_layout()
# fig.set_size_inches(6, 4, forward=True)
plt.savefig('./figs/original_heatmap')


df = clean_data(data_features)
y = df["misc_price"]
X = df.drop(["misc_price"], axis=1)

df['misc_price'] = y.apply(y_classify)
# two across all feature combinations
# sns.pairplot(df, hue="misc_price",vars=["platform_cpu", "platform_gpu", "display_size"])


col1 = ['key_index', 'oem', 'launch_announced', 'body_dimensions',
       'features_sensors', 'platform_gpu', 'main_camera_single',
       'main_camera_video', 'misc_price']
col2 = ['misc_price', 'selfie_camera_video',
       'selfie_camera_single', 'battery', 'clock_speed', 'screen_size',
       'scn_bdy_ratio', 'rom', 'ram']
fig, axes = plt.subplots(4,2,sharex=False,sharey=False, figsize=(10, 9))
bp_dict = pd.DataFrame(df, columns=col1).boxplot(
by="misc_price", ax=axes, 
return_type='both',
patch_artist = True,
)
fig.tight_layout()
fig, axes = plt.subplots(4,2,sharex=False,sharey=False, figsize=(10, 9))
bp_dict = pd.DataFrame(df, columns=col2).boxplot(
by="misc_price", ax=axes, 
return_type='both',
patch_artist = True,
)
fig.tight_layout()

