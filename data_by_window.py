import tensorflow as tf
from ajout_colonne import ajout_colonne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


PATH_XTRAIN = "X_train.csv"
PATH_YTRAIN = "y_train.csv"
PATH_XTEST = "X_test.csv"


def make_windows(data, window_size=60, features=["consumption"], step=60):
    """ 
    data : dataframe
    window_size : int
    features : list of str
    step : int
    """
    data = data.copy()
    for feature in features:
        data[feature] = data[feature].astype(float)
    data = data[features].values
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    windows = []
    for i in range(0, data.shape[0]-window_size, step):
        window = data[i:i+window_size]
        windows.append(window)
    return tf.stack(windows)


def put_windows_back_together(windows, window_size, step=60):
    """ 
    windows : tensor
    window_size : int
    step : int
    """
    step = int(window_size / step)
    data = []
    for i in range(0, windows.shape[0], step):
        window = windows[i]
        data.append(window)
    return np.concatenate(data)


# test data_by_window
if __name__ == "__main__":
    # testing put_windows_back_together
    windows = tf.convert_to_tensor(np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5],
                                             [4, 5, 6], [5, 6, 7], [6, 7, 8],
                                             [7, 8, 9], [8, 9, 10], [9, 10, 11],
                                             [10, 11, 12], [11, 12, 13], [12, 13, 14]]))

    print(put_windows_back_together(windows, 3, 1))
