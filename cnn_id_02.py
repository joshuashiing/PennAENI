import os
import datetime
import argparse
import numpy as np
import scipy.io as sio
import random as rn
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, TimeDistributed, Input, concatenate, Embedding, Lambda, \
    Dropout, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.backend import squeeze
import matplotlib.pyplot as plt


def load_data(data_file, n0, sr):

    # n0: length of sample
    # sr: split ratio

    mat_data = sio.loadmat(data_file)
    d = mat_data['d']
    l = mat_data['l']
    N = np.floor(l.shape[0] / n0).astype(int)
    N1 = np.ceil(N * sr[0]).astype(int)    # training set size
    N2 = (N - N1).astype(int)              # test set size
    train_x = d[: (N1 * n0), :].reshape(N1, n0)
    train_y = l[: (N1 * n0), :].reshape(N1, n0)
    test_x = d[(N1 * n0) : (N * n0), :].reshape(N2, n0)
    test_y = l[(N1 * n0) : (N * n0), :].reshape(N2, n0)

    # Pre-processing
    train_x = np.expand_dims(train_x, axis=2)
    train_x = np.expand_dims(train_x, axis=3)
    train_y = np.expand_dims(train_y, axis=2)
    train_y = np.expand_dims(train_y, axis=3)
    train_y = to_categorical(train_y, num_classes=2)
    test_x = np.expand_dims(test_x, axis=2)
    test_x = np.expand_dims(test_x, axis=3)
    test_y = np.expand_dims(test_y, axis=2)
    test_y = np.expand_dims(test_y, axis=3)
    test_y = to_categorical(test_y, num_classes=2)

    return train_x, train_y, test_x, test_y