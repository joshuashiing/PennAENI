import os
import datetime
import argparse
import numpy as np
import scipy.io as sio
import random as rn
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, TimeDistributed, Input, concatenate, Embedding, Lambda, Dropout, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.backend import squeeze
import matplotlib.pyplot as plt

# Random seeds
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(123)
rn.seed(123)
tf.set_random_seed(123)


data_file = os.path.join('AE_label_data.mat')
split_ratio = [0.8, 0.2]
n0 = 2048
shrink = 4
h0 = 16

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


# # Define IoU metric
# def mean_iou(y_true, y_pred):
#     prec = []
#     for t in np.arange(0.5, 1.0, 0.05):
#         y_pred_ = tf.to_int32(y_pred > t)
#         score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
#         K.get_session().run(tf.local_variables_initializer())
#         with tf.control_dependencies([up_opt]):
#             score = tf.identity(score)
#         prec.append(score)
#     return K.mean(K.stack(prec), axis=0)


def cnn_model_01(n0, sr, h0):
    
    # n0: length of sample
    # sr: shrink ratio of length
    # h0: thickness of the first conv


    X = Input(shape=(n0, 1, 1), name='X')

    c1a = Conv2D(h0, (3, 1), activation='relu', padding='same')(X)
    c1b = Conv2D(h0, (3, 1), activation='relu', padding='same')(c1a)
    p1 = MaxPooling2D((sr, 1))(c1b)

    c2a = Conv2D(h0 * 2, (3, 1), activation='relu', padding='same')(p1)
    c2b = Conv2D(h0 * 2, (3, 1), activation='relu', padding='same')(c2a)
    p2 = MaxPooling2D((sr, 1))(c2b)

    c3a = Conv2D(h0 * 4, (3, 1), activation='relu', padding='same')(p2)
    c3b = Conv2D(h0 * 4, (3, 1), activation='relu', padding='same')(c3a)
    p3 = MaxPooling2D((sr, 1))(c3b)

    c4a = Conv2D(h0 * 8, (3, 1), activation='relu', padding='same')(p3)
    c4b = Conv2D(h0 * 8, (3, 1), activation='relu', padding='same')(c4a)
    p4 = MaxPooling2D((sr, 1))(c4b)

    c5a = Conv2D(h0 * 16, (3, 1), activation='relu', padding='same')(p4)
    c5b = Conv2D(h0 * 16, (3, 1), activation='relu', padding='same')(c5a)

    u6a = Conv2DTranspose(h0 * 8, (5, 1), strides=(sr, 1), padding='same')(c5b)
    u6b = concatenate([u6a, c4b])
    c6a = Conv2D(h0 * 8, (3, 1), activation='relu', padding='same')(u6b)
    c6b = Conv2D(h0 * 8, (3, 1), activation='relu', padding='same')(c6a)

    u7a = Conv2DTranspose(h0 * 4, (5, 1), strides=(sr, 1), padding='same')(c6b)
    u7b = concatenate([u7a, c3b])
    c7a = Conv2D(h0 * 4, (3, 1), activation='relu', padding='same')(u7b)
    c7b = Conv2D(h0 * 4, (3, 1), activation='relu', padding='same')(c7a)

    u8a = Conv2DTranspose(h0 * 4, (5, 1), strides=(sr, 1), padding='same')(c7b)
    u8b = concatenate([u8a, c2b])
    c8a = Conv2D(h0 * 2, (3, 1), activation='relu', padding='same')(u8b)
    c8b = Conv2D(h0 * 2, (3, 1), activation='relu', padding='same')(c8a)

    u9a = Conv2DTranspose(h0 * 2, (5, 1), strides=(sr, 1), padding='same')(c8b)
    u9b = concatenate([u9a, c1b])
    c9a = Conv2D(h0, (3, 1), activation='relu', padding='same')(u9b)
    c9b = Conv2D(h0, (3, 1), activation='relu', padding='same')(c9a)

    Y = Conv2D(2, (1, 1), activation='softmax', padding='same')(c9b)

    model = Model(inputs=X, outputs=Y)
    model.summary()

    return model


def cnn_model_02(n0, sr, h0, dr):
    # n0: length of sample
    # sr: shrink ratio of length
    # h0: thickness of the first conv
    # dr: dropout rate

    X = Input(shape=(n0, 1, 1), name='X')

    c1a = Conv2D(h0, (3, 1), activation='relu', padding='same')(X)
    c1b = Conv2D(h0, (3, 1), activation='relu', padding='same')(c1a)
    p1 = MaxPooling2D((sr, 1))(c1b)
    p1 = Dropout(dr)(p1)

    c2a = Conv2D(h0 * 2, (3, 1), activation='relu', padding='same')(p1)
    c2b = Conv2D(h0 * 2, (3, 1), activation='relu', padding='same')(c2a)
    p2 = MaxPooling2D((sr, 1))(c2b)
    p2 = Dropout(dr)(p2)

    c3a = Conv2D(h0 * 4, (3, 1), activation='relu', padding='same')(p2)
    c3b = Conv2D(h0 * 4, (3, 1), activation='relu', padding='same')(c3a)
    p3 = MaxPooling2D((sr, 1))(c3b)
    p3 = Dropout(dr)(p3)

    c4a = Conv2D(h0 * 8, (3, 1), activation='relu', padding='same')(p3)
    c4b = Conv2D(h0 * 8, (3, 1), activation='relu', padding='same')(c4a)
    p4 = MaxPooling2D((sr, 1))(c4b)
    p4 = Dropout(dr)(p4)

    c5a = Conv2D(h0 * 16, (3, 1), activation='relu', padding='same')(p4)
    c5a = Dropout(dr)(c5a)
    c5b = Conv2D(h0 * 16, (3, 1), activation='relu', padding='same')(c5a)

    u6a = Conv2DTranspose(h0 * 8, (5, 1), strides=(sr, 1), padding='same')(c5b)
    u6b = concatenate([u6a, c4b])
    u6b = Dropout(dr)(u6b)
    c6a = Conv2D(h0 * 8, (3, 1), activation='relu', padding='same')(u6b)
    c6b = Conv2D(h0 * 8, (3, 1), activation='relu', padding='same')(c6a)

    u7a = Conv2DTranspose(h0 * 4, (5, 1), strides=(sr, 1), padding='same')(c6b)
    u7b = concatenate([u7a, c3b])
    u7b = Dropout(dr)(u7b)
    c7a = Conv2D(h0 * 4, (3, 1), activation='relu', padding='same')(u7b)
    c7b = Conv2D(h0 * 4, (3, 1), activation='relu', padding='same')(c7a)

    u8a = Conv2DTranspose(h0 * 4, (5, 1), strides=(sr, 1), padding='same')(c7b)
    u8b = concatenate([u8a, c2b])
    u8b = Dropout(dr)(u8b)
    c8a = Conv2D(h0 * 2, (3, 1), activation='relu', padding='same')(u8b)
    c8b = Conv2D(h0 * 2, (3, 1), activation='relu', padding='same')(c8a)

    u9a = Conv2DTranspose(h0 * 2, (5, 1), strides=(sr, 1), padding='same')(c8b)
    u9b = concatenate([u9a, c1b])
    u9b = Dropout(dr)(u9b)
    c9a = Conv2D(h0, (3, 1), activation='relu', padding='same')(u9b)
    c9b = Conv2D(h0, (3, 1), activation='relu', padding='same')(c9a)

    Y = Conv2D(2, (1, 1), activation='softmax', padding='same')(c9b)

    model = Model(inputs=X, outputs=Y)
    model.summary()

    return model



train_x, train_y, test_x, test_y = load_data(data_file, n0, split_ratio)


# model = cnn_model_01(n0, shrink, h0)
model = cnn_model_02(n0, shrink, h0, dr=0.5)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])


es = EarlyStopping(patience=30, verbose=1)
cp = ModelCheckpoint('model_cnn_id_02.h5', verbose=1, save_best_only=True)
# tb = TensorBoard(os.path.join('.', 'cnn_id_01'))
tb = TensorBoard(os.path.join('.', 'cnn_id_02'))
rl = ReduceLROnPlateau(factor=0.3, patience=20, verbose=1)

model.fit(train_x, train_y, batch_size=64, epochs=1000, verbose=1, callbacks=[tb, cp], validation_data=[test_x, test_y])
