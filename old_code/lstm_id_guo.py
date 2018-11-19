import os

os.environ['KERAS_BACKEND'] = 'theano'


# os.environ["THEANO_FLAGS"] = "module=FAST_RUN,device=gpu0,floatX=float32"
import datetime
import argparse
import numpy as np
import scipy.io as sio
import random as rn
# from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import LSTM, Dense, TimeDistributed, Input, concatenate, Embedding, Bidirectional
# from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(123)
rn.seed(123)

data_file = os.path.join('AE_data.mat')
split_ratio = [0.7, 0.2, 0.1]  # splitting ratio of training, validation and test set
train_flag = True

batch_size = 50
n_epochs = 50
n_iter = 100  # Test set


class Args(object):
    def __init__(self):
        self.data_file = data_file
        self.split_ratio = [0.7, 0.2, 0.1]
        self.train_flag = True
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.model_name = 'model_002'
        self.case_name = 'run02_batch16_guo'
        self.teach_force = True


def arg_parser():
    args = Args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', type=str, help='The path for the AE data (.mat)')
    parser.add_argument('-s', '--split_ratio', type=tuple, help='The split ratio for training, valid and test set')
    parser.add_argument('-t', '--train_flag', action='store_true', help='Flag indicating if training or not')
    parser.add_argument('-b', '--batch_size', type=int, help='Size of the mini-batch')
    parser.add_argument('-n', '--n_epochs', type=int, help='number of epochs')
    parser.add_argument('-m', '--model_name', type=str, help='Name of the model to be implemented')
    parser.add_argument('-c', '--case_name', type=str, help='Name of the case (run)')
    parser.add_argument('-f', '--teach_force', action='store_true', help='Use teacher forcing or not')
    parser.add_argument('-l', '--load_model', type=str, help='The path to load the trained model for prediction')
    par = parser.parse_args()
    if par.data_file:
        args.data_file = par.data_file
    if par.split_ratio:
        args.split_ratio = par.split_ratio
    if par.train_flag:
        args.train_flag = par.train_flag
    if par.batch_size:
        args.batch_size = par.batch_size
    if par.n_epochs:
        args.n_epochs = par.n_epochs
    if par.model_name:
        args.model_name = par.model_name
    if par.case_name:
        args.case_name = '_' + par.case_name
    if par.teach_force:
        args.teach_force = par.teach_force
    if par.load_model:
        args.load_model = par.load_model
    """
    return args


def to_categorical(y, num_classes):
    y_onehot = np.zeros((y.shape[0], y.shape[1], num_classes))
    for id in range(y.shape[0]):
        y_onehot[id, np.arange(y.shape[1]), y[id]] = 1
    return y_onehot


def load_data(data_file, split_ratio, batch_size, teach_force=False):
    mat_data = sio.loadmat(data_file)
    dd = mat_data['dd']
    # tt = mat_data['tt']
    ll = mat_data['ll']

    N = dd.shape[0]  # Number of segments
    n_tra = (int(N * split_ratio[0]) // batch_size + 1) * batch_size
    n_val = int(N * split_ratio[1])

    train_x = dd[0: n_tra, :]
    train_y = ll[0: n_tra, :]
    valid_x = dd[n_tra: (n_tra + n_val), :]
    valid_y = ll[n_tra: (n_tra + n_val), :]
    test_x = dd[(n_tra + n_val):, :]
    test_y = ll[(n_tra + n_val):, :]

    # train_x = np.expand_dims(train_x, axis=-1)
    # valid_x = np.expand_dims(valid_x, axis=-1)
    # test_x  = np.expand_dims(test_x,  axis=-1)
    train_y = to_categorical(train_y, num_classes=2)
    valid_y = to_categorical(valid_y, num_classes=2)
    test_y = to_categorical(test_y, num_classes=2)

    n_step = train_x.shape[1]

    if teach_force:
        ll_vec = ll.reshape(-1)
        ll_pre_vec = np.zeros(ll_vec.shape)
        ll_pre_vec[1:] = ll_vec[:-1]
        ll_pre = ll_pre_vec.reshape(ll.shape)

        train_y_pre = ll_pre[0: n_tra, :]
        valid_y_pre = ll_pre[n_tra: (n_tra + n_val), :]
        test_y_pre = ll_pre[(n_tra + n_val):, :]

        train_y_pre = np.expand_dims(train_y_pre, axis=-1)
        valid_y_pre = np.expand_dims(valid_y_pre, axis=-1)
        test_y_pre = np.expand_dims(test_y_pre, axis=-1)

        return train_x, train_y, valid_x, valid_y, test_x, test_y, n_step, train_y_pre, valid_y_pre, test_y_pre

    else:
        return train_x, train_y, valid_x, valid_y, test_x, test_y, n_step


def model_001(n_step):
    hidden_size = 256

    X = Input(shape=(n_step, 1), name='X')
    LSTM_out1 = LSTM(units=hidden_size, return_sequences=True, input_shape=(n_step, 1))(X)
    LSTM_out2 = LSTM(units=hidden_size, return_sequences=True)(LSTM_out1)
    Y = TimeDistributed(Dense(2, activation='softmax'), name='Y')(LSTM_out2)
    model = Model(inputs=X, outputs=Y)
    model.summary()
    return model


def model_002(n_step):
    hidden_size = 256

    X = Input(shape=(n_step,), name='X')
    Embedded = Embedding(2000, 64, input_length=n_step)(X)
    # print Embedded
    Y_pre = Input(shape=(n_step, 1), name='Y_pre')
    Z = concatenate([Embedded, Y_pre])
    LSTM_out1 = LSTM(units=hidden_size, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(Z)
    LSTM_out2 = LSTM(units=hidden_size, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(LSTM_out1)
    Y = TimeDistributed(Dense(2, activation='softmax'), name='Y')(LSTM_out2)
    model = Model(inputs=[X, Y_pre], outputs=Y)
    model.summary()

    return model


def get_model(model_name, n_step):
    return globals()[model_name](n_step)


def perf_measure(y_true, y_pred):
    TP, FP, TN, FN = 0.0, 0.0, 0.0, 0.0

    pos_id = np.where(y_true == 1)[0]
    TP_id = np.where(y_pred[pos_id] == 1)[0]
    TP += TP_id.shape[0]

    FN_id = np.where(y_pred[pos_id] == 0)[0]
    FN += FN_id.shape[0]

    neg_id = np.where(y_true == 0)[0]
    TN_id = np.where(y_pred[neg_id] == 0)[0]
    TN += TN_id.shape[0]

    FP_id = np.where(y_pred[neg_id] == 1)[0]
    FP += FP_id.shape[0]

    pre = TP / (TP + FP + 1e-5)
    rec = TP / (TP + FN + 1e-5)
    acc = (TP + TN) / (TP + FP + FN + TN)
    f1 = (2 * pre * rec) / (pre + rec + 1e-5)

    return (pre, rec, acc, f1)


if __name__ == '__main__':
    # print(datetime.datetime.now())
    args = arg_parser()
    if args.teach_force:
        train_x, train_y, valid_x, valid_y, test_x, test_y, n_step, train_y_pre, valid_y_pre, test_y_pre_true = \
            load_data(args.data_file, args.split_ratio, args.batch_size, True)
    else:
        train_x, train_y, valid_x, valid_y, test_x, test_y, n_step = load_data(args.data_file, args.split_ratio,
                                                                               args.batch_size)

    # model = get_model(args.model_name, n_step)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    #
    # tb_dir = './' + args.model_name + args.case_name + '/logs'
    # tb = TensorBoard(tb_dir)
    # cp_file = './model_cp_' + args.model_name + args.case_name + '.hdf5'
    # cp = ModelCheckpoint(filepath=cp_file, verbose=1)

    if args.train_flag:
        model = get_model(args.model_name, n_step)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        """
        tb_dir = './' + args.model_name + args.case_name + '/logs'
        tb = TensorBoard(tb_dir)
        cp_file = './model_cp_' + args.model_name + args.case_name + '.hdf5'
        cp = ModelCheckpoint(filepath=cp_file, verbose=1)
        """
        if args.teach_force:
            model.fit({'X': train_x, 'Y_pre': train_y_pre}, {'Y': train_y}, batch_size=args.batch_size,
                      epochs=args.n_epochs, verbose=1,
                      validation_data=[{'X': valid_x, 'Y_pre': valid_y_pre}, {'Y': valid_y}])
        else:
            model.fit(train_x, train_y, batch_size=args.batch_size, epochs=args.n_epochs, verbose=1,
                      validation_data=(valid_x, valid_y))

        model.save('./' + args.model_name + args.case_name + '/final_' + args.model_name + '.hdf5')
        print('Done')

    else:
        # test_y = np.argmax(test_y, axis=2)

        model = load_model('./' + args.model_name + args.case_name + '/final_' + args.model_name + '.hdf5')
        y_pre_1 = valid_y[-1, -1, :]

        # test_y_pre = np.random.randint(0, 2, test_x.shape)
        test_y_pre = np.zeros(test_x.shape)

        test_y_pre = np.expand_dims(test_y_pre, -1)
        test_y_pre[0, 0, :] = max(y_pre_1)

        pre = np.zeros(n_iter)
        rec = np.zeros(n_iter)
        acc = np.zeros(n_iter)
        f1 = np.zeros(n_iter)

        # test_y_pre = test_y_pre_true

        for i in range(n_iter):
            # print(i)
            pred = np.argmax(model.predict({'X': test_x, 'Y_pre': test_y_pre}), axis=2)
            acc[i] = model.evaluate({'X': test_x, 'Y_pre': test_y_pre}, test_y)[1]
            test_y_pre = np.concatenate(([0], pred.reshape(-1)[:-1])).reshape(-1, n_step)

            test_y_pre = np.expand_dims(test_y_pre, -1)
            test_y_pre[0, 0, :] = max(y_pre_1)
            print(i, acc[i])

        plt.figure(1, figsize=(10, 6))
        plt.subplot(221)
        plt.plot(pre * 100)
        plt.xlabel('# Iterations')
        plt.ylabel('Precision (%)')

        plt.subplot(222)
        plt.plot(rec * 100)
        plt.xlabel('# Iterations')
        plt.ylabel('Recall (%)')

        plt.subplot(223)
        plt.plot(acc * 100)
        plt.xlabel('# Iterations')
        plt.ylabel('Accuracy (%)')

        plt.subplot(224)
        plt.plot(f1)
        plt.xlabel('# Iterations')
        plt.ylabel('f1')

        plt.savefig('result_init0.png', dpi=300)
