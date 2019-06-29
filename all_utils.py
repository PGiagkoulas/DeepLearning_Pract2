from keras.layers import CuDNNLSTM
from keras.layers import Lambda
from keras.layers.merge import add
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.utils import to_categorical
from itertools import product



########################### grid functions

def giveSingleParameters():
    verbose = 0
    batch_size = 128
    optimizer = 'adam'
    epochs = 9
    activation = 'relu'
    kernel_size_2D = (1, 3)
    kernel_size_1D = 3
    filters = 64
    pool_size = 2
    loss = 'categorical_crossentropy'
    out_activation = 'softmax'
    dropout_rate = 0.5
    return dict(verbose=verbose, epochs=epochs, batch_size=batch_size, activation=activation,
                kernel_size_2D=kernel_size_2D, kernel_size_1D=kernel_size_1D, filters=filters, pool_size=pool_size,
                loss=loss, out_activation=out_activation, optimizer=optimizer, dropout_rate=dropout_rate)

# give the parameters for grid search
def giveParameters():
    verbose = [0]
    batch_size = [64, 128]
    optimizer = ['adam', 'sgd']
    epochs = [15]
    activation = ['relu', 'sigmoid', 'tanh']
    kernel_size_2D = [(1, 3)]
    kernel_size_1D = [3]
    filters = [64]
    pool_size = [2]
    loss = ['categorical_crossentropy']
    out_activation = ['softmax']
    dropout_rate = [0, 0.25, 0.5]
    return dict(verbose=verbose, epochs=epochs, batch_size=batch_size, activation=activation,
                kernel_size_2D=kernel_size_2D, kernel_size_1D=kernel_size_1D, filters=filters, pool_size=pool_size,
                loss=loss, out_activation=out_activation, optimizer=optimizer, dropout_rate=dropout_rate)

def summarize_gridresults(gridresults):
    for x in gridresults:
        print(x[0])  # prints the config
        summarize_results(x[1])  # prints the scores


def defineConfigurations():
    cfgs = list()
    parameters = giveParameters()
    return list((dict(zip(parameters, x)) for x in product(*parameters.values())))


def printBestGrid(gridresults):
    best = gridresults[0]
    for x in gridresults:
        if mean(x[1]) > mean(best[1]):
            best = x
    print("best result:")
    print(best[0])
    summarize_results(best[1])

########################### summarize scores

def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# helper function to save model results to csv files
def saveResults(name, fittingProcess, accuracy, aux_accuracy, loss, aux_loss, n):
    loss_history = fittingProcess.history['main_output_loss']
    acc_history = fittingProcess.history['main_output_acc']
    lstm_loss_history = fittingProcess.history['aux_output_loss']
    lstm_acc_history = fittingProcess.history['aux_output_acc']
    val_loss_history = fittingProcess.history['val_main_output_loss']
    val_acc_history = fittingProcess.history['val_main_output_acc']
    val_lstm_loss_history = fittingProcess.history['val_aux_output_loss']
    val_lstm_acc_history = fittingProcess.history['val_aux_output_acc']

    with open(name + str(n) + '.csv', "w") as outfile:
        outfile.write("loss,accuracy,val_loss,val_acc")
        outfile.write("\n")
        for ind in range(len(loss_history)):
            outfile.write(
                str(loss_history[ind]) + ',' + str(acc_history[ind]) + ',' + str(val_loss_history[ind]) + ',' + str(
                    val_acc_history[ind]))
            outfile.write("\n")

    with open(name + '-lstm' + str(n) + '.csv', "w") as outfile:
        outfile.write("lstm_loss,lstm_accuracy,val_lstm_loss,val_lstm_acc")
        outfile.write("\n")
        for ind in range(len(loss_history)):
            outfile.write(str(lstm_loss_history[ind]) + ',' + str(lstm_acc_history[ind]) + ',' + str(
                val_lstm_loss_history[ind]) + ',' + str(val_lstm_acc_history[ind]))
            outfile.write("\n")

    with open(name + '-modelevaluate' + str(n) + '.csv', "w") as outfile:
        outfile.write("lstm_loss,")
        outfile.write("lstm_accuracy,")
        outfile.write("loss,")
        outfile.write("accuracy,")
        outfile.write("\n")
        outfile.write(str(loss) + ',')
        outfile.write(str(accuracy) + ',')
        outfile.write(str(aux_loss)+',')
        outfile.write(str(aux_accuracy))
        outfile.write("\n")

# helper function to assign hyperparameters
def unfold_general_hyperparameters(cfg):
    verbose = cfg.get('verbose') if ('verbose' in cfg) else 0
    epochs = cfg.get('epochs') if ('epochs' in cfg) else 25
    batch_size = cfg.get('batch_size') if ('batch_size' in cfg) else 64
    activation = cfg.get('activation') if ('activation' in cfg) else 'relu'
    # kernel_size_1D = cfg.get('kernel_size_1D') if ('kernel_size_1D' in cfg) else 3
    filters = cfg.get('filters') if ('filters' in cfg) else 64
    pool_size = cfg.get('pool_size') if ('pool_size' in cfg) else 2
    loss = cfg.get('loss') if ('loss' in cfg) else 'categorical_crossentropy'
    out_activation = cfg.get('out_activation') if ('out_activation' in cfg) else 'softmax'
    optimizer = cfg.get('optimizer') if ('optimizer' in cfg) else 'adam'
    dropout_rate = cfg.get('dropout_rate') if ('dropout_rate' in cfg) else 0.5

    return verbose, epochs, batch_size, activation, filters, pool_size, loss, out_activation, optimizer, dropout_rate

# helper function for PCA feature selection
def feature_selection(all_aux_trainX, all_aux_testX):
    data = np.concatenate((all_aux_trainX, all_aux_testX), axis=0)
    scaler = MinMaxScaler(feature_range=[0, 1])
    data_rescaled = scaler.fit_transform(data)
    pca = PCA(n_components=175)
    dataset = pca.fit_transform(data_rescaled)
    aux_trainX = dataset[0:all_aux_trainX.shape[0]][:]
    aux_testX = dataset[all_aux_trainX.shape[0]:][:]
    return aux_trainX, aux_testX

# residual lstm layer generator
def residual_lstm_layers(input, rnn_width, rnn_depth, rnn_dropout):
    x = input
    for i in range(rnn_depth):
        return_sequences = i < rnn_depth - 1
        # if the return_sequences is true, which means that this LSTM layer will output 3D instead of 2D(By default LSTM output 2D(the last time step of sequence)).
        # have the LSTM output a value for each time step in the input data.
        # x_rnn = LSTM(rnn_width, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=return_sequences)(x)
        x_rnn = CuDNNLSTM(rnn_width, return_sequences=return_sequences)(x)
        if return_sequences:

            if i > 0 or input.shape[-1] == rnn_width:
                x = add([x, x_rnn])
            else:

                x = x_rnn
        else:
            # Last layer does not return sequences, just the last element
            # so we select only the last element of the previous output.
            def slice_last(x):
                return x[..., -1, :]

            x = add([Lambda(slice_last)(x), x_rnn])
            # x = TimeDistributed(Dense(6, activation='softmax'))(x)
    return x

########################### preparing the data


# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded


# load a dataset group, such as train or test
def load_timeseries_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
    # body acceleration
    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
    # body gyroscope
    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_' + group + '.txt')
    return X, y


def load_const_dataset_group(group, prefix=''):
    X = load_file(prefix + group + '/X_' + group + '.txt')
    y = load_file(prefix + group + '/y_' + group + '.txt')
    return X, y


# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_timeseries_dataset_group('train', prefix + 'HARDataset/')
    all_aux_trainX, aux_trainy = load_const_dataset_group('train', prefix + 'HARDataset/')
    print(">> Time series data shape: {0} , {1}".format(trainX.shape, trainy.shape))
    print(">> Constant data shape: {0} , {1}".format(all_aux_trainX.shape, aux_trainy.shape))
    # load all test
    testX, testy = load_timeseries_dataset_group('test', prefix + 'HARDataset/')
    all_aux_testX, aux_testy = load_const_dataset_group('test', prefix + 'HARDataset/')
    print(">> Time series data shape: {0} , {1}".format(testX.shape, testy.shape))
    print(">> Constant data shape: {0} , {1}".format(all_aux_testX.shape, aux_testy.shape))
    # feature selection on constant features
    aux_trainX, aux_testX = feature_selection(all_aux_trainX, all_aux_testX)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    aux_trainy = aux_trainy - 1
    aux_testy = aux_testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    aux_trainy = to_categorical(aux_trainy)
    aux_testy = to_categorical(aux_testy)
    print(">> Final shapes of time series dataset: {0}, {1}, {2}, {3}".format(trainX.shape, trainy.shape, testX.shape,
                                                                              testy.shape))
    print(">> Final shapes of constant dataset: {0}, {1}, {2}, {3}".format(aux_trainX.shape, aux_trainy.shape,
                                                                           aux_testX.shape, aux_testy.shape))
    return trainX, trainy, testX, testy, aux_trainX, aux_trainy, aux_testX, aux_testy

