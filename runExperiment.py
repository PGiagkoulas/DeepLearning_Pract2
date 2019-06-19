# convlstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from itertools import product

from allModels import *
from utils import feature_selection

# how to use:

# in run experiment: set grid to true or not, set amount of repeats (cross validation)
# in run experiment: set the evaluate function, choose from the allModels file
# in giveParameters: set values for the hyper-parameters to be tested

# the current implementation will poop out (value of repeat) amount of csv's with results during each epoch,
# it will override these files per configuration.

# if grid is enabled it will print the best configuration at the end, use this anew with only those parameters in giveParameters to get the csv's of the best config

def giveParameters():
    # learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    verbose = [0]
    batch_size = [64, 128]
    optimizer = ['adam', 'sgd']
    epochs = [50]
    activation = ['relu', 'sigmoid', 'tanh']
    kernel_size_2D = [(1, 3)]
    kernel_size_1D = [3]
    filters = [64, 128]
    pool_size = [2]
    loss = ['categorical_crossentropy']
    out_activation = ['softmax']
    dropout_rate = [0, 0.25, 0.5]
    return dict(verbose=verbose, epochs=epochs, batch_size=batch_size, activation=activation,
                kernel_size_2D=kernel_size_2D, kernel_size_1D=kernel_size_1D, filters=filters, pool_size=pool_size,
                loss=loss, out_activation=out_activation, optimizer=optimizer, dropout_rate=dropout_rate)


# run an experiment
def run_experiment(repeats=10):
    # load data
    trainX, trainy, testX, testy, aux_trainX, aux_trainy, aux_testX, aux_testy = load_dataset()
    grid = True

    cfg_list = defineConfigurations()  # list with all possible configurations

    gridresults = list()

    for cfg in cfg_list:
        scores = list()
        for r in range(repeats):
            score, aux_score = evaluate_stacked_lstm_multi_model(trainX, trainy, testX, testy, aux_trainX, aux_trainy,
                                                            aux_testX, aux_testy, cfg,
                                                            r)  # change if you want to run another model
            score = score * 100.0
            aux_score = aux_score * 100.00  # also remove everything regarding aux_score if a different model is used
            print('>#%d: LSTM = %.3f and Multi = %.3f' % (r + 1, score, aux_score))
            scores.append(score)
        gridresults.append((cfg, scores))

    if grid:
        summarize_gridresults(gridresults)
    printBestGrid(gridresults)


########################### grid functions

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


########################### summarize scores

def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


########################### run the experiment

run_experiment()
