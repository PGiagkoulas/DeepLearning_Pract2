# convlstm model
from numpy import mean
from numpy import std
from matplotlib import pyplot
from itertools import product

from allModels import *
from all_utils import load_dataset
from fineTuning import fine_tune_convlstm

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
    batch_size = [128]
    optimizer = ['adam']
    epochs = [9]
    activation = ['relu']
    kernel_size_2D = [(1, 3)]
    kernel_size_1D = [3]
    filters = [64]
    pool_size = [2]
    loss = ['categorical_crossentropy']
    out_activation = ['softmax']
    dropout_rate = [0.5]
    return dict(verbose=verbose, epochs=epochs, batch_size=batch_size, activation=activation,
                kernel_size_2D=kernel_size_2D, kernel_size_1D=kernel_size_1D, filters=filters, pool_size=pool_size,
                loss=loss, out_activation=out_activation, optimizer=optimizer, dropout_rate=dropout_rate)

# run an experiment
def run_experiment(repeats=1):
    # load data
    trainX, trainy, testX, testy, aux_trainX, aux_trainy, aux_testX, aux_testy = load_dataset()
    grid = True

    cfg_list = defineConfigurations()  # list with all possible configurations

    gridresults = list()

    for cfg in cfg_list:
        scores = list()
        for r in range(repeats):
            score, aux_score, model = evaluate_cnnlstm_multi_model(trainX, trainy, testX, testy, aux_trainX, aux_trainy,
                                                            aux_testX, aux_testy, cfg,
                                                            r)  # change if you want to run another model
            score = score * 100.0
            aux_score = aux_score * 100.00  # also remove everything regarding aux_score if a different model is used
            print('>#%d: LSTM = %.3f and Multi = %.3f' % (r + 1, score, aux_score))
            scores.append(score)
            # save model and architecture to single file
            # name your saved model file
            model.save("cnnlstm.h5")
            print("Saved model to disk")
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


########################### summarize scores

def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


########################### run the experiment

if __name__== "__main__":
    run_experiment()