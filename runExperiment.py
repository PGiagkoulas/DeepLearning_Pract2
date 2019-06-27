# convlstm model
from numpy import mean
from numpy import std
from matplotlib import pyplot

import argparse

import allModels
import all_utils 
from fineTuning import fine_tune_convlstm

# how to use:

# in run experiment: set grid to true or not, set amount of repeats (cross validation)
# in run experiment: set the evaluate function, choose from the allModels file
# in giveParameters: set values for the hyper-parameters to be tested

# the current implementation will poop out (value of repeat) amount of csv's with results during each epoch,
# it will override these files per configuration.

# if grid is enabled it will print the best configuration at the end, use this anew with only those parameters in giveParameters to get the csv's of the best config

parser = argparse.ArgumentParser(description='Run a model.')
parser.add_argument('--arch', type=str, default='conv_lstm',
					help='Architecture to use.')
parser.add_argument('--grid', type=bool, default=False,
					help='Wether or not to do grid search.')
parser.add_argument('--repeats', type=int, default=5,
					help='Amount of repeats to get mean score.')
args = parser.parse_args()


                


# run an experiment
def run_experiment(repeats=1):
    # load data
    trainX, trainy, testX, testy, aux_trainX, aux_trainy, aux_testX, aux_testy = all_utils.load_dataset()
    grid = True

    if args.grid:
        cfg_list = all_utils.defineConfigurations()  # list with all possible configurations
    else:
        cfg_list = [all_utils.giveSingleParameters()] # when only 1 configuration needs to be run

    repeats = args.repeats
    # get architecture from arguments
    model = allModels.MODELS[args.arch]
    
    print(">> Running {0}".format(args.arch))
    gridresults = list()


    
    for cfg in cfg_list:
        scores = list()
        for r in range(repeats):
            score, aux_score = model(trainX, trainy, testX, testy, aux_trainX, aux_trainy,
                                                            aux_testX, aux_testy, cfg,
                                                            args.grid, r)  # change if you want to run another model
            score = score * 100.0
            aux_score = aux_score * 100.00  # also remove everything regarding aux_score if a different model is used
            print('>#%d: LSTM = %.3f and Multi = %.3f' % (r + 1, aux_score, score))
            scores.append(score)
        gridresults.append((cfg, scores))

    if grid:
        all_utils.summarize_gridresults(gridresults)
    all_utils.printBestGrid(gridresults)


########################### summarize scores

def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


########################### run the experiment

if __name__== "__main__":
    run_experiment()
