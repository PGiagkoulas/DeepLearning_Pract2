# convlstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.utils import to_categorical
from matplotlib import pyplot
from itertools import product

from allModels import *
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
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

def load_const_dataset_group(group, prefix=''):
	X = load_file(prefix+group+'/X_'+group+'.txt')
	y = load_file(prefix + group + '/y_' + group + '.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_timeseries_dataset_group('train', prefix + 'HARDataset/')
	aux_trainX, aux_trainy = load_const_dataset_group('train', prefix + 'HARDataset/')
	print(">> Time series data shape: {0} , {1}".format(trainX.shape, trainy.shape))
	print(">> Constant data shape: {0} , {1}".format(aux_trainX.shape, aux_trainy.shape))
	# load all test
	testX, testy = load_timeseries_dataset_group('test', prefix + 'HARDataset/')
	aux_testX, aux_testy = load_const_dataset_group('test', prefix + 'HARDataset/')
	print(">> Time series data shape: {0} , {1}".format(testX.shape, testy.shape))
	print(">> Constant data shape: {0} , {1}".format(aux_testX.shape, aux_testy.shape))
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
	print(">> Final shapes of time series dataset: {0}, {1}, {2}, {3}".format(trainX.shape, trainy.shape, testX.shape, testy.shape))
	print(">> Final shapes of constant dataset: {0}, {1}, {2}, {3}".format(aux_trainX.shape, aux_trainy.shape, aux_testX.shape, aux_testy.shape))
	return trainX, trainy, testX, testy, aux_trainX, aux_trainy, aux_testX, aux_testy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
	# load data
	trainX, trainy, testX, testy, aux_trainX, aux_trainy, aux_testX, aux_testy = load_dataset() # TODO do we need the labels again?
	
	cfg_list=defineConfigurations()#list with all possible configurations
	gridresults=list()
	
	for cfg in cfg_list:
		scores = list()
		for r in range(repeats):
			score, aux_score = evaluate_cnnlstm_multi_model(trainX, trainy, testX, testy, aux_trainX, aux_trainy, aux_testX, aux_testy) # change if you want to run another model
			score = score * 100.0
			aux_score = aux_score * 100.00 # also remove everything regarding aux_score if a different model is used
			print('>#%d: LSTM = %.3f and Multi = %.3f' % (r+1, score, aux_score))
			scores.append(score)
		gridresults.append((cfg, scores))
	# summarize results
	summarize_gridresults(gridresults)
	
def summarize_gridresults(gridresults):
	for x in gridresults:
		print(x[0]) #prints the config
		summarize_results(x[1]) #prints the scores

# grid search configs
def grid_search(data, cfg_list, n_test):
	# evaluate configs
	scores = [repeat_evaluate(data, cfg, n_test) for cfg in cfg_list] #scores and cfg are dictionaries
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores


def defineConfigurations():
	cfgs = list()
	parameters = giveParameters()
	return list((dict(zip(parameters, x)) for x in product(*parameters.values())))
	
	
def giveParameters():
	#learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
	#verbose = [0]
	batch_size = [64]
	#optimiser= ['adam', 'sgd']
	epochs=[10]
	return dict(epochs=epochs, batch_size=batch_size)
	
    
    

# run the experiment
run_experiment()
