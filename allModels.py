from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dropout, Concatenate, ConvLSTM2D, Flatten, MaxPooling1D, Conv1D
from keras.layers import TimeDistributed
from keras.utils import to_categorical
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import csv        
    
# fit and evaluate a multi-input/multi-output ConvLSTM model
def evaluate_convlstm_multi_model(trainX, trainy, testX, testy, aux_trainX, aux_trainy, aux_testX, aux_testy, cfg, n):
    ## datastuff
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    n_aux_features = aux_trainX.shape[1]
    # reshape into subsequences (samples, time steps, rows, cols, channels)
    n_steps, n_length = 4, 32
    trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
    trainX = trainX[:50]
    trainy = trainy[:50]
    aux_trainX = aux_trainX[:50]
    aux_trainy = aux_trainy[:50]
    
    ## parameterstuff
    verbose = cfg.get('verbose') if ('verbose' in cfg) else 0
    epochs = cfg.get('epochs') if ('epochs' in cfg) else 25
    batch_size = cfg.get('batch_size') if ('batch_size' in cfg) else 64
    activation = cfg.get('activation') if ('activation' in cfg) else 'relu'
    kernel_size_2D = cfg.get('kernel_size_2D') if ('kernel_size_2D' in cfg) else (1,3)
    filters = cfg.get('filters') if ('filters' in cfg) else 64
    pool_size = cfg.get('pool_size') if ('pool_size' in cfg) else 2
    loss = cfg.get('loss') if ('loss' in cfg) else 'categorical_crossentropy'
    out_activation = cfg.get('out_activation') if ('out_activation' in cfg) else 'softmax'
    optimizer = cfg.get('optimizer') if ('optimizer' in cfg) else 'adam'
    dropout_rate = cfg.get('dropout_rate') if ('dropout_rate' in cfg) else 0.5
    
    ## modelstuff
    main_input = Input(shape=(n_steps, 1, n_length, n_features), dtype='float32', name='main_input')
    lstm_out = ConvLSTM2D(filters=filters, kernel_size=kernel_size_2D, activation=activation)(main_input)
    x = Dropout(rate=dropout_rate)(lstm_out)
    x = Flatten()(x)
    x = Dense(100, activation=activation)(x)
    # auxiliary output of CNNLSTM part
    auxiliary_output = Dense(n_outputs, activation=out_activation, name='aux_output')(x)

    # flatten output
    lstm_out_flat = Flatten()(lstm_out)
    auxiliary_input = Input(shape=(n_aux_features,), name='aux_input')
    # combine inputs
    x = Concatenate()([lstm_out_flat, auxiliary_input])
    # rest of the network
    x = Dense(128, activation=activation)(x)
    x = Dense(64, activation=activation)(x)
    x = Dense(32, activation=activation)(x)
    # final output
    main_output = Dense(6, activation=out_activation, name='main_output')(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    
    ## train/test
    history = model.fit(x=[trainX, aux_trainX], y=[trainy, aux_trainy], epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=([testX, aux_testX], [testy, aux_testy]))
    #print(history.history)
    #print(model.metrics_names)
    #print(model.summary())
    _, loss, aux_loss, accuracy, aux_acc = model.evaluate(x=[testX, aux_testX], y=[testy, aux_testy], batch_size=batch_size, verbose=1)
    saveResults("convLstmMulti", history, accuracy, aux_acc, loss, aux_loss, n)
    
    return accuracy, aux_acc

# fit and evaluate a multi-input/multi-output CNN-LSTM model
def evaluate_cnnlstm_multi_model(trainX, trainy, testX, testy, aux_trainX, aux_trainy, aux_testX, aux_testy, cfg, n):
    
    
    ## data stuff
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # reshape data into time steps of sub-sequences
    n_steps, n_length = 4, 32
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
    trainX = trainX[:50]
    trainy = trainy[:50]
    aux_trainX = aux_trainX[:50]
    aux_trainy = aux_trainy[:50]
    
    ## parameterstuff
    verbose = cfg.get('verbose') if ('verbose' in cfg) else 0
    epochs = cfg.get('epochs') if ('epochs' in cfg) else 25
    batch_size = cfg.get('batch_size') if ('batch_size' in cfg) else 64
    activation = cfg.get('activation') if ('activation' in cfg) else 'relu'
    kernel_size_1D = cfg.get('kernel_size_1D') if ('kernel_size_1D' in cfg) else 3
    filters = cfg.get('filters') if ('filters' in cfg) else 64
    pool_size = cfg.get('pool_size') if ('pool_size' in cfg) else 2
    loss = cfg.get('loss') if ('loss' in cfg) else 'categorical_crossentropy'
    out_activation = cfg.get('out_activation') if ('out_activation' in cfg) else 'softmax'
    optimizer = cfg.get('optimizer') if ('optimizer' in cfg) else 'adam'
    dropout_rate = cfg.get('dropout_rate') if ('dropout_rate' in cfg) else 0.5
    
    
    ## define model
    main_input = Input(shape=(None, n_length, n_features), dtype='float32', name='main_input')
    x = TimeDistributed(Conv1D(filters=filters, kernel_size=kernel_size_1D, activation='relu'))(main_input)
    x = TimeDistributed(Conv1D(filters=filters, kernel_size=kernel_size_1D, activation='relu'))(x)
    x = TimeDistributed(Dropout(dropout_rate))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=pool_size))(x)
    x = TimeDistributed(Flatten())(x)
    lstm_out = LSTM(units=100)(x)
    x = Dropout(rate=dropout_rate)(lstm_out)
    x = Dense(100, activation=activation)(x)
    auxiliary_output = Dense(n_outputs, activation=out_activation, name='aux_output')(x)
    # model = Model(inputs=main_input, outputs=auxiliary_output)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # auxiliary input shape
    num_features = aux_trainX.shape[1]
    # flatten output
    #lstm_out_flat = Flatten()(lstm_out)
    auxiliary_input = Input(shape=(num_features,), name='aux_input')
    # combine inputs
    x = Concatenate()([lstm_out, auxiliary_input])
    # rest of the network
    x = Dense(128, activation=activation)(x)
    x = Dense(64, activation=activation)(x)
    x = Dense(32, activation=activation)(x)
    # final output
    main_output = Dense(6, activation=out_activation, name='main_output')(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    
    # print(model.summary())
    history = model.fit(x=[trainX, aux_trainX], y=[trainy, aux_trainy], epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=([testX, aux_testX], [testy, aux_testy]))
    _, loss, aux_loss, accuracy, aux_acc = model.evaluate(x=[testX, aux_testX], y=[testy, aux_testy], batch_size=batch_size, verbose=1)
    
    saveResults("cnnLstmMulti", history, accuracy, aux_acc, loss, aux_loss, n)
    return accuracy, aux_acc

# fit and evaluate ConvLSTM a model
def evaluate_model(trainX, trainy, testX, testy, cfg):
	# define model
	verbose, epochs, batch_size = 0, 25, 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# reshape into subsequences (samples, time steps, rows, cols, channels)
	n_steps, n_length = 4, 32
	trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
	# trainX = trainX
	# trainy = trainy
	testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
	# define model
	model = Sequential()
	model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# fit and evaluate a CNN-LSTM model
def evaluate_cnnlstm_model(trainX, trainy, testX, testy, cfg):
	# define model
	verbose, epochs, batch_size = 0, 25, 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# reshape data into time steps of sub-sequences
	n_steps, n_length = 4, 32
	trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
	testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
	# define model
	model = Sequential()
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
	model.add(TimeDistributed(Dropout(0.5)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(LSTM(100))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

def saveResults(name, fittingProcess, accuracy, aux_accuracy, loss, aux_loss, n):
    loss_history = fittingProcess.history['main_output_loss']
    acc_history = fittingProcess.history['main_output_acc']
    lstm_loss_history = fittingProcess.history['aux_output_loss']
    lstm_acc_history = fittingProcess.history['aux_output_acc']
    val_loss_history = fittingProcess.history['val_main_output_loss']
    val_acc_history = fittingProcess.history['val_main_output_acc']
    val_lstm_loss_history = fittingProcess.history['val_aux_output_loss']
    val_lstm_acc_history = fittingProcess.history['val_aux_output_acc']
    
    with open( name + str(n) +'.csv', "w") as outfile:
        outfile.write("loss,accuracy,val_loss,val_acc")
        outfile.write("\n")
        for ind in range(len(loss_history)):
            outfile.write(str(loss_history[ind])+','+str(acc_history[ind])+','+str(val_loss_history[ind])+','+str(val_acc_history[ind]))
            outfile.write("\n")
    
    with open( name + '-lstm' + str(n) + '.csv', "w") as outfile:
        outfile.write("lstm_loss,lstm_accuracy,val_lstm_loss,val_lstm_acc")
        outfile.write("\n")
        for ind in range(len(loss_history)):
            outfile.write(str(lstm_loss_history[ind])+','+str(lstm_acc_history[ind])+','+str(val_lstm_loss_history[ind])+','+str(val_lstm_acc_history[ind]))
            outfile.write("\n")
            
    with open( name + '-modelevaluate' + str(n) + '.csv', "w") as outfile:
        outfile.write("loss,")
        outfile.write("accuracy")
        outfile.write("aux_loss,")
        outfile.write("aux_accuracy")
        outfile.write("\n")
        outfile.write(str(accuracy)+',')
        outfile.write(str(loss)+',')
        outfile.write(str(aux_accuracy))
        outfile.write(str(aux_loss))
        outfile.write("\n")
