from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dropout, Concatenate, ConvLSTM2D, Flatten
from keras.layers import TimeDistributed
from keras.utils import to_categorical
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model


def evaluate_multi_model(trainX, trainy, testX, testy, aux_trainX, aux_trainy, aux_testX, aux_testy, cfg):
    ## datastuff
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    n_aux_features = aux_trainX.shape[1]
    # reshape into subsequences (samples, time steps, rows, cols, channels)
    n_steps, n_length = 4, 32
    trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
    verbose, epochs, batch_size = 0, 25, 64
    trainX = trainX[:100]
    trainy = trainy[:100]
    aux_trainX = aux_trainX[:100]
    aux_trainy = aux_trainy[:100]
    
    ## parameterstuff
    verbose = cfg.get('verbose') if ('verbose' in cfg) else 0
    epochs = cfg.get('epochs') if ('epochs' in cfg) else 25
    batch_size = cfg.get('batch_size') if ('batch_size' in cfg) else 64
    #print(verbose)
    #print(epochs)
    #print(batch_size)
	#verbose, epochs, batch_size): = 0, 25, 64
    
    ## modelstuff
    main_input = Input(shape=(n_steps, 1, n_length, n_features), dtype='float32', name='main_input')
    lstm_out = ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu')(main_input)
    x = Dropout(rate=0.5)(lstm_out)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    # auxiliary output of CNNLSTM part
    auxiliary_output = Dense(n_outputs, activation='softmax', name='aux_output')(x)
    # for cnnlstm only
    # main_output = Dense(n_outputs, activation='sigmoid', name='main_output')(x)
    # model = Model(inputs=main_input, outputs=auxiliary_output)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=0)

    # flatten output
    lstm_out_flat = Flatten()(lstm_out)
    auxiliary_input = Input(shape=(n_aux_features,), name='aux_input')
    # combine inputs
    x = Concatenate()([lstm_out_flat, auxiliary_input])
    # rest of the network
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    # final output
    main_output = Dense(6, activation='softmax', name='main_output')(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    ## train/test
    model.fit(x=[trainX, aux_trainX], y=[trainy, aux_trainy], epochs=epochs, batch_size=batch_size, verbose=verbose)
    #print(model.metrics_names)
    #print(model.summary())
    _, _, _, accuracy, aux_acc = model.evaluate(x=[testX, aux_testX], y=[testy, aux_testy], batch_size=batch_size, verbose=1)
    return accuracy, aux_acc
    
    
