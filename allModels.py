from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dropout, Concatenate, ConvLSTM2D, Flatten, MaxPooling1D, Conv1D
from keras.layers import TimeDistributed
from keras.layers import Input, Embedding, LSTM, Dense, CuDNNLSTM
from keras.models import Model
import csv
from keras.callbacks import ModelCheckpoint

from all_utils import saveResults, unfold_general_hyperparameters, residual_lstm_layers


# fit and evaluate a multi-input/multi-output ConvLSTM model
def evaluate_convlstm_multi_model(trainX, trainy, testX, testy, aux_trainX, aux_trainy, aux_testX, aux_testy, cfg, n):
    ## datastuff
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    n_aux_features = aux_trainX.shape[1]
    # reshape into subsequences (samples, time steps, rows, cols, channels)
    n_steps, n_length = 4, 32
    trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))

    ## parameterstuff
    verbose, epochs, batch_size, activation, \
    filters, pool_size, loss, out_activation, \
    optimizer, dropout_rate = unfold_general_hyperparameters(cfg)
    kernel_size_2D = cfg.get('kernel_size_2D') if ('kernel_size_2D' in cfg) else (1, 3)

    ## modelstuff
    main_input = Input(shape=(n_steps, 1, n_length, n_features), dtype='float32', name='main_input')
    lstm_out = ConvLSTM2D(filters=filters, kernel_size=kernel_size_2D, activation=activation, name='convlstm_0')(main_input)
    x = Dropout(rate=dropout_rate)(lstm_out)
    x = Flatten()(x)
    x = Dense(100, activation=activation, name='dense_0')(x)
    # auxiliary output of CNNLSTM part
    auxiliary_output = Dense(n_outputs, activation=out_activation, name='aux_output')(x)

    # flatten output
    lstm_out_flat = Flatten()(lstm_out)
    auxiliary_input = Input(shape=(n_aux_features,), name='aux_input')
    # combine inputs
    x = Concatenate()([lstm_out_flat, auxiliary_input])
    # rest of the network
    x = Dense(96, activation=activation, name='dense_1')(x)
    x = Dropout(rate=dropout_rate)(x)
    x = Dense(64, activation=activation, name='dense_2')(x)
    x = Dropout(rate=dropout_rate)(x)
    x = Dense(32, activation=activation, name='dense_3')(x)
    x = Dropout(rate=dropout_rate)(x)
    # final output
    main_output = Dense(n_outputs, activation=out_activation, name='main_output')(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    # checkpoint
    checkpointer = ModelCheckpoint(filepath='conv_lstm.h5', monitor = 'val_main_output_acc', verbose = 1, save_best_only = True, save_weights_only = False, mode = 'auto', period = 1)
    ## train/test
    history = model.fit(x=[trainX, aux_trainX], y=[trainy, aux_trainy], epochs=epochs, batch_size=batch_size,
                        verbose=verbose, validation_data=([testX, aux_testX], [testy, aux_testy]), callbacks=[checkpointer])
    # print(history.history)
    # print(model.metrics_names)
    # print(model.summary())
    _, loss, aux_loss, accuracy, aux_acc = model.evaluate(x=[testX, aux_testX], y=[testy, aux_testy],
                                                          batch_size=batch_size, verbose=1)
    # saveResults("convLstmMulti", history, accuracy, aux_acc, loss, aux_loss, n)

    return accuracy, aux_acc, model


# fit and evaluate a multi-input/multi-output CNN-LSTM model
def evaluate_cnnlstm_multi_model(trainX, trainy, testX, testy, aux_trainX, aux_trainy, aux_testX, aux_testy, cfg, n):
    ## data stuff
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # reshape data into time steps of sub-sequences
    n_steps, n_length = 4, 32
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))

    ## parameterstuff
    verbose, epochs, batch_size, activation, \
    filters, pool_size, loss, out_activation, \
    optimizer, dropout_rate = unfold_general_hyperparameters(cfg)
    kernel_size_1D = cfg.get('kernel_size_1D') if ('kernel_size_1D' in cfg) else 3

    ## define model
    main_input = Input(shape=(None, n_length, n_features), dtype='float32', name='main_input')
    x = TimeDistributed(Conv1D(filters=filters, kernel_size=kernel_size_1D, activation='relu'), name='tdconv1d_0')(main_input)
    x = TimeDistributed(Conv1D(filters=filters, kernel_size=kernel_size_1D, activation='relu'), name='tdconv1d_1')(x)
    x = TimeDistributed(Dropout(dropout_rate))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=pool_size))(x)
    x = TimeDistributed(Flatten())(x)
    lstm_out = LSTM(units=100, name='lstm_0')(x)
    x = Dropout(rate=dropout_rate)(lstm_out)
    x = Dense(100, activation=activation, name='dense_0')(x)
    auxiliary_output = Dense(n_outputs, activation=out_activation, name='aux_output')(x)
    num_features = aux_trainX.shape[1]
    auxiliary_input = Input(shape=(num_features,), name='aux_input')
    # combine inputs
    x = Concatenate()([lstm_out, auxiliary_input])
    # rest of the network
    x = Dense(128, activation=activation, name='dense_1')(x)
    x = Dense(64, activation=activation, name='dense_2')(x)
    x = Dense(32, activation=activation, name='dense_3')(x)
    # final output
    main_output = Dense(n_outputs, activation=out_activation, name='main_output')(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    # checkpoint
    checkpointer = ModelCheckpoint(filepath='cnn_lstm.h5', monitor='val_main_output_acc', verbose=1,
                                   save_best_only=True, save_weights_only=False, mode='auto', period=1)
    history = model.fit(x=[trainX, aux_trainX], y=[trainy, aux_trainy], epochs=epochs, batch_size=batch_size,
                        verbose=verbose, validation_data=([testX, aux_testX], [testy, aux_testy]), callbacks=[checkpointer])
    _, loss, aux_loss, accuracy, aux_acc = model.evaluate(x=[testX, aux_testX], y=[testy, aux_testy],
                                                          batch_size=batch_size, verbose=1)

    # saveResults("cnnLstmMulti", history, accuracy, aux_acc, loss, aux_loss, n)
    return accuracy, aux_acc, model


# fit and evaluate a multi-input/multi-output residual LSTM model
def evaluate_res_lstm_multi_model(trainX, trainy, testX, testy, aux_trainX, aux_trainy, aux_testX, aux_testy, cfg, n):
    verbose, epochs, batch_size = 0, 25, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # define model
    main_input_res = Input(shape=(n_timesteps, n_features), name='residual_lstm_input')
    lstm_out = residual_lstm_layers(main_input_res, rnn_width=100, rnn_depth=4, rnn_dropout=0.2)
    dense_out = Dense(100, activation='relu')(lstm_out)
    auxiliary_output = Dense(n_outputs, activation='softmax', name='aux_output')(dense_out)
    num_features = aux_trainX.shape[1]
    auxiliary_input = Input(shape=(num_features,), name='aux_input')
    # combine inputs
    x = Concatenate()([lstm_out, auxiliary_input])
    # rest of the network
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    # final output
    main_output = Dense(n_outputs, activation='softmax', name='main_output')(x)
    model = Model(inputs=[main_input_res, auxiliary_input], outputs=[main_output, auxiliary_output])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # checkpoint
    checkpointer = ModelCheckpoint(filepath='res_lstm.h5', monitor='val_main_output_acc', verbose=1,
                                   save_best_only=True, save_weights_only=False, mode='auto', period=1)
    history = model.fit(x=[trainX, aux_trainX], y=[trainy, aux_trainy], epochs=epochs, batch_size=batch_size,
                        verbose=verbose, validation_data=([testX, aux_testX], [testy, aux_testy]), callbacks=[checkpointer])
    _, loss, aux_loss, accuracy, aux_acc = model.evaluate(x=[testX, aux_testX], y=[testy, aux_testy],
                                                          batch_size=batch_size, verbose=1)
    saveResults("resLstmMulti", history, accuracy, aux_acc, loss, aux_loss, n)
    return accuracy, aux_acc


# fit and evaluate a multi-input/multi-output stacked LSTM model
def evaluate_stacked_lstm_multi_model(trainX, trainy, testX, testy, aux_trainX, aux_trainy, aux_testX, aux_testy, cfg,
                                      n):
    verbose, epochs, batch_size = 0, 25, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    main_input_stacked = Input(shape=(n_timesteps, n_features), name='stacked_lstm_input')
    lstm_out0 = CuDNNLSTM(100, return_sequences=True)(main_input_stacked)
    lstm_out1 = CuDNNLSTM(100, return_sequences=True)(lstm_out0)
    lstm_out2 = CuDNNLSTM(100, return_sequences=False)(lstm_out1)
    lstm_out3 = CuDNNLSTM(100, return_sequences=False)(lstm_out2)
    drop_out0 = Dropout(rate=0.5)(lstm_out3)
    # lstm_out4=LSTM(9, activation='tanh',recurrent_dropout=0.2, dropout=0.2, return_sequences=False)(lstm_out3)
    Dense_out = Dense(100, activation='relu')(drop_out0)
    auxiliary_output = Dense(n_outputs, activation='softmax', name='aux_output')(Dense_out)
    num_features = aux_trainX.shape[1]
    auxiliary_input = Input(shape=(num_features,), name='aux_input')
    # combine inputs
    x = Concatenate()([lstm_out2, auxiliary_input])
    # rest of the network
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    # final output
    main_output = Dense(n_outputs, activation='softmax', name='main_output')(x)
    model = Model(inputs=[main_input_stacked, auxiliary_input], outputs=[main_output, auxiliary_output])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # checkpoint
    checkpointer = ModelCheckpoint(filepath='res_lstm.h5', monitor='val_main_output_acc', verbose=1,
                                   save_best_only=True, save_weights_only=False, mode='auto', period=1)
    history = model.fit(x=[trainX, aux_trainX], y=[trainy, aux_trainy], epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=([testX, aux_testX], [testy, aux_testy]), callbacks=[checkpointer])
    _, loss, aux_loss, accuracy, aux_acc = model.evaluate(x=[testX, aux_testX], y=[testy, aux_testy], batch_size=batch_size, verbose=1)
    saveResults("stackedLstmMulti", history, accuracy, aux_acc, loss, aux_loss, n)
    return accuracy, aux_acc


# fit and evaluate ConvLSTM a model
def evaluate_convlst_model(trainX, trainy, testX, testy, cfg):
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
    model.add(
        ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
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
    model.add(
        TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None, n_length, n_features)))
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


# fit and evaluate a residual LSTM model
def evaluate_res_lstm_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 25, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    input_res = Input(shape=(n_timesteps, n_features), name='residual_lstm_input')
    lstm_out = residual_lstm_layers(input_res, rnn_width=100, rnn_depth=8, rnn_dropout=0.2)
    Dense_out = Dense(100, activation='relu')(lstm_out)
    output = Dense(n_outputs, activation='softmax')(Dense_out)
    model = Model(inputs=input_res, outputs=output)
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # fit the network
    model.fit(x=trainX, y=trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    _, accuracy = model.evaluate(x=testX, y=testy, batch_size=batch_size, verbose=verbose)
    return accuracy


# fit and evaluate a stacked LSTM model
def evaluate_stacked_lstm_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 25, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    input_stacked = Input(shape=(n_timesteps, n_features), name='stacked_lstm_input')
    lstm_out0 = LSTM(9, activation='tanh', recurrent_dropout=0.2, dropout=0.2, return_sequences=True)(input_stacked)
    lstm_out1 = LSTM(9, activation='tanh', recurrent_dropout=0.2, dropout=0.2, return_sequences=True)(lstm_out0)
    lstm_out2 = LSTM(9, activation='tanh', recurrent_dropout=0.2, dropout=0.2, return_sequences=True)(lstm_out1)
    lstm_out3 = LSTM(9, activation='tanh', recurrent_dropout=0.2, dropout=0.2, return_sequences=True)(lstm_out2)
    lstm_out4 = LSTM(9, activation='tanh', recurrent_dropout=0.2, dropout=0.2, return_sequences=False)(lstm_out3)
    Dense_out = Dense(100, activation='relu')(lstm_out4)
    output = Dense(n_outputs, activation='softmax')(Dense_out)
    model = Model(inputs=input_stacked, outputs=output)
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # fit the network
    model.fit(x=trainX, y=trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    _, accuracy = model.evaluate(x=testX, y=testy, batch_size=batch_size, verbose=verbose)
    return accuracy

MODELS = {
    'conv_lstm': evaluate_convlstm_multi_model,
    'cnn_lstm': evaluate_cnnlstm_multi_model,
    'res_lstm': evaluate_res_lstm_multi_model,
    'stacked_lstm': evaluate_stacked_lstm_multi_model
}