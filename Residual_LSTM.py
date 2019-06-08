import numpy as np
from keras.models import Model
from keras.models import Input
from keras.layers import LSTM, Bidirectional
from keras.layers import Lambda
from keras.layers import Dense, TimeDistributed, Dropout, Flatten
from keras.layers.merge import add

# Stacked LSTM with residual connection in depth direction which means that we add residual connection in depth.
# Naturally LSTM has something like residual connections in time.
# We are going to demonstrate that the residual connection can allow to use much deeper stacked RNNs.
# Without residual connections, the depth should be limited to around 4 layers of depth at maximum. (checking)

# Define Vanilla LSTM with a single hidden layer of LSTM units.
def evaluate_Vanilla_lstm(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 1, 16
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
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

# define stacked LSTM
# Multiple hidden LSTM layers stacked on top of another
# LSTMs by default will produce a 2D output as an interpretation from the end of the sequence.
def stacked_lstm_model(trainX, trainy, testX, testy):

    verbose, epochs, batch_size = 0, 1, 16
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    input_stacked = Input(shape=(n_timesteps, n_features), name='stacked_lstm_input')
    lstm_out0=LSTM(9, activation='tanh',recurrent_dropout=0.2, dropout=0.2, return_sequences=True)(input_stacked)
    lstm_out1=LSTM(9, activation='tanh',recurrent_dropout=0.2, dropout=0.2, return_sequences=True)(lstm_out0)
    lstm_out2=LSTM(9, activation='tanh',recurrent_dropout=0.2, dropout=0.2, return_sequences=True)(lstm_out1)
    lstm_out3=LSTM(9, activation='tanh',recurrent_dropout=0.2, dropout=0.2, return_sequences=True)(lstm_out2)
    lstm_out4=LSTM(9, activation='tanh',recurrent_dropout=0.2, dropout=0.2, return_sequences=False)(lstm_out3)
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

def residual_lstm_layers(input, rnn_width, rnn_depth, rnn_dropout):

    x = input
    for i in range(rnn_depth):
        return_sequences = i < rnn_depth - 1
        # if the return_sequences is true, which means that this LSTM layer will output 3D instead of 2D(By default LSTM output 2D(the last time step of sequence)).
        # have the LSTM output a value for each time step in the input data.
        x_rnn = LSTM(rnn_width, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=return_sequences)(
            x)
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

def residual_lstm_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 25, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    input_res = Input(shape=(n_timesteps, n_features), name='residual_lstm_input')
    lstm_out = residual_lstm_layers(input_res, rnn_width=9, rnn_depth=8, rnn_dropout=0.2)
    Dense_out=Dense(100, activation='relu')(lstm_out)
    output=Dense(n_outputs,activation='softmax')(Dense_out)
    model = Model(inputs=input_res, outputs=output)
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # fit the network
    model.fit(x=trainX, y=trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    _, accuracy = model.evaluate(x=testX, y=testy, batch_size=batch_size, verbose=verbose)
    return accuracy


