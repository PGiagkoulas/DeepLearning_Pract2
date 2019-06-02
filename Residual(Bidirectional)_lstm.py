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
# Without residual connections, the depth should be limited to around 4 layers of depth at maximum.
def residual_lstm_layer(input, rnn_width, rnn_depth, rnn_dropout):
    # trainX.shape[1] = 128
    # trainX.shape[2] = 9
    # trainy.shape[1] = 6
    # Final shapes of time series dataset: (7352, 128, 9), (7352, 6), (2947, 128, 9), (2947, 6)
    # n_timesteps, n_features = trainX.shape[1], trainX.shape[2]
    # In Pandas code,
    # main_input = Input(shape=(None, n_length, n_features), dtype='float32', name='main_input')
    # n_length = 32
    # n_features= trainX.shape[2] = 9.
    # input_res = Input(shape=(None,n_timesteps,n_features),dtype='float32',name='residual_lstm_input')
    #auxiliary_input = Input(shape=(num_features,), name='aux_input')
    #input = Input(shape=(max_len,))
    x = input
    for i in range(rnn_depth):
        real_return_sequence = i < rnn_depth-1
        through_result=LSTM(units=rnn_width, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=real_return_sequence)(x)
        if real_return_sequence:
            # Intermediate layer should return sequences, input is also a sequence.
            if i > 0 or input.shape[-1]==rnn_width:
                x_out= add([x,through_result])
            else:
                # It is necessary for us to make sure that the input size and RNN output has to match, due to the sum operation.
                # If we try to define different LSTM layers, we have to perform the sum from layer 2 on.
                x_out= through_result
        else:
            # last layer just return the last element, rather than sequence.
            # Based on this, we just select only the last element of the previous output.
            def slice_for_last(x):
                return x[...,-1, :]
            x_out=([Lambda(slice_for_last)(x), through_result])
    return x_out


def residual_lstm_model(trainX, trainy, testX, testy):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    input_res = Input(shape=(None, n_timesteps, n_features), dtype='float32', name='residual_lstm_input')
    lstm_out= residual_lstm_layer(input_res, rnn_width=10, rnn_depth=8, rnn_dropout=0.2)
    dropout_out=Dropout(rate=0.5)(lstm_out)
    flatten_out=Flatten()(dropout_out)
    Dense_out=Dense(100, activation='relu')(flatten_out)
    output=Dense(n_outputs,activation='softmax')(Dense_out)
    model = Model(inputs=input_res, outputs=output)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the network
    model.fit(x=trainX, y=trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    _, accuracy = model.evaluate(x=testX, y=testy, batch_size=batch_size, verbose=1)
    return accuracy

def Bidirect_lstm_model(trainX, trainy, testX, testy, rnn_dropout):

    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    input_bidir = Input(shape=(None, n_timesteps, n_features), dtype='float32', name='bidirectional_lstm_input')
    x=Bidirectional(LSTM(units=100, return_sequence=True, recurrent_dropout=rnn_dropout))(input_bidir)
    x_out=TimeDistributed(Dense(n_outputs, activation="softmax"))(x) # softmax output layer
    model=Model(input_bidir,x_out)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the network
    model.fit(x=trainX, y=trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    _, accuracy = model.evaluate(x=testX, y=testy, batch_size=batch_size, verbose=1)
    return accuracy