from keras.layers import LSTM
from keras.layers import Lambda
from keras.layers.merge import add
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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
        outfile.write(str(aux_loss))
        outfile.write(str(aux_accuracy) + ',')
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