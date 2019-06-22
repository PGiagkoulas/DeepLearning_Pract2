from keras.models import load_model
from keras.models import Model
from all_utils import load_dataset


# fine-tuning FC layers of ConvLSTM
def fine_tune_convlstm():
    # load data
    trainX, trainy, testX, testy, aux_trainX, aux_trainy, aux_testX, aux_testy = load_dataset()
    ## datastuff
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # reshape into subsequences (samples, time steps, rows, cols, channels)
    n_steps, n_length = 4, 32
    trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
    # load model
    model = load_model('convlstm.h5')
    print(model.summary())
    _, _, _, saved_accuracy, _ = model.evaluate(x=[testX, aux_testX], y=[testy, aux_testy], batch_size=128, verbose=1)
    # freeze layers
    model.get_layer('convlstm_0').trainable = False
    model.get_layer('dense_0').trainable = False
    model.get_layer('aux_output').trainable = False
    # recompile for freeze to take effect
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # re-train
    model.fit(x=[trainX, aux_trainX], y=[trainy, aux_trainy], epochs=30, batch_size=128,
              verbose=1, validation_data=([testX, aux_testX], [testy, aux_testy]))
    # evaluate re-trained model
    _, _, _, tuned_accuracy, _ = model.evaluate(x=[testX, aux_testX], y=[testy, aux_testy],
                                                          batch_size=128, verbose=1)
    print(">> Saved model accuracy is: {0}".format(saved_accuracy))
    print(">> Fine-tuned model accuracy is: {0}".format(tuned_accuracy))
    model.save("convlstm_final.h5")
    print("Saved fine-tuned model to disk")

# fine-tuning FC layers of ConvLSTM
def fine_tune_cnnlstm():
    # load data
    trainX, trainy, testX, testy, aux_trainX, aux_trainy, aux_testX, aux_testy = load_dataset()
    ## datastuff
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # reshape into subsequences (samples, time steps, rows, cols, channels)
    n_steps, n_length = 4, 32
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
    # load model
    model = load_model('cnnlstm.h5')
    print(model.summary())
    _, _, _, saved_accuracy, _ = model.evaluate(x=[testX, aux_testX], y=[testy, aux_testy], batch_size=128, verbose=1)
    # freeze layers
    model.get_layer('tdconv1d_0').trainable = False
    model.get_layer('tdconv1d_1').trainable = False
    model.get_layer('lstm_0').trainable = False
    model.get_layer('dense_0').trainable = False
    model.get_layer('aux_output').trainable = False
    # recompile for freeze to take effect
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # re-train
    model.fit(x=[trainX, aux_trainX], y=[trainy, aux_trainy], epochs=25, batch_size=128,
              verbose=1, validation_data=([testX, aux_testX], [testy, aux_testy]))
    # evaluate re-trained model
    _, _, _, tuned_accuracy, _ = model.evaluate(x=[testX, aux_testX], y=[testy, aux_testy],
                                                          batch_size=128, verbose=1)
    print(">> Saved model accuracy is: {0}".format(saved_accuracy))
    print(">> Fine-tuned model accuracy is: {0}".format(tuned_accuracy))
    model.save("cnnlstm_final.h5")
    print("Saved fine-tuned model to disk")

if __name__== "__main__":
    fine_tune_cnnlstm()