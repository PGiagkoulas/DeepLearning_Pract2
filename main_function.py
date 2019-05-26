import tensorflow as tf
import numpy as np
from ops import *

# set up hyper-parameters
import argparse

"""parsing and configuration"""''
'''Use parser arguments to set up parameter accordingly'''
def parse_args():

    desc="TensorFlow implementation of LSTM_RNN on Human Activity Recognition! author: Hazard Wen"
    parser=argparse.ArgumentParser(description=desc)

    # all the outside arguments can be passed successfully.
    # add dropout or no Dropout, probability to keep units
    parser.add_argument('--dropout_keep_prob', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=0.8)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.1)
    parser.add_argument('--max_epochs', type=int, help='Number of epochs to run.', default=100)
    parser.add_argument('--batch_size', type=int,
                        help='Number of samples to process in a batch.', default=64)
    parser.add_argument('--activation_function', type=str, choices=['elu', 'relu', 'relu6', 'leaky_relu', 'sigmoid','tanh'],
                        help='which activation function we are going to use', default='None')
    parse.add_argument('--lamda_loss_amount',type=float,help='l2 loss which is used to prevent this overkill neural network to overfit the data',default=0.0015)

    # This 300 seems same with max_epoch
    parse.add_argument('--training_iters',type=int,help='The number of multiple 300',default=7352*300)
    parse.add_argument('--display_iter',type=int,help='The number of displaying test set accuracy',default=30000)
    #******************************************************************************************************************#
    # add arguments for different optimizer
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='None')

    return check_args(parser.parse_args())

def check_args(args):

    return args


# Load the training data and label
def main():

    #print('The version of tensorflow:', tf.__version__)
    #print('The version of keras:', keras.__version__)

    # Loading the train and test data
    X_train, X_test, Y_train, Y_test = load_data()
    list_train_test=[]
    list_train_test.append(X_train) #(7352, 128, 9)
    list_train_test.append(X_test)
    list_train_test.append(Y_train)
    list_train_test.append(Y_test)
    for _ in list_train_test:
        print('The shape is:', _.shape)

    n_steps=len(X_train[0])     # 128 timesteps per series
    n_input=len(X_train[0][0])  # 9 input parameters per timestep
    n_classes=6

     # Graph input/output
    x= tf.placeholder(tf.float32, [None, n_steps, n_input])
    y_true= tf.placeholder(tf.float32, [None, n_classes])


if __name__ == '__main__':

    main()