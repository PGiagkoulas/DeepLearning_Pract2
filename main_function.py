import tensorflow as tf
import numpy as np
from ops import *

# Load the training data and label
def main():

    #print('The version of tensorflow:', tf.__version__)
    #print('The version of keras:', keras.__version__)

    # Loading the train and test data
    X_train, X_test, Y_train, Y_test = load_data()
    list_train_test=[]
    list_train_test.append(X_train)
    list_train_test.append(X_test)
    list_train_test.append(Y_train)
    list_train_test.append(Y_test)
    for _ in list_train_test:
        print('The shape is:', _.shape)
if __name__ == '__main__':

    main()