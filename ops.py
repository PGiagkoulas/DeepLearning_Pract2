import numpy as np
import pandas as pd 
from datetime import datetime
import time
import os.path

SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z"
]
DATADIR='UCI_HAR_Dataset'
datadir = "~/Data/"
subdir="UCI_HAR_Dataset/"

UCIdir= os.path.join(os.path.expanduser(datadir),subdir)

print("The whole dir for dataset is", UCIdir)

TRAIN_Datapath = os.path.join(UCIdir,"train/")
print("The train dir for dataset is", TRAIN_Datapath)
TRAIN_Datapath_files = [ TRAIN_Datapath+ 'Inertial Signals/'+ _ + '_train.txt' for _ in SIGNALS ]
print('\n')
print("The paths list for multi-files is",TRAIN_Datapath_files)
TEST_Datapath = os.path.join(UCIdir,"test/")
print("The test dir for dataset is", TEST_Datapath)
TEST_Datapath_files = [ TEST_Datapath+ 'Inertial Signals/'+ _ + '_test.txt' for _ in SIGNALS ]
# Train and test label path
TRAIN_label=TRAIN_Datapath+'y_train.txt'
TEST_label=TEST_Datapath+'y_test.txt'


def gnrat_csv(filename):
    return pd.read_csv(filename,delim_whitespace=True,header=None)

# load training signal
'''
def load_X_signal(UCIdir,subset="/"):
    signals_data=[]
    for signal in SIGNALS:
        filename = UCIdir+subset+"Inertial Signals/"+signal+{subset}.txt
        filename=
        filename=f'/{subset}/Inertial Signals/{signal}_{subset}.txt'
        signals_data.append(gnrat_csv(filename).as_matrix())
    return np.transpose(signals_data,(1,2,0))
'''

'''
def load_X_signal(subset):
    signals_data=[]
    for signal in SIGNALS:
        filename = f'UCI_HAR_Dataset/{subset}/Inertial Signals/{signal}_{subset}.txt'
        #filename=f'UCI_HAR_Dataset/{subset}/Inertial Signals/{signal}_{subset}.txt'
        signals_data.append(gnrat_csv(filename).as_matrix())
    return np.transpose(signals_data,(1,2,0))
'''




# load training and test signal
def load_X_signal(filespaths):
    signals_data=[]
    for filepath in filespaths:
        signals_data.append(gnrat_csv(filepath).as_matrix())
    return np.transpose(signals_data,(1,2,0))

# load training and test label
def load_y_label(filepath):
    label= gnrat_csv(filepath)[0]
    return pd.get_dummies(label).as_matrix()

def load_data():
    # Returns :  X_train, X_test, y_train, y_test

    X_train, X_test = load_X_signal(TRAIN_Datapath_files), load_X_signal(TEST_Datapath_files)
    y_train,y_test=load_y_label(TRAIN_label),load_y_label(TEST_label)

    return X_train, X_test, y_train, y_test
