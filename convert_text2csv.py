import numpy as np
import pandas as pd
import sys
import os
# @Hazard 
# list
# list function (which takes any sequence as an argument and produces a new list with the same elements as the original sequence)
features= list()

# filepath



with open('./UCI_HAR_Dataset/features.txt') as f_feature: # after with we can not use f_feature any more.
# python split method: break a large string down into smaller chunks, or strings.
# if no separator is defined when you call upon the function. whitespace will be default.
    features=[ eve_line.split()[1] for eve_line in f_feature.readlines() ]
    print('The feature[1st:20th] we chosen looks like',features[0:20])
print('Number of Features: {}'.format(len(features)))
# tBodyAcc-mean()-X
# tBodyAcc-mean()-Y The indexs are [1] for them.
# tBodyAcc-mean()-Z

# Obtain the train data
# get the data from txt files to pandas dataframe

X_train = pd.read_csv('./UCI_HAR_Dataset/train/X_train.txt', header=None, delim_whitespace=True)
print('The shape of Pandas dataframe:', X_train.shape)
print('No of duplicates in X_train: {}'.format(sum(X_train.duplicated())))
# add subject column to the dataframe
X_train['subject'] = pd.read_csv('./UCI_HAR_Dataset/train/subject_train.txt', header=None, squeeze=True)
print('The shape of X_train[subject]',X_train['subject'].shape)

# get training label from y_train.txt
train_label=pd.read_csv('./UCI_HAR_Dataset/train/y_train.txt',names=['Activity'],squeeze=True)
train_maped_label=train_label.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',
                       4:'SITTING', 5:'STANDING',6:'LAYING'})

# put all columns in a single dataframe
X_train['Activity_num']=train_label
X_train['Activity_name']=train_maped_label
#print('The added dataframe looks like',X_train.sample)
print('The shape of final X_train Pandas dataframe:', X_train.shape)
print('*************************************************************************************************************')
# Do some work on test data
X_test = pd.read_csv('./UCI_HAR_Dataset/test/X_test.txt', header=None, delim_whitespace=True)
print('The shape of Pandas dataframe:', X_test.shape)

# add subject column to the dataframe
X_test['subject']=pd.read_csv('./UCI_HAR_Dataset/test/subject_test.txt', header=None, squeeze=True)

# get test label from y_test.txt
test_label=pd.read_csv('./UCI_HAR_Dataset/test/y_test.txt',names=['Activity'],squeeze=True)
test_maped_label=test_label.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',
                       4:'SITTING', 5:'STANDING',6:'LAYING'})
# put all columns in a single dataframe
X_test['Activity_num']=test_label
X_test['Activity_name']=test_maped_label
print('The added dataframe looks like',X_test.sample)
print('The shape of final X_test Pandas dataframe:', X_test.shape)

#for col in X_train.columns:
#    print(col)
# Check if there is duplicates
# .duplicated method return boolean Series denoting duplicate rows.
print('No of duplicates in X_train: {}'.format(sum(X_train.duplicated())))
print('No of duplicates in X_test : {}'.format(sum(X_test.duplicated())))

# Checking for NaN/null values
print('We have {} NaN/Null values in X_train'.format(X_train.isnull().values.sum()))
print('We have {} NaN/Null values in X_test'.format(X_test.isnull().values.sum()))


# Save data frame in a csv file

X_train.to_csv('./UCI_HAR_Dataset/csv_files/trainmixlabel.csv', index=False)
X_test.to_csv('./UCI_HAR_Dataset/csv_files/testmixedlabel.csv', index=False)