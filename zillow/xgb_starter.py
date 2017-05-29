#! /usr/bin/python3

import pandas as pd
import numpy as np
import xgboost as xgb

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df_properties = pd.read_csv("properties_2016.csv")
df_train = pd.read_csv("train_2016.csv")

df_sample = pd.read_csv("sample_submission.csv")
df_train = df_properties.merge(df_train, how='inner', on='parcelid')


for c in df_train.dtypes[df_train.dtypes == object].index.values:
    df_train[c] = (df_train[c] == True)

df_train, df_valid = train_test_split(df_train, test_size=0.1, random_state=55)
x_train = df_train.ix[:,1:58]
y_train = df_train.ix[:,58:59]

x_valid = df_valid.ix[:,1:58]
y_valid = df_valid.ix[:,58:59]


print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

print('Training ...')

params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)


##############################################################
################### Testing    #############################
##############################################################


print('Building test set ...')

## merge with properties so that we get the logerror on test data
df_sample['parcelid'] = df_sample['ParcelId']
df_test = df_properties.merge(df_sample, on='parcelid', how='inner')
df_test.columns = df_test.columns.str.strip()

train_columns = list(df_train.columns)

train_columns.remove("logerror")
train_columns.remove("transactiondate")
train_columns.remove("parcelid")



## treat the "objects" the same way we treated the train data
x_test = df_test[train_columns]

for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

    
d_test = xgb.DMatrix(x_test)

print('Predicting on test ...')
p_test = clf.predict(d_test)


sub = pd.read_csv('sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

print('Writing csv ...')
sub.to_csv('xgb_starter.csv', index=False, float_format='%.4f')

