#! /usr/bin/python3
import pandas as pd
import numpy as np
import xgboost as xgb

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from sklearn import preprocessing

df_properties = pd.read_csv("properties_2016.csv")
df_train = pd.read_csv("train_2016.csv")

df_sample = pd.read_csv("sample_submission.csv")
df_train = df_properties.merge(df_train, how='inner', on='parcelid')


########## Create the month field  ##########
df_train["transactiondate"] = pd.to_datetime(df_train["transactiondate"])
df_train["transactionmonth"] = df_train["transactiondate"].apply(lambda x : int(str(x)[5:7]))

# df_train["transactionmonth"] = df_train["transactiondate"].apply(lambda x: x.strftime('%B'))
# labelling with scikit labelling
# le = preprocessing.LabelEncoder()
# encoder_model = le.fit(list(set(list(df_train["transactionmonth"]))))
# df_train["month"] = df_train["transactionmonth"].apply(lambda x : encoder_model.transform([x])[0])


cols = list(df_train.columns)
features = cols[1:58] + cols[60:61] + cols[58:59]

for c in df_train.dtypes[df_train.dtypes == object].index.values:
    df_train[c] = (df_train[c] == True)

df_train = df_train[features]    

df_train, df_valid = train_test_split(df_train, test_size=0.1, random_state=55)
x_train = df_train.ix[:,0:58]
y_train = df_train.ix[:,58:59]

x_valid = df_valid.ix[:,0:58]
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


train_columns = list(df_train.columns)
train_columns.remove("transactionmonth")
train_columns.remove("logerror")

## treat the "objects" the same way we treated the train data
x_test = df_test[train_columns]


for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

## lets find predictions for the month of october    
x_test["transactionmonth"] = 10    
d_test_oct = xgb.DMatrix(x_test)
print('Predicting on test ...')
p_test_oct = clf.predict(d_test_oct)


## predictions for the month of november
x_test["transactionmonth"] = 11
d_test_nov = xgb.DMatrix(x_test)
print('Predicting on test ...')
p_test_nov = clf.predict(d_test_nov)


## predictions for the month of december
x_test["transactionmonth"] = 12
d_test_dec = xgb.DMatrix(x_test)
print('Predicting on test ...')
p_test_dec = clf.predict(d_test_dec)

sub = pd.read_csv('sample_submission.csv')

sub["201610"] = p_test_oct
sub["201611"] = p_test_nov
sub["201612"] = p_test_dec
sub["201710"] = p_test_oct
sub["201711"] = p_test_nov
sub["201712"] = p_test_dec
# for c in sub.columns[sub.columns != 'ParcelId']:
#     sub[c] = p_test

print('Writing csv ...')
sub.to_csv('xgb_v2.csv', index=False, float_format='%.4f')

