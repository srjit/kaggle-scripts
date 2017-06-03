#! /usr/bin/python3
import pandas as pd
import numpy as np
import xgboost as xgb

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn import preprocessing



df_properties = pd.read_csv("properties_2016.csv")
df_train = pd.read_csv("train_2016.csv")

df_train = df_properties.merge(df_train, how='inner', on='parcelid')


########## Create the month field  ##########
df_train["transactiondate"] = pd.to_datetime(df_train["transactiondate"])
df_train["transactionmonth"] = df_train["transactiondate"].apply(lambda x : int(str(x)[5:7]))


cols = list(df_train.columns)
features = cols[1:58] + cols[60:61] + cols[58:59]

## write the joined data to a file
## df_train.to_csv('train_data.csv', index=False)

## Building class type id
## building_class_type_ids =  set(list(df_train["buildingclasstypeid"]))
## fixing the nan issue adding a label 0 to those buildings with no values


def check4(x):
    if(x != 4):
        return 0
    return x

df_train["buildingclasstypeid"] = df_train["buildingclasstypeid"].apply(lambda x : check4(x))

## drop "decktypeid"
del df_train["decktypeid"]
del df_train["poolcnt"]
del df_train["pooltypeid10"]
del df_train["pooltypeid2"]
del df_train["pooltypeid7"]
del df_train["storytypeid"]
del df_train["transactiondate"]

## only one unique value
del df_train["hashottuborspa"]
del df_train["propertycountylandusecode"]
del df_train["fireplaceflag"]


## only 1 value - i don't think this will be of any use
del df_train["assessmentyear"]

## only 2 values - probably not useful
del df_train["taxdelinquencyflag"]


## encoding propertyzoningdesc - there are around 2000 of these things
le = preprocessing.LabelEncoder()

## check for missing values - Nans have formed a level - 0 while it was encoded
prop_zon_uniq = set(list(df_train["propertyzoningdesc"]))
encoder_model = le.fit(list(prop_zon_uniq))
propertyzoningdesc_ = list(df_train["propertyzoningdesc"])
df_train["propertyzoningdesc"] = pd.Series(le.transform(propertyzoningdesc_))

## what is the max count of df_train["propertyzoningdesc"] - LAR1 : Lets use it as missing value
## df_train[['propertyzoningdesc']].groupby(['propertyzoningdesc']).agg(['count'])



del df_train["parcelid"]

features = list(df_train.columns)
features.remove("logerror")
features.append("logerror")

df_input = df_train[features]
df_train, df_valid = train_test_split(df_input, test_size=0.1, random_state=55)
variables = features[0:47]
response = features[47]


x_train = df_train[variables]
y_train = df_train[response]

x_valid = df_valid[variables]
y_valid = df_valid[response]


print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)


print('Training ...')

params = {}
params['eta'] = 0.01
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)




##############################################################
################### Testing    #############################
##############################################################

df_sample = pd.read_csv("sample_submission.csv")

print('Building test set ...')

## merge with properties so that we get the logerror on test data
df_sample['parcelid'] = df_sample['ParcelId']
df_test = df_properties.merge(df_sample, on='parcelid', how='inner')
df_test.columns = df_test.columns.str.strip()


## treat the "objects" the same way we treated the train data

def check4(x):
    if(x != 4):
        return 0
    return x

df_test["buildingclasstypeid"] = df_test["buildingclasstypeid"].apply(lambda x : check4(x))


## replacing values which wasnt there in the train set with nan - we can make some changes later
propertyzoningdesc_test = list(df_test["propertyzoningdesc"])
propertyzoningdesc_test_update = list(map(lambda x: x if x in prop_zon_uniq else np.nan, propertyzoningdesc_test))

df_test["propertyzoningdesc"] = pd.Series(le.transform(propertyzoningdesc_test_update))

## october predictions
df_test["transactionmonth"] = 10
x_test = df_test[variables]
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
sub.to_csv('xgb_v3.csv', index=False, float_format='%.4f')
