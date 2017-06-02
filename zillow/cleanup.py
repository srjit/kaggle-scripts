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

## only 1 value - i don't think this will be of any use
del df_train["assessmentyear"]

## only 2 values - probably not useful
del df_train["taxdelinquencyflag"]


## encoding propertyzoningdesc - there are around 2000 of these things
le = preprocessing.LabelEncoder()
encoder_model = le.fit(list(set(list(df_train["propertyzoningdesc"]))))
propertyzoningdesc_ = list(df_train["propertyzoningdesc"])
df_train["propertyzoningdesc"] = pd.Series(le.transform(propertyzoningdesc_))




train_data = pd.read_csv("train_data.csv")

