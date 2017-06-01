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

for c in df_train.dtypes[df_train.dtypes == object].index.values:
    df_train[c] = (df_train[c] == True)

df_train = df_train[features]    

correlations = df_train.corr()

corr_file = "variable_correlations.csv"
correlations.to_csv(corr_file, sep=',')



