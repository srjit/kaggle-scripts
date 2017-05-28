#! /usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_input = pd.read_csv("properties_2016.csv")
df_properties = pd.read_csv("train_2016.csv")

df_sample = df_input.sample(frac=0.0002)

df_train = df_input.merge(df_properties, how='inner', on='parcelid')

# ## dropping all the Nans after the join
# without_na = df_train["logerror"].dropna()
# df_train = df_train[~df_train.index.isin(without_na.index)]


x_train = df_train.ix[:,1:58]
y_train = df_train.ix[:,58:59]


columns = list(x_train.columns)
variable = df_train.ix[:,1]

    
def replacenan(x):
    return 0 if np.isnan(x) else x
    

# x_train['airconditioningtypeid'] = x_train['airconditioningtypeid'].apply(lambda x: replacenan(x))

feature_indices = list(range(0,58))


def plot(feature_values, responses, column):
    plt.figure()
    plt.scatter(feature_values, responses)
    plt.xlabel(column)
    plt.ylabel('logerror')
    plt.show()

    
## let's test plotting one variable with response
def pass_values(i):
    x_train[columns[i]] = x_train[columns[i]].apply(lambda x: replacenan(x))
    feature_values = list(x_train.ix[:, i])
    responses = list(y_train.ix[:, 0])
    plot(feature_values, responses, columns[i])

    
map(pass_values, feature_indices)    
