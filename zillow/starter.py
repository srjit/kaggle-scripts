#! /usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_input = pd.read_csv("properties_2016.csv")
df_parcels = pd.read_csv("train_2016.csv")

df_sample = df_input.sample(frac=0.0002)

df_input = pd.merge(df_sample, df_parcels, how='left', on=['parcelid'])
fields = list(df_input.columns)

feature_indices = list(range(1,58,1))
response_variable =  fields[58]

## let's test plotting one variable with response
feature_values = df_input.ix[:, 1].tolist()
responses = df_input.ix[:, 58].tolist()


# x = list(range(1,11))
# y = [a*i + b for i in x]

# plt.figure()
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('Ax + B')

# plt.show()