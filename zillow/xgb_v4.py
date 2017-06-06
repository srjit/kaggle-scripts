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


del df_train["decktypeid"]
del df_train["poolcnt"]
del df_train["pooltypeid10"]
del df_train["pooltypeid2"]
del df_train["pooltypeid7"]
del df_train["storytypeid"]
del df_train["transactiondate"]
del df_train["hashottuborspa"]
del df_train["propertycountylandusecode"]
del df_train["fireplaceflag"]


## only 1 value - i don't think this will be of any use
del df_train["assessmentyear"]

## only 2 values - probably not useful
del df_train["taxdelinquencyflag"]
del df_train["parcelid"]


transmonth_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',7 :'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
df_train['transactionmonth'] = df_train['transactionmonth'].apply(lambda x: transmonth_dict.get(float(x)))
transactionmonth_encoded = pd.get_dummies(df_train['transactionmonth'])



actype = {1.0: "Central_AC", 3.0 : "EvaporativeCooler", 4.0 : "GeoThermal", 5.0:"None_AC", 9.0:"Refrigeration", 11.0:"WallUnit", 13.0: "Yes_AC"}
df_train['airconditioningtypeid'] = df_train["airconditioningtypeid"].apply(lambda x: actype.get(float(x),"ac_unknown"))
airconditioningtypeid_encoded = pd.get_dummies(df_train['airconditioningtypeid'])


archtype = {1:'A-Frame', 2:'Bungalow', 3:'Cape Cod', 4:'Cottage', 5:'Colonial', 6:'Custom', 7:'Contemporary', 8:'Conventional', 9:'Dome', 10:'French Provincial', 11:'Georgian', 12:'High Rise', 13:'Historical', 14:'Log Cabin/Rustic', 15:'Mediterranean', 16:'Modern', 17:'Mansion', 18:'English', 19:'Other', 20:'Prefab', 21:'Ranch/Rambler', 22:'Raised Ranch', 23:'Spanish', 24:'Traditional', 25:'Tudor', 26:'Unfinished/Under Construction', 27:'Victorian'}
df_train['architecturalstyletypeid'] = df_train["architecturalstyletypeid"].apply(lambda x: archtype.get(float(x),"arch_unknown"))
archtype_encoded = pd.get_dummies(df_train['architecturalstyletypeid'])


buildingclasstype = {1: "FPSS", 2: "FPRCC", 3: "NC", 4:"WOOD", 5:"OTHER"}
df_train["buildingclasstypeid"] = df_train["buildingclasstypeid"].apply(lambda x : buildingclasstype.get(float(x), "buildingclass_unknown"))
buildingclass_encoded = pd.get_dummies(df_train['buildingclasstypeid'])


heatingtypd = {1:'Baseboard', 2:'Central_Heat', 3:'Coal', 4:'Convection', 5:'Electric', 6:'Forced air', 7:'Floor/Wall', 8:'Gas', 9:'Geo Thermal', 10:'Gravity', 11:'Heat Pump', 12:'Hot Water', 13:'None', 14:'Other', 15:'Oil', 16:'Partial', 17:'Propane', 18:'Radiant', 19:'Steam', 20:'Solar', 21:'Space/Suspended', 22:'Vent', 23:'Wood Burning', 24:'Yes_Heat', 25:'Zone'}
df_train["heatingorsystemtypeid"] = df_train["heatingorsystemtypeid"].apply(lambda x: heatingtypd.get(float(x), "heating_unknown"))
heatingtype_encoded = pd.get_dummies(df_train["heatingorsystemtypeid"])


propertytypeid = {31:'Commercial/Office/Residential Mixed Used', 46:'Multi-Story Store', 47:'Store/Office (Mixed Use)', 246:'Duplex (2 Units, Any Combination)', 247:'Triplex (3 Units, Any Combination)', 248:'Quadruplex (4 Units, Any Combination)', 260:'Residential General', 261:'Single Family Residential', 262:'Rural Residence', 263:'Mobile Home', 264:'Townhouse', 265:'Cluster Home', 266:'Condominium', 267:'Cooperative', 268:'Row House', 269:'Planned Unit Development', 270:'Residential Common Area', 271:'Timeshare', 273:'Bungalow', 274:'Zero Lot Line', 275:'Manufactured, Modular, Prefabricated Homes', 276:'Patio Home', 279:'Inferred Single Family Residential', 290:'Vacant Land - General', 291:'Residential Vacant Land'}
df_train["propertylandusetypeid"] = df_train["propertylandusetypeid"].apply(lambda x: propertytypeid.get(float(x), "property_unknown"))
propertyid_encoded = pd.get_dummies(df_train["propertylandusetypeid"])

df_train = pd.concat([df_train, transactionmonth_encoded, airconditioningtypeid_encoded, archtype_encoded, buildingclass_encoded, heatingtype_encoded, propertyid_encoded], axis=1)

del df_train["airconditioningtypeid"]
del df_train["architecturalstyletypeid"]
del df_train["buildingclasstypeid"]
del df_train["heatingorsystemtypeid"]
del df_train["propertylandusetypeid"]
del df_train["transactionmonth"]
del df_train["propertyzoningdesc"]


## we can check this later - ToDo
## df_train['bedroomcnt']=df_train['bedroomcnt'].fillna(np.nan)
features = list(df_train.columns)
features.remove("logerror")
features.append("logerror")

df_train = df_train[features]


## intermediate write
df_train.to_csv("train_data_v4.csv", sep=',')


# df_train['basementsqft']=df_train['basementsqft'].fillna(450)
# basement_bins = range(100, 1600, 50)
# out = pd.cut(df_train.basementsqft, bins = basement_bins)


variables = features[0:96]
response = features[96]

df_input, df_valid = train_test_split(df_train, test_size=0.1, random_state=55)

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
clf = xgb.train(params, d_train, 15000, watchlist, early_stopping_rounds=100, verbose_eval=10)





##############################################################
################### Testing    #############################
##############################################################

df_sample = pd.read_csv("sample_submission.csv")

print('Building test set ...')

## merge with properties so that we get the logerror on test data
df_sample['parcelid'] = df_sample['ParcelId']
df_test = df_properties.merge(df_sample, on='parcelid', how='inner')
df_test.columns = df_test.columns.str.strip()


df_test['airconditioningtypeid'] = df_test["airconditioningtypeid"].apply(lambda x: actype.get(float(x),"ac_unknown"))
airconditioningtypeid_encoded_test = pd.get_dummies(df_test['airconditioningtypeid'])

df_test['architecturalstyletypeid'] = df_test["architecturalstyletypeid"].apply(lambda x: archtype.get(float(x),"arch_unknown"))
archtype_encoded_test = pd.get_dummies(df_test['architecturalstyletypeid'])

df_test["buildingclasstypeid"] = df_test["buildingclasstypeid"].apply(lambda x : buildingclasstype.get(float(x), "buildingclass_unknown"))
buildingclass_encoded_test = pd.get_dummies(df_test['buildingclasstypeid'])

df_test["heatingorsystemtypeid"] = df_test["heatingorsystemtypeid"].apply(lambda x: heatingtypd.get(float(x), "heating_unknown"))
heatingtype_encoded_test = pd.get_dummies(df_test["heatingorsystemtypeid"])

df_test["propertylandusetypeid"] = df_test["propertylandusetypeid"].apply(lambda x: propertytypeid.get(float(x), "property_unknown"))
propertyid_encoded_test = pd.get_dummies(df_test["propertylandusetypeid"])


df_test = pd.concat([df_test, airconditioningtypeid_encoded_test, archtype_encoded_test, buildingclass_encoded_test, heatingtype_encoded_test, propertyid_encoded_test], axis=1)

del df_test["airconditioningtypeid"]
del df_test["architecturalstyletypeid"]
del df_test["buildingclasstypeid"]
del df_test["heatingorsystemtypeid"]
del df_test["propertylandusetypeid"]



del df_test["propertyzoningdesc"]


df_test["Jan"] = 0
df_test["Feb"] = 0
df_test["Mar"] = 0
df_test["Apr"] = 0
df_test["May"] = 0
df_test["Jun"] = 0
df_test["Jul"] = 0
df_test["Aug"] = 0
df_test["Sep"] = 0
df_test["Oct"] = 1
df_test["Nov"] = 0
df_test["Dec"] = 0

## october predictions
x_test = df_test[variables]
d_test_oct = xgb.DMatrix(x_test)
print('Predicting on test for october ...')
p_test_oct = clf.predict(d_test_oct)


## predictions for the month of november
x_test["Oct"] = 0
x_test["Nov"] = 1
d_test_nov = xgb.DMatrix(x_test)
print('Predicting on test november ...')
p_test_nov = clf.predict(d_test_nov)


## predictions for the month of december
x_test["Nov"] = 0
x_test["Dec"] = 1
d_test_dec = xgb.DMatrix(x_test)
print('Predicting on test decemeber ...')
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
sub.to_csv('xgb_v5.csv', index=False, float_format='%.4f')







