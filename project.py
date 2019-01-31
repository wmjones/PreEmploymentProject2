import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin

from sklearn.cluster import MiniBatchKMeans
# from sklearn.decomposition import dict_learning_online
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# df_fhv = pd.read_csv("./fhv_tripdata_2015-09.csv")
df_green_raw = pd.read_csv("./green_tripdata_2015-09.csv")
# df_green = df_green[:10000]
# df_yellow = pd.read_csv("./yellow_tripdata_2015-09.csv")

# think about how to get taxi zones to take into account which ones are near each other
# store_and_fwd means time info is off
# tip amount not included for cash so need to take that into account (turn it into missing if cash?)
# https://medium.com/analytics-vidhya/machine-learning-to-predict-taxi-fare-part-one-exploratory-analysis-6b7e6b1fbc78
# https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration
# https://code.google.com/archive/p/s2-geometry-library/
# do a hash for the taxi zones
# use clustering to handle lat long (maybe use time of day as well)
# Perform some Hierarchical Clustering instead of KMeans because of KMeans workes along the maximizing variance if the feature space is linear in nature but if it is non-linear, then Hierarchical Clusterings like PAM, CLARA, and DBSCAN are best to use

# Data Processing:
# store_and_fwd means time info is off
# tip amount not included for cash so need to take that into account (turn it into missing if cash?)
# think about how to get taxi zones to take into account which ones are near each other
# payment_type = 5 is unknown
# create a base fare (either take off tip for non cash or write formula to calculate total_amount)

# Feature Engineering:
# hierarchical clustering spaio-temporal
# hierarchical clustering spaio
# time into bins

# figure out how to make start of week end of week close / start of day end of day

# Plan:
# hierarchical clustering spaio-temporal -> hash/embedding layer -> XGBoost/Random Forest
# hierarchical clustering spaio-temporal -> OHE -> PCA -> NN
# drop lat/long just use zones -> hash/embedding layer -> XGBoost/Random Forest
# drop lat/long just use zones -> OHE -> PCA -> NN

# Center of NYC is 40.7831° N, 73.9712° W

n_jobs = 4
geoScaleFactor = 100
n_clusters = 100

df_green = (df_green_raw.pipe(lambda x: x[x.Tip_amount >= 0][x.Total_amount >= 0])
            .pipe(lambda x: x[x.Payment_type != 2])
            .pipe(lambda x: x.rename(columns={'Trip_type ': 'Trip_type'}))
            .pipe(lambda x: x[x.Trip_type.isna() == False])
            .pipe(lambda x: x.assign(Store_and_fwd_flag=x.Store_and_fwd_flag.eq('Y').mul(1))))

df_green[['Pickup_longitude_center', 'Dropoff_longitude_center']] = (
    df_green[['Pickup_longitude', 'Dropoff_longitude']] + 73.9712)*geoScaleFactor
df_green[['Pickup_latitude_center', 'Dropoff_latitude_center']] = (
    df_green[['Pickup_latitude', 'Dropoff_latitude']] - 40.7831)*geoScaleFactor

df_green[['Lpep_dropoff_datetime', 'lpep_pickup_datetime']] = df_green[['Lpep_dropoff_datetime', 'lpep_pickup_datetime']].apply(
    lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
df_green['Trip_dayofweek'] = df_green['lpep_pickup_datetime'].dt.dayofweek
df_green['Trip_total_time'] = (df_green['Lpep_dropoff_datetime'] - df_green['lpep_pickup_datetime']).astype('timedelta64[s]')
timeFromStart = df_green['lpep_pickup_datetime'] - pd.to_datetime('2015-09-01')
timeFromStart_components = timeFromStart.dt.components
df_green['Trip_day'] = timeFromStart.dt.days
df_green['Trip_timeofday'] = timeFromStart_components.hours*60 + \
                             timeFromStart_components.minutes + \
                             timeFromStart_components.seconds/60


def to_circle(k, n):
    return(math.cos(k*2*math.pi/n), math.sin(k*2*math.pi/n))


df_green['Trip_timeofday_x'], df_green['Trip_timeofday_y'] = zip(
    *df_green['Trip_timeofday'].apply(lambda x: to_circle(x, 24)))

df_green['Trip_dayofweek_x'], df_green['Trip_dayofweek_y'] = zip(
    *df_green['Trip_dayofweek'].apply(lambda x: to_circle(x, 7)))

df_green_train = df_green[df_green.Trip_day < 27]
df_green_test = df_green[df_green.Trip_day >= 27]

train_labels = ['Pickup_longitude_center', 'Dropoff_longitude_center',
                'Pickup_latitude_center', 'Dropoff_latitude_center',
                'Trip_dayofweek', 'Trip_total_time', 'Trip_day', 'Trip_timeofday',
                'VendorID', 'Store_and_fwd_flag', 'RateCodeID', 'Passenger_count',
                'Trip_distance', 'Trip_timeofday_x', 'Trip_timeofday_y',
                'Trip_dayofweek_x', 'Trip_dayofweek_y', 'Payment_type', 'Trip_type', 'Extra',
                'MTA_tax']
df_train = df_green_train[train_labels]
df_test = df_green_test[train_labels]
y_train = df_green_train[['Fare_amount']].values.ravel()
y_test = df_green_test[['Fare_amount']].values.ravel()


class my_cluster_transformer(TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, *_):
        cluster = MiniBatchKMeans(n_clusters=n_clusters).fit(X[cluster_features])
        print(cluster.labels_)
        return(cluster.labels_)

    def fit(self, *_):
        return self

# def my_cluster_transformer(x):
#     cluster = KMeans(n_clusters=n_clusters, n_jobs=n_jobs).fit(x)
#     return(cluster.labels_)


cluster_features = ['Pickup_longitude_center', 'Pickup_latitude_center',
                    'Trip_dayofweek_x', 'Trip_dayofweek_y',
                    'Trip_timeofday_x', 'Trip_timeofday_y']
cluster_transformer = Pipeline(steps=[
    ('clustering', my_cluster_transformer()),
    ('onehot', OneHotEncoder(categories='auto'))
])

# cluster = KMeans(n_clusters=n_clusters, n_jobs=n_jobs).fit(
#     df_train[['Pickup_longitude_center', 'Pickup_latitude_center',
#               'Trip_dayofweek_x', 'Trip_dayofweek_y',
#               'Trip_timeofday_x', 'Trip_timeofday_y']])
# df_train['cluster_labels'] = cluster.labels_


# maybe remove Extra, MTA_tax (needs to be changed to binary), improvement_surcharge, Tolls_amount
cat_features = ['Trip_dayofweek', 'Trip_day', 'VendorID', 'RateCodeID', 'Store_and_fwd_flag',
                'Payment_type', 'Trip_type']
# cat_features = ['Trip_dayofweek', 'Trip_day', 'VendorID', 'RateCodeID', 'Store_and_fwd_flag',
#                 'Payment_type', 'Extra', 'MTA_tax', 'improvement_surcharge',
#                 'Tolls_amount', 'Trip_type']
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(categories='auto'))
])

# cluster_features = ['cluster_labels']
# cluster_transformer = Pipeline(steps=[
#     # ('dictionary', FeatureHasher(n_features=100, input_type='string'))
#     ('onehot', OneHotEncoder(categories='auto'))
# ])

num_features = ['Pickup_longitude_center', 'Pickup_latitude_center',
                'Dropoff_longitude_center', 'Dropoff_latitude_center',
                'Passenger_count', 'Trip_distance']
# 'Total_amount']
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features),
    ('cluster', cluster_transformer, cluster_features)
])

estimator = GradientBoostingRegressor(n_estimators=100, verbose=10)
# 4.61 RMSE for Fare_amount with only num
# 4.26 RMSE for Fare_amount with num, cat

# estimator = RandomForestRegressor(n_estimators=70, n_jobs=n_jobs, verbose=10, random_state=0)
# 1.98 RMSE for Fare_amount with only num


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    # ('pca', TruncatedSVD()),  # PCA()
    ('estimator', estimator)
])
pipeline.fit(df_train, y_train)
prediction = pipeline.predict(df_train)
print(math.sqrt(mean_squared_error(y_train, prediction)))

# Hyperparameter Tuning
# param_test1 = {'n_estimators': [10, 50, 100]}
# gsearch1 = GridSearchCV(
#     estimator=estimator,
#     param_grid=param_test1, n_jobs=n_jobs, cv=5)
# gsearch1.fit(df_train, y_train)
# print(gsearch1.results_)
