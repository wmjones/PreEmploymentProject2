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
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

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

n_jobs = -1
geoScaleFactor = 10
n_clusters = 200

df_green = (df_green_raw.pipe(lambda x: x[x.Tip_amount >= 0][x.Total_amount >= 0])
            .pipe(lambda x: x[x.Payment_type != 2])
            .pipe(lambda x: x.rename(columns={'Trip_type ': 'Trip_type'}))
            .pipe(lambda x: x[-x.Trip_type.isna()])
            .pipe(lambda x: x.assign(Store_and_fwd_flag=x.Store_and_fwd_flag.eq('Y').mul(1))))  # need to check this

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
        cluster = MiniBatchKMeans(n_clusters=n_clusters, random_state=0).fit(X[cluster_features])
        return(cluster.labels_.reshape(-1, 1))

    def fit(self, *_):
        return self


cluster_features = ['Pickup_longitude_center', 'Pickup_latitude_center',
                    'Trip_dayofweek_x', 'Trip_dayofweek_y',
                    'Trip_timeofday_x', 'Trip_timeofday_y']
cluster_transformer = Pipeline(steps=[
    ('clustering', my_cluster_transformer()),
    ('onehot', OneHotEncoder(categories='auto'))
])

# maybe remove Extra, MTA_tax (needs to be changed to binary), improvement_surcharge, Tolls_amount
cat_features = ['Trip_dayofweek', 'Trip_day', 'VendorID', 'RateCodeID', 'Store_and_fwd_flag',
                'Payment_type', 'Trip_type']
# cat_features = ['Trip_dayofweek', 'Trip_day', 'VendorID', 'RateCodeID', 'Store_and_fwd_flag',
#                 'Payment_type', 'Extra', 'MTA_tax', 'improvement_surcharge',
#                 'Tolls_amount', 'Trip_type']
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(categories='auto'))
])

num_features = ['Pickup_longitude_center', 'Pickup_latitude_center',
                'Dropoff_longitude_center', 'Dropoff_latitude_center',
                'Passenger_count', 'Trip_distance']
# 'Total_amount']
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
    # ('cluster', cluster_transformer, cluster_features)
])

estimator = GradientBoostingRegressor(n_estimators=500, verbose=10, max_depth=15,
                                      max_features='sqrt', min_samples_split=1000,
                                      min_samples_leaf=50, learning_rate=0.1, random_state=0)
# 4.61 RMSE for Fare_amount with num
# 3.69 RMSE for Fare_amount with num, cat
# 3.81 RMSE for Fare_amount with num, cat, clust

# estimator = RandomForestRegressor(n_estimators=10, n_jobs=n_jobs, verbose=10, random_state=0)
# 1.98 RMSE for Fare_amount with num
# 1.74 RMSE for Fare_amount with num, cat
# 1.81 RMSE for Fare_amount with num, cat, clust

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    # ('TruncatedSVD', TruncatedSVD()),  # PCA()
    # ('PCA', PCA()),
    ('estimator', estimator)
])
# print(cross_val_score(df_train, y_train, n_jobs=n_jobs, cv=5, verbose=10))


# Hyperparameter Tuning
param_test = {'n_estimators': [10, 50, 100], 'max_features': [.3, .6, 1]}
gsearch = GridSearchCV(estimator=estimator, param_grid=param_test, n_jobs=n_jobs, cv=5, verbose=10)
gsearch.fit(df_train, y_train)
print("best_params: ", gsearch.best_params_)
print("best_score: ", gsearch.best_score_)


# Plotting
BB = (-74.3, -73.7, 40.5, 40.9)
nyc_map = plt.imread('https://aiblog.nl/download/nyc_-74.3_-73.7_40.5_40.9.png')


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


n_lon, n_lat = 200, 200
bins_lon = np.zeros(n_lon+1)
bins_lat = np.zeros(n_lat+1)
delta_lon = (BB[1]-BB[0]) / n_lon
delta_lat = (BB[3]-BB[2]) / n_lat
bin_width_miles = distance(BB[2], BB[1], BB[2], BB[0]) / n_lon
bin_height_miles = distance(BB[3], BB[0], BB[2], BB[0]) / n_lat
for i in range(n_lon+1):
    bins_lon[i] = BB[0] + i * delta_lon
for j in range(n_lat+1):
    bins_lat[j] = BB[2] + j * delta_lat

inds_pickup_lon = np.digitize(df_green.Pickup_longitude, bins_lon)
inds_pickup_lat = np.digitize(df_green.Pickup_latitude, bins_lat)
inds_dropoff_lon = np.digitize(df_green.Dropoff_longitude, bins_lon)
inds_dropoff_lat = np.digitize(df_green.Dropoff_latitude, bins_lat)

density_pickup, density_dropoff = np.zeros((n_lat, n_lon)), np.zeros((n_lat, n_lon))
dxdy = bin_width_miles * bin_height_miles
for i in range(n_lon):
    for j in range(n_lat):
        density_pickup[j, i] = np.sum((inds_pickup_lon == i+1) & (inds_pickup_lat == (n_lat-j))) / dxdy
        density_dropoff[j, i] = np.sum((inds_dropoff_lon == i+1) & (inds_dropoff_lat == (n_lat-j))) / dxdy

fig, axs = plt.subplots(2, 1, figsize=(18, 24))
axs[0].imshow(nyc_map, zorder=0, extent=BB)
im = axs[0].imshow(np.log1p(density_pickup), zorder=1, extent=BB, alpha=0.6, cmap='plasma')
axs[0].set_title('Pickup density [datapoints per sq mile]')
cbar = fig.colorbar(im, ax=axs[0])
cbar.set_label('log(1 + #datapoints per sq mile)', rotation=270)

axs[1].imshow(nyc_map, zorder=0, extent=BB)
im = axs[1].imshow(np.log1p(density_dropoff), zorder=1, extent=BB, alpha=0.6, cmap='plasma')
axs[1].set_title('Dropoff density [datapoints per sq mile]')
cbar = fig.colorbar(im, ax=axs[1])
cbar.set_label('log(1 + #datapoints per sq mile)', rotation=270)
plt.show()
