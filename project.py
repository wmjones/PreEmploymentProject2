import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin

from sklearn.feature_selection import SelectFromModel

from sklearn.cluster import MiniBatchKMeans

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import scipy.stats as stats
from statsmodels.formula.api import ols

df_green_raw = pd.read_csv("./green_tripdata_2015-09.csv")
df_green = (df_green_raw.pipe(lambda x: x[x.Tip_amount >= 0][x.Total_amount >= 0])  # drop negative tip and negative total_amounts
            .pipe(lambda x: x[x.Payment_type != 2])  # drop since cash transactions have no payment info
            .pipe(lambda x: x.rename(columns={'Trip_type ': 'Trip_type'}))
            .pipe(lambda x: x[-x.Trip_type.isna()])
            .pipe(lambda x: x.assign(Store_and_fwd_flag=x.Store_and_fwd_flag.eq('Y').mul(1))))  # changes Y to 1 N to 0

geoScaleFactor = 10
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
df_green['Trip_tippercent'] = df_green.Tip_amount / df_green.Fare_amount
df_green['Trip_tippercent'] = df_green['Trip_tippercent'].fillna(0)
df_green['Trip_tippercent'] = df_green['Trip_tippercent'].clip(upper=100)
df_green[['Passenger_count']] = df_green[['Passenger_count']].astype(np.float64)
df_green['Trip_hour'] = timeFromStart_components.hours


def to_circle(k, n):
    # map days of week and hours to circle so clustering recognizes day 1 and day 7 are close and hour 0 and hour 23 are
    # close for clustering
    return(math.cos(k*2*math.pi/n), math.sin(k*2*math.pi/n))


df_green['Trip_timeofday_x'], df_green['Trip_timeofday_y'] = zip(
    *df_green['Trip_timeofday'].apply(lambda x: to_circle(x, 24)))

df_green['Trip_dayofweek_x'], df_green['Trip_dayofweek_y'] = zip(
    *df_green['Trip_dayofweek'].apply(lambda x: to_circle(x, 7)))


# Question 1

print(df_green_raw.shape)


# Question 2
sns.distplot(df_green_raw.Trip_distance)
plt.title('Trip Distance Histogram')
# plt.show()

# https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration was used as a basis for the plotting in this question
BB = (-74.3, -73.7, 40.5, 40.9)
nyc_map = plt.imread('https://aiblog.nl/download/nyc_-74.3_-73.7_40.5_40.9.png')


def distance(lat1, lon1, lat2, lon2):
    # this calculates the distance between two locations
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

fig = plt.figure(figsize=(18, 12))
plt.imshow(nyc_map, zorder=0, extent=BB)
im = plt.imshow(np.log1p(density_pickup), zorder=1, extent=BB, alpha=0.6, cmap='plasma')
plt.title('Pickup density [datapoints per sq mile]')
cbar = fig.colorbar(im, shrink=.75)
cbar.set_label('log(1 + #datapoints per sq mile)', rotation=270)
# plt.show()

fig = plt.figure(figsize=(18, 12))
plt.imshow(nyc_map, zorder=0, extent=BB)
im = plt.imshow(np.log1p(density_dropoff), zorder=1, extent=BB, alpha=0.6, cmap='plasma')
plt.title('Dropoff density [datapoints per sq mile]')
cbar = fig.colorbar(im, shrink=.75)
cbar.set_label('log(1 + #datapoints per sq mile)', rotation=270)
# plt.show()


# Question 3
tmp = df_green
tmp['Trip_hour'] = timeFromStart_components.hours
print(tmp.groupby('Trip_hour').Trip_distance.mean())
print(tmp.groupby('Trip_hour').Trip_distance.median())

# 40.6413° N, 73.7781° W JFK
# 40.7769° N, 73.8740° W LaGuardia

d1 = distance(df_green.Pickup_latitude, df_green.Pickup_longitude, 40.6413, -73.7781)
d2 = distance(df_green.Dropoff_latitude, df_green.Dropoff_longitude, 40.6413, -73.7781)
d3 = distance(df_green.Pickup_latitude, df_green.Pickup_longitude, 40.7769, -73.8740)
d4 = distance(df_green.Dropoff_latitude, df_green.Dropoff_longitude, 40.7769, -73.8740)
dist_tol = 1
airport_bool = (d1 < dist_tol) | (d2 < dist_tol) | (d3 < dist_tol) | (d4 < dist_tol)
tmp['Trip_airport_bool'] = airport_bool
Airport_Trip_data = tmp[tmp.Trip_airport_bool]
print(Airport_Trip_data.Fare_amount.describe())
print(Airport_Trip_data.Trip_distance.describe())

BB = (-74.3, -73.7, 40.5, 40.9)
nyc_map = plt.imread('https://aiblog.nl/download/nyc_-74.3_-73.7_40.5_40.9.png')
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

inds_pickup_lon = np.digitize(Airport_Trip_data.Pickup_longitude, bins_lon)
inds_pickup_lat = np.digitize(Airport_Trip_data.Pickup_latitude, bins_lat)
inds_dropoff_lon = np.digitize(Airport_Trip_data.Dropoff_longitude, bins_lon)
inds_dropoff_lat = np.digitize(Airport_Trip_data.Dropoff_latitude, bins_lat)

density_pickup, density_dropoff = np.zeros((n_lat, n_lon)), np.zeros((n_lat, n_lon))
dxdy = bin_width_miles * bin_height_miles
for i in range(n_lon):
    for j in range(n_lat):
        density_pickup[j, i] = np.sum((inds_pickup_lon == i+1) & (inds_pickup_lat == (n_lat-j))) / dxdy
        density_dropoff[j, i] = np.sum((inds_dropoff_lon == i+1) & (inds_dropoff_lat == (n_lat-j))) / dxdy

fig = plt.figure(figsize=(18, 12))
plt.imshow(nyc_map, zorder=0, extent=BB)
im = plt.imshow(np.log1p(density_pickup), zorder=1, extent=BB, alpha=0.6, cmap='plasma')
plt.title('Pickup density [datapoints per sq mile]')
cbar = fig.colorbar(im, shrink=.75)
cbar.set_label('log(1 + #datapoints per sq mile)', rotation=270)
# plt.show()

fig = plt.figure(figsize=(18, 12))
plt.imshow(nyc_map, zorder=0, extent=BB)
im = plt.imshow(np.log1p(density_dropoff), zorder=1, extent=BB, alpha=0.6, cmap='plasma')
plt.title('Dropoff density [datapoints per sq mile]')
cbar = fig.colorbar(im, shrink=.75)
cbar.set_label('log(1 + #datapoints per sq mile)', rotation=270)
# plt.show()


# Question 4
n_jobs = -1
train_labels = ['Pickup_longitude_center', 'Dropoff_longitude_center',
                'Pickup_latitude_center', 'Dropoff_latitude_center',
                'Trip_dayofweek', 'Trip_total_time', 'Trip_timeofday',
                'VendorID', 'Store_and_fwd_flag', 'RateCodeID', 'Passenger_count',
                'Trip_distance', 'Trip_timeofday_x', 'Trip_timeofday_y',
                'Trip_dayofweek_x', 'Trip_dayofweek_y', 'Payment_type', 'Trip_type', 'Extra',
                'MTA_tax', 'Fare_amount', 'Trip_hour']
# removed Trip_day since it is inappropriate for this model
df_train, df_test, y_train, y_test = train_test_split(df_green[train_labels], df_green['Trip_tippercent'])

n_clusters = 200


class my_cluster_transformer(TransformerMixin):
    # I used this to engineer additional features but it ended up not increasing performance
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

cat_features = ['Trip_dayofweek', 'Trip_hour', 'VendorID', 'RateCodeID',
                'Payment_type', 'Trip_type']
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(categories='auto'))
])

num_features = ['Pickup_longitude_center', 'Pickup_latitude_center',
                'Dropoff_longitude_center', 'Dropoff_latitude_center',
                'Passenger_count', 'Trip_distance', 'Fare_amount', 'Trip_timeofday']
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features),
    # ('cluster', cluster_transformer, cluster_features)  # reduces performance but may be able to be useful features with tuning
])


# Feature Selection
# estimator = Pipeline(steps=[('preprocess', preprocessor),
#                             ('Estimator', RandomForestRegressor(n_estimators=10, n_jobs=n_jobs, verbose=10, random_state=0))])
# estimator = estimator.fit(df_train, y_train)

# model = SelectFromModel(estimator.named_steps['Estimator'], prefit=True, threshold='mean')  # I used this to remove features
# df_train = model.transform(preprocessor.fit_transform(df_train))
# df_test = model.transform(preprocessor.fit_transform(df_test))


# Hyperparameter Tuning
estimator = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('Estimator', MLPRegressor(hidden_layer_sizes=(200, 100, 50,), verbose=0))])
param_test = {'Estimator__hidden_layer_sizes': [(200, 100, 50,), (100, 50, 25,), (100, 10, 10, 10,)]}
gsearch = GridSearchCV(estimator=estimator, param_grid=param_test, n_jobs=n_jobs, cv=5, verbose=10)
gsearch.fit(df_train, y_train)
print("MLPRegressor")
print("best_params: ", gsearch.best_params_)  # best_params:  {'Estimator__hidden_layer_sizes': (100, 50, 25)}
print("best_score: ", gsearch.best_score_)  # best_score:  0.8093715046827097
print("validation_score: ", gsearch.score(df_test, y_test))  # validation_score:  0.802029158074559

estimator = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('Estimator', GradientBoostingRegressor(verbose=0, max_depth=15,
                                            max_features='sqrt', min_samples_split=1000,
                                            min_samples_leaf=50, learning_rate=0.1, random_state=0))])
param_test = {'Estimator__n_estimators': [10, 50, 100], 'Estimator__max_features': [.3, .6, 1]}
gsearch = GridSearchCV(estimator=estimator, param_grid=param_test, n_jobs=n_jobs, cv=5, verbose=10)
gsearch.fit(df_train, y_train)
print("GradientBoostingRegressor")
print("best_params: ", gsearch.best_params_)  # {'Estimator__max_features': 0.6, 'Estimator__n_estimators': 100}
print("best_score: ", gsearch.best_score_)    # best_score:  0.8825297047528468
print("validation_score: ", gsearch.score(df_test, y_test))  # validation_score:  0.631063023717847

estimator = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('Estimator', RandomForestRegressor(n_estimators=10, n_jobs=n_jobs, verbose=0, random_state=0))])
param_test = {'Estimator__n_estimators': [10, 50, 100], 'Estimator__max_features': [.3, .6, 1]}
param_test = {'Estimator__n_estimators': [10, 50, 100], 'Estimator__max_features': [.6]}
gsearch = GridSearchCV(estimator=estimator, param_grid=param_test, n_jobs=n_jobs, cv=5, verbose=10)
gsearch.fit(df_train, y_train)
print("RandomForestRegressor")
print("best_params: ", gsearch.best_params_)  # best_params:  {'Estimator__max_features': 0.6, 'Estimator__n_estimators': 50}
print("best_score: ", gsearch.best_score_)    # best_score:  0.8853407874663699
print("validation_score: ", gsearch.score(df_test, y_test))  # validation_score:  0.5252477612371402


# Option A
df_green_speed_sub = df_green[df_green.Trip_total_time > 0]
df_green_speed_sub[['Trip_day']] = df_green_speed_sub[['Trip_day']].astype(str)
df_green_speed_sub[['Trip_hour']] = df_green_speed_sub[['Trip_hour']].astype(str)
df_green_speed_sub['Trip_avgspeed'] = (df_green_speed_sub.Trip_distance/df_green_speed_sub.Trip_total_time).clip(upper=.01)
speed_avgs_day = [np.array(df_green_speed_sub[df_green_speed_sub.Trip_day == str(i)].Trip_avgspeed) for i in range(0, 30)]
print(stats.bartlett(*speed_avgs_day))
print(stats.levene(*speed_avgs_day))
print(stats.f_oneway(*speed_avgs_day))
print(stats.kruskal(*speed_avgs_day))
results = ols('Trip_avgspeed ~ C(Trip_day)', data=df_green_speed_sub).fit()
print(results.summary())
print(df_green_speed_sub.groupby('Trip_day').Trip_avgspeed.count())
print(df_green_speed_sub.groupby('Trip_day').Trip_avgspeed.count().describe())

speed_avgs_hour = [np.array(df_green_speed_sub[df_green_speed_sub.Trip_hour == str(i)].Trip_avgspeed) for i in range(0, 24)]
print(stats.bartlett(*speed_avgs_hour))
print(stats.levene(*speed_avgs_hour))
print(stats.f_oneway(*speed_avgs_hour))
print(stats.kruskal(*speed_avgs_hour))
results = ols('Trip_avgspeed ~ C(Trip_hour)', data=df_green_speed_sub).fit()
print(results.summary())
print(df_green_speed_sub.groupby('Trip_hour').Trip_avgspeed.count())
print(df_green_speed_sub.groupby('Trip_hour').Trip_avgspeed.count().describe())
