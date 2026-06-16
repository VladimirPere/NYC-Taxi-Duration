import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error

data = pd.read_csv('/home/vladimir/Downloads/Data/NYCtaxi.csv')

def haversine(lat1: float, lon1: float, lat2: float, lon2: float):
    """Outputs the total distance between the initial and destination coordinates relative to earth"""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    theta = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(theta))

# Converting Datetime strings to pd.datetime
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
data['hour'] = data['pickup_datetime'].dt.hour
data['day_of_week'] = data['pickup_datetime'].dt.dayofweek
data['month'] = data['pickup_datetime'].dt.month
data['day_of_month'] = data['pickup_datetime'].dt.day
data['is_weekend'] = (data['pickup_datetime'].dt.dayofweek >= 5).astype(int)

# Adding total Distance traveled
data['distance_km'] = haversine(
    data['pickup_latitude'], data['pickup_longitude'],
    data['dropoff_latitude'], data['dropoff_longitude']
)

# Adding Rush Hour and Direction
data['is_rush_hour'] = data['hour'].isin([7,8,9,17,18,19]).astype(int)
data['direction'] = np.arctan2(
    data['dropoff_latitude'] - data['pickup_latitude'],
    data['dropoff_longitude'] - data['pickup_longitude']
)

# Outliers, Between 1min and 24hrs
data = data[(data['trip_duration'] >= (60 * 1)) & (data['trip_duration'] <= (60 * 60 * 24))]

# Making data
target_name = 'trip_duration'
target = np.log1p(data[target_name])

# 'id', 'vendor_id' and 'store_and_fwd_flag' dropped due to having no correlation with target
# 'pickup_datetime' removed because datetime was put in seperate columns above
# 'dropoff_datetime' removed due to trivializing the calculations
training_data = data.drop(columns=['id', 
                                   'vendor_id', 
                                   'pickup_datetime', 
                                   'dropoff_datetime', 
                                   'store_and_fwd_flag', 
                                   target_name])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Unnecessary, added for best practice
            ('scalar', StandardScaler()),
        ]), training_data.columns.tolist())
    ],
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor( 
    # learning_rate=.09,
    # max_iter=1100,
    # max_depth=17,
    # min_samples_leaf=300,
    random_state=2)),
    ]
)

train_data, test_data, train_target, test_target = train_test_split(
    training_data, 
    target, 
    test_size=.2, 
    random_state=2
)

param_dist = {
    'regressor__learning_rate': [.08, .09, .1],
    'regressor__max_iter': [1000, 1100, 1200],
    'regressor__max_depth': [16, 17, 18, 19],
    'regressor__min_samples_leaf': [275, 300, 325]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_dist,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,                              # n_jobs=1 originally used due to computer not being strong enough
)

grid_search.fit(train_data, train_target)

best_est = grid_search.best_estimator_

preds = best_est.predict(test_data)
score = root_mean_squared_error(test_target, preds)

print(f'Best parameters: {grid_search.best_params_}')
print(f'The best RMSLE score of training data: {-grid_search.best_score_}')
print(f'The best RMSLE score of test data using 20% of the data is: {score}')


# OUTPUTS
# Best parameters: {'regressor__learning_rate': 0.09, 'regressor__max_depth': 17, 'regressor__max_iter': 1100, 'regressor__min_samples_leaf': 300}
# The best RMSLE score of training data: 0.36149121788090877
# The best RMSLE score of test data using 20% of the data is: 0.36043515904410806


# Testing data
test_raw = pd.read_csv('/home/vladimir/Downloads/Data/nyc_test.csv')

test_raw['pickup_datetime'] = pd.to_datetime(test_raw['pickup_datetime'])
test_raw['hour'] = test_raw['pickup_datetime'].dt.hour
test_raw['day_of_week'] = test_raw['pickup_datetime'].dt.dayofweek
test_raw['month'] = test_raw['pickup_datetime'].dt.month
test_raw['day_of_month'] = test_raw['pickup_datetime'].dt.day
test_raw['is_weekend'] = (test_raw['pickup_datetime'].dt.dayofweek >= 5).astype(int)

# Adding total Distance traveled
test_raw['distance_km'] = haversine(
    test_raw['pickup_latitude'], test_raw['pickup_longitude'],
    test_raw['dropoff_latitude'], test_raw['dropoff_longitude']
)

# Adding Rush Hour and Direction
test_raw['is_rush_hour'] = test_raw['hour'].isin([7,8,9,17,18,19]).astype(int)
test_raw['direction'] = np.arctan2(
    test_raw['dropoff_latitude'] - test_raw['pickup_latitude'],
    test_raw['dropoff_longitude'] - test_raw['pickup_longitude']
)

ids = test_raw['id']

kaggle_test = test_raw.drop(columns=[ 'id', 
                                    'vendor_id', 
                                    'pickup_datetime', 
                                    'store_and_fwd_flag', 
                                    ])

best_est.fit(training_data, target)

log_preds = best_est.predict(kaggle_test)
preds = np.expm1(log_preds)

submission = pd.DataFrame({'id': ids, 'trip_duration': preds})
submission.to_csv('nyc_submission.csv', index=False)