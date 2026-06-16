Data From: "New York City Taxi Trip Duration" Hosted by Kaggle
Found Here: https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview

Data: ~1.45 Million taxi trips in New York City with 8 features

# Data Configuring
- The 'pickup_datetime' feature was reformatted into pandas datetime format and split into parts. For easier regression.
- 'distance_km' was added as the straight line distance between the pickup and dropoff coordinates using the Haversine formula.
- 'is_rush_hour' added to emphasize how rush hour affects taxi times. Rush hour was defined to be 7-9am and 5-7pm.
- Outliers were then removed so results were less affected
- 'dropoff_datetime' removed due it revealing the target, thus trivializing the prediction
- 'store_and_fwd_flag' removed due to having no correlation to target

# Process
- All columns were imputed using median and scaled using the standard scaler
- HistGradientBoostingRegressor used due to its robustness to noise, mixed features, and efficiency with very large datasets
- Data was split for testing purposes
- RandomizedSearchCV was first used, later GridSearchCV was used for more accurate hyperparameterizing
- The best estimator was then used for the Kaggle test data, which was also transformed as the training data was.

80% CV RMSLE: .36149
20% test RMSLE: .36044
Kaggle test RMSLE: .42101

The gap between the test CV and Kaggle score likely due to distribution differences in the test set or overfitting during tuning.
More hyperparametizing is needed to close the gap.
