import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
import matplotlib.pyplot as plt

# Read the generation data
generation_data_p1 = pd.read_csv('/Users/moi/Desktop/project research/archive/Plant_1_Generation_Data.csv')
generation_data_p1['DC_POWER'] = generation_data_p1['DC_POWER'].div(1000)
generation_data_p1['AC_POWER'] = generation_data_p1['AC_POWER'].div(100)
generation_data_p1['DATE_TIME'] = pd.to_datetime(generation_data_p1['DATE_TIME'])
generation_data_p1.set_index('DATE_TIME', inplace=True)

# Read the weather data
weather_data_p1 = pd.read_csv('/Users/moi/Desktop/project research/archive/Plant_1_Weather_Sensor_Data.csv')
weather_data_p1['DATE_TIME'] = pd.to_datetime(weather_data_p1['DATE_TIME'])
weather_data_p1.set_index('DATE_TIME', inplace=True)

# Resample both datasets to daily data
df_weather = weather_data_p1.drop('SOURCE_KEY', axis=1).resample('D').mean()
df_generation = generation_data_p1.drop('SOURCE_KEY', axis=1).resample('D').mean()

# Merge the resampled daily datasets
df = pd.merge(df_generation, df_weather, left_index=True, right_index=True, how='inner', suffixes=('_gen', '_weather'))

# Add a column for the previous day's DC_POWER
df['DC_POWER-1'] = df['DC_POWER'].shift(1)

# Drop unwanted columns
df = df.drop(['PLANT_ID_gen', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD', 'PLANT_ID_weather'], axis=1)

# Read the CSV file containing the generation data for computing the daily average DC_POWER
generation_data = pd.read_csv('/Users/moi/Desktop/project research/archive/Plant_1_Generation_Data.csv')
generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'])

# Group the data by date and calculate the average DC_POWER
average_dc_power = generation_data.groupby(generation_data['DATE_TIME'].dt.date)['DC_POWER'].mean() / 1000

# Shift the average DC_POWER to represent the next day's value
average_dc_power = average_dc_power.shift(-1).iloc[:-1]

# Align the index of the main DataFrame to match the length of average_dc_power
df = df.iloc[:-1]

# Insert the average next day's DC_POWER into the main DataFrame
df.insert(0, 'DC_Power_Next_Day_AVG', average_dc_power.values)

# Create KNN imputer object
imputer = KNNImputer(n_neighbors=5)

# Fit and transform data to impute missing values
imputed_data = imputer.fit_transform(df)

# Recreate DataFrame with imputed values and original column names
df_imputed = pd.DataFrame(imputed_data, index=df.index, columns=df.columns)

# Prepare the features and target
X_multivariate = df_imputed.drop(columns=['DC_Power_Next_Day_AVG'])
y_multivariate = df_imputed['DC_Power_Next_Day_AVG']
X_univariate = df_imputed[['DC_POWER-1']]
y_univariate = df_imputed['DC_Power_Next_Day_AVG']

# Split the data into training and testing sets for both models using the same parameters as the provided code
test_size = 7
train_size = int(len(y_univariate) - test_size)

# Train and test splits for univariate model
X_train_uni, X_test_uni = X_univariate[:train_size], X_univariate[train_size:]
y_train_uni, y_test_uni = y_univariate[:train_size], y_univariate[train_size:]

# Train and test splits for multivariate model
X_train_multi, X_test_multi = X_multivariate[:train_size], X_multivariate[train_size:]
y_train_multi, y_test_multi = y_multivariate[:train_size], y_multivariate[train_size:]

# Train and evaluate the univariate model
univariate_model = KNeighborsRegressor(n_neighbors=7)
univariate_model.fit(X_train_uni, y_train_uni)
y_pred_uni = univariate_model.predict(X_test_uni)
rmse_uni = np.sqrt(mse(y_test_uni, y_pred_uni))

# Train and evaluate the multivariate model
multivariate_model = KNeighborsRegressor(n_neighbors=7)
multivariate_model.fit(X_train_multi, y_train_multi)
y_pred_multi = multivariate_model.predict(X_test_multi)
rmse_multi = np.sqrt(mse(y_test_multi, y_pred_multi))

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))

# Actual values
y_test_series = pd.Series(y_test_uni.values, index=df_imputed.index[-test_size:])
ax.plot(y_test_series.index, y_test_series, color='#2ca02c', marker='o', label='Actual')

# Predictions
ax.plot(y_test_series.index, y_pred_uni, color='#ff7f0e', marker='o', linestyle='--', label=f'Univariate (RMSE={rmse_uni:.3f})')
ax.plot(y_test_series.index, y_pred_multi, color='b', marker='o', linestyle='--', label=f'Multivariate (RMSE={rmse_multi:.3f})')

ax.set_xlabel('Date [yyyy-mm-dd]')
ax.set_ylabel('DC Power Generated [kW]')
plt.xticks(rotation=45)
plt.tight_layout()
ax.legend()
ax.grid(True)  

plt.show()

# Scatter plot for actual vs. predicted values
fig, ax = plt.subplots(figsize=(5, 5))
sc1 = ax.scatter(y_test_series, y_pred_uni, color='#ff7f0e', label=f'Univariate (RMSE={rmse_uni:.3f})', edgecolors='k', s=64)
sc2 = ax.scatter(y_test_series, y_pred_multi, color='b', label=f'Multivariate (RMSE={rmse_multi:.3f})', edgecolors='k', s=64)
min_val = min(y_test_series.min(), min(y_pred_uni.min(), y_pred_multi.min()))
max_val = max(y_test_series.max(), max(y_pred_uni.max(), y_pred_multi.max()))
ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='r', label='Base x = y Line')
ax.set_xlabel('Actual DC Power Generated [kW]')
ax.set_ylabel('Predicted DC Power Generated [kW]')
plt.tight_layout()
ax.legend()
ax.grid(True)  
plt.show()
