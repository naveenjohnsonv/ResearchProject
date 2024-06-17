import pandas as pd
from sklearn.impute import KNNImputer

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

# Merge the resampled datasets
df = pd.merge(df_generation, df_weather, left_index=True, right_index=True, how='inner', suffixes=('_gen', '_weather'))

# Add a column for previous day's DC_POWER
df['DC_POWER-1'] = df['DC_POWER'].shift(1)

# Drop unwanted columns
df = df.drop(['PLANT_ID_gen', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD', 'PLANT_ID_weather'], axis=1)

# Resample both datasets to hourly data
df_weather_hrs = weather_data_p1.drop('SOURCE_KEY', axis=1).resample('h').mean()
df_generation_hrs = generation_data_p1.drop('SOURCE_KEY', axis=1).resample('h').mean()

# Merge the resampled hourly datasets
df_hrs = pd.merge(df_generation_hrs, df_weather_hrs, left_index=True, right_index=True, how='inner', suffixes=('_gen', '_weather'))

# Add an hour column
df_hrs['hour'] = df_hrs.index.hour

# Shift hour data to columns
df_hrs_2 = df_hrs.set_index('hour', append=True).unstack(level=-1)[['DC_POWER', 'IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']]

# Rename the columns
df_hrs_2.columns = [f'{column}-{hour:02d}' for column, hour in df_hrs_2.columns]

# Keep only the first row (00:00:00) for each day
df_hrs_f = df_hrs_2.groupby(df_hrs_2.index.date).first()

# Create KNN imputer object
imputer = KNNImputer(n_neighbors=5) # Use 5 nearest neighbors

# Fit and transform data to impute missing values
imputed_data = imputer.fit_transform(df_hrs_f)

# Recreate DataFrame with imputed values and original column names
df_hrs_imputed = pd.DataFrame(imputed_data, index=df_hrs_f.index, columns=df_hrs_f.columns)
merged_df = df_hrs_imputed.iloc[:-1]

# Read the CSV file containing the generation data
data = pd.read_csv('/Users/moi/Desktop/project research/archive/Plant_1_Generation_Data.csv')
# Convert the DATE_TIME column to datetime type
data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'])

# Group the data by date and calculate the average DC_POWER
average_dc_power = data.groupby(data['DATE_TIME'].dt.date)['DC_POWER'].mean()/1000

average_dc_power = average_dc_power.shift(-1)
average_dc_power = average_dc_power[:-1]  # Remove the last element

# Save average DC_POWER per day to CSV
average_dc_power.to_csv('/Users/moi/Desktop/project research/jsp_Data.csv', index=False)

# Merge average_dc_power with merged_df
merged_df.insert(0, 'DC_Power_Next_Day_AVG', average_dc_power.values)

# Save merged DataFrame to CSV
merged_df.to_csv('/Users/moi/Desktop/project research/merged_Data.csv', index=True)