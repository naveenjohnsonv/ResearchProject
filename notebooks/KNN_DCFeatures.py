import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import matplotlib.pyplot as plt
from Dataset import merged_df

# Features division based on the columns specified
# DC_POWER: columns 1 to 24
# IRRADIATION: columns 25 to 48
# AMBIENT_TEMPERATURE: columns 49 to 72
# MODULE_TEMPERATURE: columns 73 to 96

X_dc_power = merged_df.iloc[:, 1:25].values
X_irradiation = merged_df.iloc[:, 25:49].values
X_ambient_temp = merged_df.iloc[:, 49:73].values
X_module_temp = merged_df.iloc[:, 73:97].values

# Combine features for different combinations
X_dc_irr = np.hstack((X_dc_power, X_irradiation))
X_dc_amb = np.hstack((X_dc_power, X_ambient_temp))
X_dc_mod = np.hstack((X_dc_power, X_module_temp))

# Target variable
y = merged_df.iloc[:, 0].values

# 1. Split the data into training and test sets
test_size = 7
train_size = int(len(y) - test_size)

# Splitting for each feature set
X_train_dc, X_test_dc = X_dc_power[:train_size], X_dc_power[train_size:]
X_train_irr, X_test_irr = X_irradiation[:train_size], X_irradiation[train_size:]
X_train_amb, X_test_amb = X_ambient_temp[:train_size], X_ambient_temp[train_size:]
X_train_mod, X_test_mod = X_module_temp[:train_size], X_module_temp[train_size:]

X_train_dc_irr, X_test_dc_irr = X_dc_irr[:train_size], X_dc_irr[train_size:]
X_train_dc_amb, X_test_dc_amb = X_dc_amb[:train_size], X_dc_amb[train_size:]
X_train_dc_mod, X_test_dc_mod = X_dc_mod[:train_size], X_dc_mod[train_size:]

y_train, y_test = y[:train_size], y[train_size:]

# Hyperparameter tuning and model training for each feature set
def train_knn(X_train, y_train):
    k_values = list(range(3, 7))
    rmse_values = []
    
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        tscv = TimeSeriesSplit(n_splits=5)
        scores = -cross_val_score(knn, X_train, y_train, cv=tscv, scoring='neg_root_mean_squared_error')
        avg_rmse = np.mean(scores)
        rmse_values.append(avg_rmse)

    # Find best k
    best_rmse = min(rmse_values)
    best_k = k_values[rmse_values.index(best_rmse)]

    # Train KNN model with best k
    knn = KNeighborsRegressor(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    
    return knn, best_k, k_values, rmse_values

# Train and predict for each feature set
knn_dc, best_k_dc, k_values_dc, rmse_values_dc = train_knn(X_train_dc, y_train)
y_pred_dc = knn_dc.predict(X_test_dc)
rmse_dc = np.sqrt(mse(y_test, y_pred_dc))

knn_dc_irr, best_k_dc_irr, k_values_dc_irr, rmse_values_dc_irr = train_knn(X_train_dc_irr, y_train)
y_pred_dc_irr = knn_dc_irr.predict(X_test_dc_irr)
rmse_dc_irr = np.sqrt(mse(y_test, y_pred_dc_irr))

knn_dc_amb, best_k_dc_amb, k_values_dc_amb, rmse_values_dc_amb = train_knn(X_train_dc_amb, y_train)
y_pred_dc_amb = knn_dc_amb.predict(X_test_dc_amb)
rmse_dc_amb = np.sqrt(mse(y_test, y_pred_dc_amb))

knn_dc_mod, best_k_dc_mod, k_values_dc_mod, rmse_values_dc_mod = train_knn(X_train_dc_mod, y_train)
y_pred_dc_mod = knn_dc_mod.predict(X_test_dc_mod)
rmse_dc_mod = np.sqrt(mse(y_test, y_pred_dc_mod))

# Plotting RMSE evolution for each feature set
plt.figure(figsize=(10, 6))
plt.plot(k_values_dc, rmse_values_dc, label='DC Power', marker='o')
plt.plot(k_values_dc_irr, rmse_values_dc_irr, label='DC Power + Irradiation', marker='X')
plt.plot(k_values_dc_amb, rmse_values_dc_amb, label='DC Power + Ambient Temperature', marker='o')
plt.plot(k_values_dc_mod, rmse_values_dc_mod, label='DC Power + Module Temperature', marker='o')
plt.xlabel('k value')
plt.ylabel('RMSE')
plt.title('RMSE Evolution for Different k Values')
plt.legend()
plt.grid(True)
plt.show()

# Create a dictionary to mimic the `predictions_dict` used in the plotting code
predictions_dict = {
    str(['DC_POWER']): pd.Series(y_pred_dc, index=merged_df.index[-test_size:]),
    str(['DC_POWER', 'IRRADIATION']): pd.Series(y_pred_dc_irr, index=merged_df.index[-test_size:]),
    str(['DC_POWER', 'AMBIENT_TEMPERATURE']): pd.Series(y_pred_dc_amb, index=merged_df.index[-test_size:]),
    str(['DC_POWER', 'MODULE_TEMPERATURE']): pd.Series(y_pred_dc_mod, index=merged_df.index[-test_size:]),
}

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))
y_test_series = pd.Series(y[-test_size:].flatten(), index=merged_df.index[-test_size:])
ax.plot(y_test_series.index, y_test_series, color='#2ca02c', marker='o', label='Actual')

# Plot model predictions for each chosen feature set
ax.plot(predictions_dict[str(['DC_POWER'])].index, predictions_dict[str(['DC_POWER'])], color='#ff7f0e', label=f'Predicted for DC (k={best_k_dc})', marker='o', linestyle='--')
ax.plot(predictions_dict[str(['DC_POWER', 'IRRADIATION'])].index, predictions_dict[str(['DC_POWER', 'IRRADIATION'])], color='b', label=f'Predicted for DC + Irr. (k={best_k_dc_irr})', marker='X', linestyle='--')
ax.plot(predictions_dict[str(['DC_POWER', 'AMBIENT_TEMPERATURE'])].index, predictions_dict[str(['DC_POWER', 'AMBIENT_TEMPERATURE'])], color='r', label=f'Predicted for DC + Amb. Temp. (k={best_k_dc_amb})', marker='o', linestyle='--')
ax.plot(predictions_dict[str(['DC_POWER', 'MODULE_TEMPERATURE'])].index, predictions_dict[str(['DC_POWER', 'MODULE_TEMPERATURE'])], color='y', label=f'Predicted for DC + Mod. Temp. (k={best_k_dc_mod})', marker='o', linestyle='--')
ax.set_xlabel('Date [yyyy-mm-dd]')
ax.set_ylabel('DC Power Generated [kW]')
plt.xticks(rotation=45)
plt.tight_layout()
ax.legend()
ax.grid(True)  

# Add RMSE value next to the last point of each plot
last_point_x = y_test_series.index[-1]
ax.text(last_point_x, predictions_dict[str(['DC_POWER'])].iloc[-1], f'{rmse_dc:.3f}', color='#ff7f0e')
ax.text(last_point_x, predictions_dict[str(['DC_POWER', 'IRRADIATION'])].iloc[-1], f'{rmse_dc_irr:.3f}', color='b')
ax.text(last_point_x, predictions_dict[str(['DC_POWER', 'AMBIENT_TEMPERATURE'])].iloc[-1], f'{rmse_dc_amb:.3f}', color='r')
ax.text(last_point_x, predictions_dict[str(['DC_POWER', 'MODULE_TEMPERATURE'])].iloc[-1], f'{rmse_dc_mod:.3f}', color='y')

plt.show()

# Scatter plot for actual vs. predicted values
fig, ax = plt.subplots(figsize=(5, 5))
y_test_series = pd.Series(y[-test_size:].flatten(), index=merged_df.index[-test_size:])
sc1 = ax.scatter(y_test_series, predictions_dict[str(['DC_POWER'])], color='w', label=f'DC (k={best_k_dc})', marker='*', edgecolors='k', s=64)
sc2 = ax.scatter(y_test_series, predictions_dict[str(['DC_POWER', 'IRRADIATION'])], color='b', label=f'DC + Irr. (k={best_k_dc_irr})', marker='X', edgecolors='k', s=64)
sc3 = ax.scatter(y_test_series, predictions_dict[str(['DC_POWER', 'AMBIENT_TEMPERATURE'])], color='r', label=f'DC + Amb. Temp. (k={best_k_dc_amb})', marker='<', edgecolors='k', s=64)
sc4 = ax.scatter(y_test_series, predictions_dict[str(['DC_POWER', 'MODULE_TEMPERATURE'])], color='y', label=f'DC + Mod. Temp. (k={best_k_dc_mod})', marker='h', edgecolors='k', s=64)
min_val = min(y_test_series.min(), min(y_pred_series.min() for y_pred_series in predictions_dict.values()))
max_val = max(y_test_series.max(), max(y_pred_series.max() for y_pred_series in predictions_dict.values()))
ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='r', label='Base x = y Line')
ax.set_xlabel('Actual DC Power Generated [kW]')
ax.set_ylabel('Predicted DC Power Generated [kW]')
plt.tight_layout()
ax.legend()
ax.grid(True)  

plt.show()
