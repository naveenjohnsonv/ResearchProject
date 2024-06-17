# Comparative Analysis of Machine Learning Models in Predictive Manners

This project compares the performance of various k-NN and Lasso regression models that predict the amount of DC power generated from a solar plant using weather conditions and lagged features as predictors. The study demonstrates how time series analysis can be leveraged for building accurate solar power generation forecasting models.

## Table of Contents

- [Approach](#approach)
- [Methodology](#methodology)
  - [Models](#models)
  - [Experiments](#experiments)
  - [Data](#data)
  - [Data Preprocessing](#data-preprocessing)
- [Results](#results)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Approach

The project uses two main machine learning techniques: Lasso regression and k-Nearest Neighbors (kNN).

- **Lasso Regression:** Lasso regression is a linear regression technique that includes regularization to prevent overfitting. It is particularly useful when dealing with a large number of features, as it can automatically select the most relevant features and shrink the coefficients of less important ones to zero. This helps to simplify the model and improve its interpretability. In the context of time series analysis, Lasso regression can be applied to forecast future values based on past observations. By incorporating lagged features (e.g., previous day's DC power), the model can capture temporal dependencies and patterns in the data, leading to more accurate predictions.

- **k-Nearest Neighbors (kNN):** kNN is a non-parametric method used for classification and regression. In the context of this project, kNN regression is used to predict DC power output based on the features of the nearest neighbors in the training data. The simplicity of kNN makes it a useful benchmark model and can also provide insights into the structure of the data.

## Methodology

### Models

- **Lasso with default alpha:** A simple implementation of Lasso regression with a default regularization parameter.
- **LassoCV with hyperparameter tuning:** Utilizes time series cross-validation to find the optimal alpha (regularization parameter) for the Lasso model.
- **kNN with hyperparameter tuning:** A basic k-Nearest Neighbors model with an automated selection of the best k neighbors.
- **kNNCV with hyperparameter tuning:** Employs time series cross-validation on top of hyperparameter tuning.

### Experiments

The project conducts several experiments to evaluate the performance of different models:

1. **Univariate one-day-ahead prediction with 1 lag:** Uses only the previous day's DC power to predict the current day's DC power.
2. **Multivariate (multiple features) one-day-ahead prediction with 1 lag:** Uses lagged daily average of multiple features (DC power, ambient temperature, module temperature, and irradiation) to predict the current day's DC power.
3. **Multivariate (hourly train data as features) one-day-ahead prediction with 1 lag:** Utilizes hourly weather data and lagged features to predict the current day's DC power. Different feature combinations are explored using LassoCV to find the best alpha and feature set.

### Data

The datasets used in this project is sourced from Kaggle: [Solar Power Generation Data](https://www.kaggle.com/datasets/ef9660b4985471a8797501c8970009f36c5b3515213e2676cf40f540f0100e54). The project uses two datasets:

1. **Generation Data:** Contains information on DC power, AC power, daily yield, and total yield recorded at 15-minute intervals over a 34 days period.
2. **Weather Sensor Data:** Includes data on ambient temperature, module temperature, and irradiation recorded at 15-minute intervals over a 34 days period.

### Data Preprocessing

- **Data Loading:** Both datasets are loaded and the date-time columns are converted to appropriate formats.
- **Resampling:** The data is resampled to daily and hourly frequencies.
- **Merging:** Generation and weather data are merged based on date-time.
- **Feature Engineering:** Lagged features are created for time series analysis.
- **Imputation:** Missing values are imputed using KNN imputer.

## Results

The project presents results for various models with different feature combinations. It includes plots showing:

- Actual vs. predicted DC power for different models.
- RMSE values for different feature sets and cross-validation splits.

Root Mean Squared Error (RMSE) was chosen as the metric for evaluation. The following table presents a comparison of different regression models and their performance in predicting the current day DC power. The best-performing models and feature sets are highlighted, providing insights into the most relevant factors for solar power prediction.

| Model  | Sample Frequency | Sample Features | No. of neighbours (n) | Regularization strength (&alpha;) | RMSE |
|:--------|:----------------|:---------------|:---:|:---:|:----:|
| Lasso | Day | D | - | 1.0 | 0.637 |
| Lasso | Day | D, A, I, M | - | 1.0 | 0.637 |
| LassoCV | Hour | D | - | 0.225| 0.542 |
| LassoCV | Hour | A | - | 0.253 | 0.593 |
| **LassoCV** | **Hour** | **D, A** | - | **0.206** | **0.538** |
| LassoCV | Hour | D, A, I | - | 0.206 | 0.538 |
| kNNCV | Hour | D | 6 | - | 0.565 |
| kNNCV | Hour | A | 6 | - | 0.536 |
| kNNCV | Hour | I | 4 | - | 0.570 |
| kNNCV | Hour | M | 6 | - | 0.561 |
| kNNCV | Hour | D | 6 | - | 0.565 |
| kNNCV | Hour | D, A | 4 | - | 0.544 |
| kNNCV | Hour | D, I | 6 | - | 0.565 |
| kNNCV | Hour | D, M | 4 | - | 0.529 |

Here D, A, I, M represents the features DC Power, Ambient Temperature, Irradiation, Module Temperature respectively. 

## Installation

### Prerequisites

- Python
- Jupyter Notebook

### Setup

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/naveenjohnsonv/ResearchProject
    cd ResearchProject
    ```

2. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebooks provided in the `notebooks` directory to explore the methods and reproduce the results.

    ```bash
    jupyter notebook
    ```

### Dependencies

The project relies on the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Project Structure

```
ResearchProject/
├── data/
│   ├── Plant_1_Generation_Data.csv
│   ├── Plant_1_Weather_Sensor_Data.csv
├── notebooks/
│   ├── solar-power-prediction-lasso.ipynb
├── report/
│   ├── tmp
├── README.md
├── requirements.txt
```

## Contributors

1. Naveen JOHNSON VALLAVANATT (Naveen.Johnson-Vallavanatt@eurecom.fr)
2. Aymeric LEGROS (Aymeric.Legros@eurecom.fr)

## Acknowledgements

The authors would like to acknowledge the providers of the Solar Power Generation Dataset used in this project.

## License

This project is licensed under the [MIT License](LICENSE).
