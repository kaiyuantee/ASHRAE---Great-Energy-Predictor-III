# ASHRAE---Great-Energy-Predictor-III
ASHRAE - Great Energy Predictor III project hosted by Kaggle.
# Overview
Building's energy consumptions all over the world are often thought to have very low efficiencies and thus, resulting a very expensive electricity bill and way too much emissions which is a direct factor of climate change.
Significant amount of invesments are being made by companies and corporations to imporove building's efficiencies in order to reduce costs and emissions.
Hence, by developing accurate models of metered building energy usage for different types of meter types can help to solve these problems.
There are 4 different types of meter which consists of:
* Electricity Meter
* Chilled Water Meter
* Steam Meter
* Hot Water Meter

The data comes from over 1,000 buildings over a three-year timeframe. 
With better estimates of these energy-saving investments, large scale investors and financial institutions will be more inclined to invest in this area to enable progress in building efficiencies and help tackle the issue of climate change.
For more info about this project please refer to the link below:

https://www.kaggle.com/c/ashrae-energy-prediction

# Data Description
Assessing the value of energy efficiency improvements can be challenging as there's no way to truly know how much energy a building would have used without the improvements. 
The best we can do is to build counterfactual models. Once a building is overhauled the new (lower) energy consumption is compared against modeled values for the original building to calculate the savings from the retrofit.
More accurate models could support better market incentives and enable lower cost financing.

The challenges of this project are to build these counterfactual models across four energy types based on historic usage rates and observed weather. 
The dataset includes three years of hourly meter readings from over one thousand buildings at several different sites around the world.

# Methodology
The flow of this project is as the following:

Train data -> Preprocessing data -> Feature Engineering -> Modelling -> Ensembling Predictions -> Local Validation -> Final Predictions

The metric for this project is Root Mean Squared Logarithmic Error (RMSLE)
## Preprocessing
### Data cleaning 
by removing some anomalies and outliers such as:
* Long streaks of constant values
* Large positive & negative spikes values
* Additional anomalies determined by visual inspection
### Timezone correlation
The timezones in the weather datasets were very different from the timezones in train & test dataset. A simple function is used to correct the timezones.
### Impute missing values in weather dataset
There were a lot missing values in the weather dataset. By imputing those missing values by interpolation helped the models performance.

### Feature Engineering and Feature Selection
After countless experiments with Shapley, some features have very little importance to the model's performance and hence they are dropped out to be used as features for the final modelling part.
Some of the features that helped:
* Raw features from the full datasets except ``year_built``, ``floor_count``, ``wind_direction``, ``wind_speed``, ``sea_level_pressure``
* Categorical interactions such as concatenaation of ``building_id`` and ``meter``.
* Time series features including holiday flags and time of day features
* Lag features with mean, median, max, min, stadard deviation & skew features.

### Models
A total of 3 different types of models that were developed and trained on different subsets of the data.
Types of models:
* LightGBM Model 
* CatBoost Model 
* Keras Embedding Model

#### LightGBM Model
This model is trained on 1 model per site_id
#### Catboost Model
This model is trained on 1 model per meter_type
#### Keras Embedding Model
This model is trained on full dataset. No splitting is done

### Cross Validation 
K-Fold Cross Validation method is adopted as the CV method for all models. 4-fold CV on consecutive months as the validation set is used.

### Ensembling
Ensembling predictions from different types of models can reduce the risk of overfitting and improving the robustness of the model.
These predictions of the models are ensembled together by weighted generalized mean and the parameters are tuned using Optuna.

## Requirements
All the codes are written in Python 3.6.

Install the requirements pacakages:
```
pip install -r requirements.txt
python setup.py develop
 ```
 
