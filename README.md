
# Industrial Copper Modeling

The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data. 

Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer . You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.



## Solution Approach

The solution includes the following steps:

Exploring skewness and outliers in the dataset.

Transform the data into a suitable format and perform any necessary cleaning and pre-processing steps.

ML Regression model which predicts continuous variable ‘Selling_Price’.

ML Classification model which predicts Status: WON or LOST.

Creating a streamlit page where you can insert each column value and you will get the Selling_Price predicted value or Status(Won/Lost)

## Features

- User Friendly
- Dynamic performance
- Colourful Theme
- Valuable Insights from the prediction
## Streamlit App

- Able to give input
- Predict the Selling Price
- Predict the Status
## Installation

Install following packages

```bash
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
```
    
## Deployment

To deploy this project run

```bash
  streamlit run copperstream.py
```


## Demo

Here is the link of the demo video


https://www.linkedin.com/feed/update/urn:li:activity:7202700191798317056/