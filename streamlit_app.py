import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics

st.title('🎈 LightGBM')

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')

# Separate data as X, y
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Model training
model = lgb.LGBMRegressor(learning_rate=0.09, max_depth=-5, random_state=42, force_col_wise=True)

model.fit(X_train, y_train)

# Model performance
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

st.write('$\text{Train} R^2$: {:.3f}'.format(r2_score(y_train, y_train_pred)))
st.write('$\text{Test} R^2$: {:.3f}'.format(r2_score(y_test, y_test_pred)))
