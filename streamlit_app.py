import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
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
model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-5, random_state=42)

model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)],
          verbose=20,eval_metric='logloss')

# Model performance
st.write('Train: {:.3f}'.format(model.score(x_train,y_train)))
st.write('Test: {:.3f}'.format(model.score(x_test,y_test)))
