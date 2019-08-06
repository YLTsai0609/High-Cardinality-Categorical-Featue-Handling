# -*- coding: utf-8 -*-
# + {}
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
import os
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# set seed and path
SEED = 17
# ROOT = os.chdir('./data/train.csv')
# os.listdir(ROOT)
ROOT = Path('.')
TRAIN_FILE = ROOT / 'data/train.csv'

# check your current working directory
# # !pwd
# -

train = pd.read_csv(TRAIN_FILE)
y = train['ACTION']
train = train[['RESOURCE', 'MGR_ID',
               'ROLE_FAMILY_DESC', 'ROLE_FAMILY',
               'ROLE_CODE']]
display(train.dtypes,
       train.shape,
       train.head(),
       'TARGET',
       y.value_counts())


# * Data 簡介
#
# > RESOURCE
#
# > MGR_ID
#
# > ROLE_FAMILY_DESC
#
# > ROLE_FAMILY
#
# > ROLE_CODE

# helper function
def get_score(model, X, y, X_val, y_val):
    model.fit(X, y)
    y_pred = model.predict_proba(X_val)[:,1]
    score = roc_auc_score(y_val, y_pred)
    return score


LogReg_model = LogisticRegression(random_state=SEED)
xgb_model = XGBClassifier(random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.2, random_state=SEED)

for col in X_train.columns:
    print(col , 'uniques : ',X_train[f'{col}'].nunique())

# # label encoding

baseline_logit_score = get_score(LogReg_model, X_train, y_train, X_val, y_val)
baseline_xgb_score = get_score(xgb_model, X_train, y_train, X_val, y_val)
print('Logistic Regression Label encoding:', baseline_logit_score)
print('Xgboost Label encoding:', baseline_xgb_score)

# # One hot encoding

# +
from sklearn.preprocessing import OneHotEncoder

one_hot_enc = OneHotEncoder(sparse=True)

print('Original number of features:', X_train.shape[1])
data_ohe_train = (one_hot_enc.fit_transform(X_train))
data_ohe_val = (one_hot_enc.transform(X_val))
print('Features after OHE', data_ohe_train.shape[1])

# -

ohe_logit_score = get_score(LogReg_model, data_ohe_train, y_train, data_ohe_val, y_val)
ohe_xgb_score = get_score(xgb_model, data_ohe_train, y_train, data_ohe_val, y_val)
print('Logistic Regression OneHot encoding:', ohe_logit_score)
print('Xgboost OneHot encoding:', ohe_xgb_score)

# # Target encoding
#

# # Regularized Target encoding

# # Embedding
