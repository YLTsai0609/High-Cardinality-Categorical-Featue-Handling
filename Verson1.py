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

def TargetEncoder(train, test, ft, target, 
                   min_samples_leaf=1,
                   smoothing=1,
                   noise_level=0,
                   handle_missing='informed', handle_unseen='overall_mean',
                   verbose=True):
    '''
        Tree model 處理 high cardinality特徵的方法, (例如, 地區, 地址, IP...)
    在此例中，處理(出發-到達) 特徵, cardinality數 = 4429
    Target encoding with global overall_mean and estimated_mean
    mu = (m / m + n) * overall_mean + (n / m+n) * estimated_mean
    n -> the value_counts of the category
    m -> factor -> when m = n -> mu = 1/2 overall_overall_mean + 1/2 estimated overall_mean


    :param train: df
    :param test: df
    :param ft: str
    :param on: series with name
    :param noise_level:
    :param handle_missing: 'overall_mean','informed'
    :param handle_unseen: 'overall_mean','return_nan'
    :param verbose:
    :return:
    # TODO min_sample_leaf --> the number to take account for smoothing
    # TODO smotthing --> smotthing parameter
    # TODO Ref https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
    # TODO use class to do that
    '''

    def add_noise(s: 'Pd.Series', noise_level: int) -> 'pd.Series':
        return s * (1 + noise_level * np.random.randn(len(s)))
    
    train = pd.concat([train, target], axis=1)
    target_name = target.name
    overall_mean = target.mean()

    # handling missing value - filling
    train[ft].fillna('is_missing')
    test[ft].fillna('is_missing')

    # Compute the number of values and the estimated_mean of each group
    agg = train.groupby(ft)[target_name].agg(['count', 'mean'])
    counts = agg['count']
    estimated_mean = agg['mean']

    if handle_missing == 'overall_mean':
        # assign zero to group "is_missing", then smooth will be overall_mean
        counts.is_missing = 0

    # Compute the "smoothed" overall_means
    # DEFAULT take missing value is informed

    smoothing = 1 / (1 + np.exp(- (counts - min_samples_leaf) / smoothing))
    smooth = (smoothing * estimated_mean + (1 - smoothing) * overall_mean)

    # create seen variable for test
    test_seen = test[ft].map(smooth)
    unseen_ratio = test_seen.isnull().sum() / len(test)

    # return nan for unseen variable
    if handle_unseen == 'return_nan':
        return add_noise(train[ft].map(smooth), noise_level), add_noise(test_seen, noise_level)

    if verbose:
        print(f'feature "{ft}" overall_mean is : ', round(overall_mean, 3))
        print(f'feature "{ft}" unssen ratio in test set is : ', round(unseen_ratio, 3))

    # DEFAULT return overall_mean for unseen variable
    return add_noise(train[ft].map(smooth), noise_level), add_noise(test_seen.fillna(overall_mean), noise_level)


data_te_train = X_train.copy()
data_te_val = X_val.copy()
for feature in data_te_train.columns:
    if feature == y_train.name:
        continue
    data_te_train[f'{feature}_te'], data_te_val[f'{feature}_te'] = TargetEncoder(data_te_train,
                                                                           X_val, ft=feature,
                                                                           target=y_train,
                                                                      min_samples_leaf=3, smoothing=2)

te_col = [feature for feature in data_te_train.columns
                              if feature.endswith('_te')]
# Training
ohe_logit_score = get_score(LogReg_model, data_te_train[te_col], y_train, data_te_val[te_col], y_val)
ohe_xgb_score = get_score(xgb_model, data_te_train[te_col], y_train, data_te_val[te_col], y_val)
print('Logistic Regression Target encoding:', ohe_logit_score)
print('Xgboost Target encoding:', ohe_xgb_score)

# # Regularized Target encoding

# +
FOLDS = StratifiedKFold(n_splits=5)
data_te_cv_train = X_train.copy()
data_te_cv_val = X_val.copy()
oof_list = []
for n_fold, (trn_idx, val_idx) in enumerate(FOLDS.split(data_te_cv_train, y_train)):
    X_train_te, y_train_te = data_te_cv_train.iloc[trn_idx], y_train.iloc[trn_idx]
    X_val_te, y_val_te = data_te_cv_train.iloc[val_idx], y_train.iloc[val_idx]
    
    for feature in X_train_te.columns:
        X_train_te[f'{feature}_te'], X_val_te[f'{feature}_te'] = TargetEncoder(X_train_te,
                                                                               X_val_te,
                                                                               ft=feature,
                                                                               target=y_train_te,
                                                                               min_samples_leaf=3,
                                                                               smoothing=2,
                                                                               verbose=False)
    oof_list.append(X_val_te.reset_index(drop=True))


te_col = [feature for feature in data_te_train.columns
                              if feature.endswith('_te')]

data_te_cv_train = pd.concat(oof_list,ignore_index=True)

# +
# Dealing with validation set

data_te_cv_train[te_col].head()
# -











# # Embedding
















