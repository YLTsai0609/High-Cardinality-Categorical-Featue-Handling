# -*- coding: utf-8 -*-
# + {}
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os
from pathlib import Path
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


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
# 這次的示範資料集是從Kaggle上2013年的[Amazon員工訪問權限預測挑戰賽](https://www.kaggle.com/c/amazon-employee-access-challenge)中取得
# 這個資料集，該資料集收集了Amazon公司中各個員工針對每個資源(例如網頁的logging)的訪問紀錄，當員工屬於能夠取得訪問權限時，系統卻不給訪問，又要向上申請才能取得權限，一來一往浪費的非常多時間，因此這場比賽希望能夠建構模型，減少員工訪問權限所需的人工流程，我們取出5個特徵如下 :
#
#
# * Feature (X)
#
# > RESOURCE : 資源ID
#
# > MGR_ID : 員工主管的ID 
#
# > ROLE_FAMILY_DESC : 員工類別擴展描述 (例如 軟體工程的零售經理)
#
# > ROLE_FAMILY : 員工類別 (例如 零售經理)
#
# > ROLE_CODE : 員工角色編碼 (例如 經理)
#
# * Target (Y)
#
# > ACTION : 
#
#  >> 1 : RESOURCE 訪問權限取得
#  
#  >> 0 : RESOURCE 禁止訪問

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
FOLDS = StratifiedKFold(n_splits=5, random_state=SEED)
VAL_FOLDS = KFold(n_splits=5, random_state=SEED)

data_te_cv_train = X_train.copy()
data_te_cv_val = X_val.copy()
oof_trn_list = []
oof_test_list = []

for (trn_idx, val_idx), (_, test_idx) in zip(FOLDS.split(data_te_cv_train, y_train),
                                             VAL_FOLDS.split(data_te_cv_val)):
    X_train_te, y_train_te = data_te_cv_train.iloc[trn_idx], y_train.iloc[trn_idx]
    X_val_te, y_val_te = data_te_cv_train.iloc[val_idx], y_train.iloc[val_idx]
    X_test_te = data_te_cv_val.iloc[test_idx]
    
    for feature in X_train_te.columns:
        X_train_te[f'{feature}_te'], X_val_te[f'{feature}_te'] = TargetEncoder(X_train_te,
                                                                               X_val_te,
                                                                               ft=feature,
                                                                               target=y_train_te,
                                                                               min_samples_leaf=3,
                                                                               smoothing=2,
                                                                               verbose=False)
        _, X_test_te[f'{feature}_te'] =TargetEncoder(X_train_te,
                                                    X_test_te,
                                                    ft=feature,
                                                    target=y_train_te,
                                                    min_samples_leaf=3,
                                                    smoothing=2,
                                                    verbose=False)
    
    oof_trn_list.append(X_val_te.reset_index(drop=True))
    oof_test_list.append(X_test_te.reset_index(drop=True))

te_col = [feature for feature in data_te_train.columns
                              if feature.endswith('_te')]

data_te_cv_train = pd.concat(oof_trn_list,ignore_index=True)
data_te_cv_val = pd.concat(oof_test_list, ignore_index=True)
# -

te_cv_logit_score = get_score(LogReg_model, data_te_cv_train[te_col], y_train, data_te_cv_val[te_col], y_val)
te_cv_xgb_score = get_score(xgb_model, data_te_cv_train[te_col], y_train, data_te_cv_val[te_col], y_val)
print('Logistic Regression Regularized Target encoding:', te_cv_logit_score)
print('Xgboost Regularized Target encoding:', te_cv_xgb_score)
# # Embedding

data_embedding_train = X_train.copy()
data_embedding_val = X_val.copy()


class EmbeddingMapping():
    """
    Helper class for handling categorical variables
    An instance of this class should be defined for each categorical variable we want to use.
    """
    def __init__(self, s : 'pd.Series') -> None:
        # get a list of unique values
        values = s.unique().tolist()
        self.feature_name = s.name
        # Set a dictionary mapping from values to integer value
        self.embedding_dict = {value: int_value + 1 for int_value, value in enumerate(values)}
        # The num_values will be used as the input_dim when defining the embedding layer.
        # we set unseen values as maximum value + 1 
        self.num_values = len(values) + 1

    def mapping(self,s : 'pd.Series', verbose = True) -> None:
        tmp_series = s.map(self.embedding_dict)
        unseen_ratio = round(tmp_series.isnull().sum() / len(s), 3)
        if verbose:
            print(f'Feature "{self.feature_name}"')
            print(f'    encode {self.num_values -1} values to label, {self.num_values} will be the unseen value ')
            print(f'    unssen ratio is : ', unseen_ratio)
        return tmp_series.fillna(self.num_values)


for feature in data_embedding_train.columns:
    Mapper = EmbeddingMapping(data_embedding_train[feature])
    data_embedding_train[f'{feature}'] = Mapper.mapping(data_embedding_train[f'{feature}'])
    data_embedding_val[f'{feature}'] = Mapper.mapping(data_embedding_val[f'{feature}'])

# +
LR = .002 # Learning rate
EPOCHS = 50 # Default number of training epochs (i.e. cycles through the training data)
hidden_units = (32,4) # Size of our hidden layers

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

def build_and_train_model(s, target, mapper_class, embedding_dimension=8,  
                          verbose=2, epochs=EPOCHS):
    tf.set_random_seed(1); np.random.seed(1); random.seed(1)
# -














