# -*- coding: utf-8 -*-
# + {}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# # Embedding

train_test_split?

data_embedding_train = X_train.copy()
data_embedding_val = X_val.copy()
# data_inner_train = train_test_split(random_state=SEED, test_size = 0.1 * len(train))

class EmbeddingMapping():
    """
    Helper class for handling categorical variables
    An instance of this class should be defined for each categorical variable we want to use.
    """
    def __init__(self, s : 'pd.Series') -> None:
        values = s.unique().tolist()
        self.feature_name = s.name
        # dictionary mapper
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
LR = .0001 # Learning rate
EPOCHS = 20 # Default number of training epochs (i.e. cycles through the training data)
hidden_units = (16,4) # Size of our hidden layers
 
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

# Google's paper tell us a good herustic size = number of category ** 0.25
embedding_size_dict = {'RESOURCE': 9,
                       'MGR_ID': 8 ,
                      'ROLE_FAMILY_DESC': 6,
                      'ROLE_FAMILY': 3,
                      'ROLE_CODE':  4,
                      }

def build_and_train_model(df, target, embedding_size_dict=embedding_size_dict,  
                          verbose=2, epochs=EPOCHS):
    tf.set_random_seed(1); np.random.seed(1); random.seed(1)
    def build_input_and_embedding_layer(s, embedding_size_dict = embedding_size_dict):
        assert s.name in embedding_size_dict.keys()
        input_layer = keras.Input(shape=(1,), name = s.name)
        embedded_layer = keras.layers.Embedding(s.max() + 1, 
                                               embedding_size_dict[s.name],
                                               input_length=1, name = f'{s.name}_embedding')(input_layer)
        return input_layer, embedded_layer
    # Create embedding layer
    emb_layer_list = []
    input_layer_list = []
    for feature in df.columns:
        input_layer, embedded_layer = build_input_and_embedding_layer(df[feature], embedding_size_dict=embedding_size_dict)
        input_layer_list.append(input_layer)
        emb_layer_list.append(embedded_layer)
    # concat
    concatenated = keras.layers.Concatenate()(emb_layer_list)
    out = keras.layers.Flatten()(concatenated)
    
    # hidden layers
    for n_hidden in hidden_units:
        out = keras.layers.Dense(n_hidden, activation='relu')(out)
        
    # output layer
    out = keras.layers.Dense(1, activation='sigmoid', name='prediction')(out)
    
    # model
    model = keras.Model(
    inputs = input_layer_list,
    outputs = out)
#     model.summary()

    model.compile(
    tf.train.AdamOptimizer(LR),
    loss='binary_crossentropy',
    metrics=[auc],
    )
    # train it
    # TODO FailedPreconditionError 
    # This could mean that the variable was uninitialized. Not found: Resource localhost/dense_33/kernel/N10tensorflow3VarE does not exist.   
    tf.initialize_all_variables()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(
    [df[feature] for feature in df.columns],
    y_train,
    batch_size= 32,
    callbacks=[callback],
    epochs=epochs,
    verbose=verbose,
    validation_split = .1
    )
    return history

history = build_and_train_model(df = data_embedding_train,
                      target = y_train,
                      verbose = 2)      
# + {}
history_FS = (15, 5)
def plot_history(histories, keys=('loss','auc',), train=True, figsize=history_FS):
    if isinstance(histories, tf.keras.callbacks.History):
        histories = [ ('', histories) ]
        print(histories, type(histories))
    print(histories, type(histories))
    for key in keys:
        plt.figure(figsize=history_FS)
        for name, history in histories:
            val = plt.plot(history.epoch, history.history['val_'+key],
                           '--', label=str(name).title()+' Val')
            
            val = plt.plot(history.epoch, np.full(len(history.epoch), 1.0), color='k',linestyle='--', alpha=.2)
            if train:
                plt.plot(history.epoch, history.history[key], color=val[0].get_color(), alpha=.5,
                         label=str(name).title()+' Train')
                
        plt.xlabel('Epochs')
        plt.ylabel(key.replace('_',' ').title())
        plt.legend()
        plt.title(key)

        plt.xlim([0,max(max(history.epoch) for (_, history) in histories)])

plot_history([ 
    ('embedding', history)
])
# -

model = history.model
embedding_vector_dict = {}
embedding_vector_column = {}
for feature in data_embedding_train.columns:
    print(f' processing ... {feature}_embedding')
    embedding_vector_dict[feature] = model.get_layer(f'{feature}_embedding').get_weights()[0]
    embedding_vector_column[feature] = [f'{feature}_emb_{i}' for i in range(embedding_vector_dict[feature].shape[1])]
print('Setup compelete')


def get_embedding_vector(feature, feature_label):
    return embedding_vector_dict[feature][feature_label - 1,:]


# +
# data_embedding_val['Ori_Des_pair_label'] = data_embedding_val['Ori_Des_pair_label'].astype(int)

data_embedding_train = data_embedding_train.astype(int).reset_index(drop=True)
data_embedding_val = data_embedding_val.astype(int).reset_index(drop=True)

data_embedding_train_prepared = data_embedding_train.copy()
data_embedding_val_prepared = data_embedding_val.copy()

for feature in data_embedding_train.columns:
    # train converting
    tmp_train_features_df = pd.DataFrame(get_embedding_vector(feature, data_embedding_train[feature]),
                                             columns=embedding_vector_column[f'{feature}'])

    # test converting
    tmp_test_features_df = pd.DataFrame(get_embedding_vector(feature,data_embedding_val[feature]),
                                    columns=embedding_vector_column[f'{feature}'])
    # train concat
    data_embedding_train_prepared = pd.concat([data_embedding_train_prepared, tmp_train_features_df],
                                               axis = 1)
    # test concat
    data_embedding_val_prepared = pd.concat([data_embedding_val_prepared, tmp_test_features_df],
                                             axis = 1)
    
# -

from pandas.core.common import flatten
EMBEDDING_COLS = list(flatten(embedding_vector_column.values()))

data_embedding_train_prepared[EMBEDDING_COLS].head()

embedding_logit_score = get_score(LogReg_model, 
                                  data_embedding_train_prepared[EMBEDDING_COLS],
                                  y_train,
                                  data_embedding_val_prepared[EMBEDDING_COLS],
                                  y_val)
embedding_xgb_score = get_score(xgb_model,
                                data_embedding_train_prepared[EMBEDDING_COLS],
                                y_train,
                                data_embedding_val_prepared[EMBEDDING_COLS],
                                y_val)
print('Logistic Regression Embedding encoding:', embedding_logit_score)
print('Xgboost Embedding encoding:', embedding_xgb_score)
# # Embedding
