# +
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold, KFold
from xgboost import XGBClassifier
from pathlib import Path
from pandas.core.common import SettingWithCopyWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from sklearn.preprocessing import OneHotEncoder
from pandas.core.common import flatten

from preprocessing import get_score, TargetEncoder, EmbeddingMapping,\
build_and_train_model, get_embedding_vector

SEED = 1
ROOT = Path('.')
TRAIN_FILE = ROOT / 'data/train.csv'


train = pd.read_csv(TRAIN_FILE)
y = train['ACTION']
train = train[['RESOURCE', 'MGR_ID',
               'ROLE_FAMILY_DESC', 'ROLE_FAMILY',
               'ROLE_CODE']]


# -


# splits train/val
X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.2, random_state=SEED, stratify = y)

# +
# Preprocessing
# ONE HOT
one_hot_enc = OneHotEncoder(sparse=True)
data_ohe_train = (one_hot_enc.fit_transform(X_train))
data_ohe_val = (one_hot_enc.transform(X_val))
# TARGET 
data_te_train = X_train.copy()
data_te_val = X_val.copy()
print('-'*60)
print('Perform Target Encoding')
print('-'*60)
for feature in data_te_train.columns:
    data_te_train[f'{feature}_te'], data_te_val[f'{feature}_te'] = TargetEncoder(data_te_train,
                                                                           X_val, ft=feature,
                                                                           target=y_train,
                                                                      min_samples_leaf=3, smoothing_slope=2)
# Regularized Target
data_te_cv_train = X_train.copy()
data_te_cv_val = X_val.copy()
FOLDS = StratifiedKFold(n_splits=5, random_state=SEED)
VAL_FOLDS = KFold(n_splits=5, random_state=SEED)

oof_trn_list = []
oof_test_list = []
print('-'*60)
print('Perform Regularized Target Encoding')
print('-'*60)
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
                                                                               smoothing_slope=2,
                                                                               verbose=False)
        _, X_test_te[f'{feature}_te'] =TargetEncoder(X_train_te,
                                                    X_test_te,
                                                    ft=feature,
                                                    target=y_train_te,
                                                    min_samples_leaf=3,
                                                    smoothing_slope=2,
                                                    verbose=False)
    
    oof_trn_list.append(X_val_te.reset_index(drop=True))
    oof_test_list.append(X_test_te.reset_index(drop=True))

te_col = [feature for feature in data_te_train.columns
                              if feature.endswith('_te')]

data_te_cv_train = pd.concat(oof_trn_list,ignore_index=True)
data_te_cv_val = pd.concat(oof_test_list, ignore_index=True)

# Embedding

data_embedding_train = X_train.copy()
data_embedding_val = X_val.copy()

for feature in data_embedding_train.columns:
    Mapper = EmbeddingMapping(data_embedding_train[feature])
    data_embedding_train[f'{feature}'] = Mapper.mapping(data_embedding_train[f'{feature}'])
    data_embedding_val[f'{feature}'] = Mapper.mapping(data_embedding_val[f'{feature}'])
    
embedding_size_dict = {'RESOURCE': 9,
                       'MGR_ID': 8 ,
                      'ROLE_FAMILY_DESC': 7,
                      'ROLE_FAMILY': 3,
                      'ROLE_CODE':  4,
                      }

LR = .00005
EPOCHS = 80
hidden_units = (8,8)
print('-'*60)
print('Perform NN - Embedding')
print('-'*60)
history = build_and_train_model(df = data_embedding_train,
                      target = y_train,
                      verbose=2, 
                      hidden_units=hidden_units,
                      epochs=EPOCHS,
                      LR=LR,
                      embedding_size_dict = embedding_size_dict,
                      SEED=SEED)
# Get vector
model = history.model
embedding_vector_dict = {}
embedding_vector_column = {}
for feature in data_embedding_train.columns:
    print(f' processing ... {feature}_embedding')
    embedding_vector_dict[feature] = model.get_layer(f'{feature}_embedding').get_weights()[0]
    embedding_vector_column[feature] = [f'{feature}_emb_{i}' for i in range(embedding_vector_dict[feature].shape[1])]
print('Setup compelete')

# drop_index to keep pd.concat works
data_embedding_train = data_embedding_train.astype(int).reset_index(drop=True)
data_embedding_val = data_embedding_val.astype(int).reset_index(drop=True)

data_embedding_train_prepared = data_embedding_train.copy()
data_embedding_val_prepared = data_embedding_val.copy()

for feature in data_embedding_train.columns:
    # train converting
    tmp_train_features_df = pd.DataFrame(get_embedding_vector(feature, data_embedding_train[feature],
                                                              embedding_vector_dict),
                                             columns=embedding_vector_column[f'{feature}'])

    # test converting
    tmp_test_features_df = pd.DataFrame(get_embedding_vector(feature,data_embedding_val[feature],
                                                            embedding_vector_dict),
                                    columns=embedding_vector_column[f'{feature}'])
    # train concat
    data_embedding_train_prepared = pd.concat([data_embedding_train_prepared, tmp_train_features_df],
                                               axis = 1)
    # test concat
    data_embedding_val_prepared = pd.concat([data_embedding_val_prepared, tmp_test_features_df],
                                             axis = 1)
    
EMBEDDING_COLS = list(flatten(embedding_vector_column.values()))
    



# +
# Result
xgb_model = XGBClassifier(random_state=SEED)
baseline_xgb_score = get_score(xgb_model, X_train, y_train, X_val, y_val)
ohe_xgb_score = get_score(xgb_model, data_ohe_train, y_train, data_ohe_val, y_val)
target_xgb_score = get_score(xgb_model, data_te_train[te_col], y_train, data_te_val[te_col], y_val)
te_cv_xgb_score = get_score(xgb_model, data_te_cv_train[te_col], y_train, data_te_cv_val[te_col], y_val)
embedding_xgb_score = get_score(xgb_model,
                                data_embedding_train_prepared[EMBEDDING_COLS],
                                y_train,
                                data_embedding_val_prepared[EMBEDDING_COLS],
                                y_val)

print('Xgboost Label encoding:', baseline_xgb_score)
print('Xgboost OneHot encoding:', ohe_xgb_score)
print('Xgboost Target encoding:', target_xgb_score)
print('Xgboost Regularized Target encoding:', te_cv_xgb_score)
print('Xgboost Embedding encoding:', embedding_xgb_score)
# -


