# +
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
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
       train.head())








