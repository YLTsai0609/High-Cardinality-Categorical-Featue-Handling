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

import sklearn
import xgboost
import tensorflow  as tf
print('np',np.__version__)
print('pd',pd.__version__)
print('tf',tf.__version__)
print('sklearn',sklearn.__version__)
print('xgb',xgboost.__version__)