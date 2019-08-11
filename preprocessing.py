# -*- coding: utf-8 -*-
# + {}
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import logging
logging.getLogger('tensorflow').disabled = True



def TargetEncoder(train, test, ft, target, 
                   min_samples_leaf=1,
                   smoothing_slope=1,
                   noise_level=0,
                   handle_missing='informed', handle_unseen='overall_mean',
                   verbose=True):
    '''
        Tree model 處理 high cardinality特徵的方法, (例如, 地區, 地址, IP...)
        由於特徵非線性且基數高，導致Tree model非常容易overfitting，
        Target encoding的中心思想為 :
        將類別特徵轉換為數值特徵，使用該特徵中每個種類的sooth_mean，
        smooth_mean可以理解為，當該種類出現的次數越多次，我們就越相信該平均值，否則資訊量太少，
        我們傾向相信總平均值。
        公式為 : smooth_mean = smoothing_factor * estimated_mean + (1 - smoothing_factor) * overall mean
        其中 smoothing_factor =  1 / (1 + np.exp(-(counts - min_samples_leaf) / smoothing_slope))
        when min_samples_leaf, smoothing_slope fixed, counts -> infinity, smoothing_factor -> 1
        min_sample_leaf 為曲線的反曲點, 當counts = min_sample_leaf 時， smoothing_factor = 0.5
        smoothing_slope 為曲線從反曲點趨近於0和1的增加量 :
        當smoothing_slope -> infinity, smoothing_factor = 0.5
        當smoothing_slope -> 0 smoothing_factor -> step function
        
    :param train: pd.DataFrame
    :param test: pd.DataFrame 
    :param ft: string 
    :param target : pd.Series with target name
    :param noise_level: float  noise level
    :param handle_missing: string 'overall_mean','informed'
    :param handle_unseen: string 'overall_mean','return_nan'
    :param verbose: bool, check the unseen in testing set
    :return: train - pd.Series, test, - pd.Series target encoding result
    
    '''

    def add_noise(s: 'pd.Series', noise_level: int) -> 'pd.Series':
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

    smoothing_factor = 1 / (1 + np.exp(- (counts - min_samples_leaf) / smoothing_slope))
    smooth_mean = (smoothing_factor * estimated_mean + (1 - smoothing_factor) * overall_mean)

    # create seen variable for test
    test_seen = test[ft].map(smooth_mean)
    unseen_ratio = test_seen.isnull().sum() / len(test)

    # return nan for unseen variable
    if handle_unseen == 'return_nan':
        return add_noise(train[ft].map(smooth_mean), noise_level), add_noise(test_seen, noise_level)

    if verbose:
        print(f'feature "{ft}" overall_mean is : ', round(overall_mean, 3))
        print(f'feature "{ft}" unssen ratio in test set is : ', round(unseen_ratio, 3))

    # DEFAULT return overall_mean for unseen variable
    return add_noise(train[ft].map(smooth_mean), noise_level), add_noise(test_seen.fillna(overall_mean), noise_level)


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



def build_and_train_model(df, target,  
                          verbose, hidden_units, epochs, LR,
                         embedding_size_dict, SEED):
    tf.set_random_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    
    def auc(y_true, y_pred):
        auc = tf.metrics.auc(y_true, y_pred)[1]
        keras.backend.get_session().run(tf.local_variables_initializer())
        return auc

    def build_input_and_embedding_layer(s, embedding_size_dict ):
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
        out = Dropout(0.3)(out)
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
    tf.initialize_all_variables()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(
    [df[feature] for feature in df.columns],
    target,
    batch_size= 32,
    callbacks=[callback],
    epochs=epochs,
    verbose=verbose,
    validation_split = .1
    )
    return history

def get_embedding_vector(feature, feature_label, embedding_vector_dict):
    return embedding_vector_dict[feature][feature_label - 1,:]

