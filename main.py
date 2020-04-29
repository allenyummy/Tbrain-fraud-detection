# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:13:08 2020

@author: Allenyummy
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import preprocessing as pre
import pylab   
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

#%% features
raw_bool_features = ['ecfg',
                     'flbmk',
                     'flg_3dsmk',
                     'insfg',
                     'ovrlt']

raw_categorial_features = ['contp',
                           'stscd',
                           'etymd',
                           'stocn',
                           'mcc',
                           'csmcu',
                           'hcefg',
                           'mchno',
                           'acqic',
                           'scity'] 

raw_contiuous_feautres = ['loctm',
                          'conam',
                          'iterm']

transaction_frequency_feautres = ['cano_days_txkey_count',
                                  'cano_locdt_txkey_count',
                                  'bacno_locdt_mchno_txkey_count' ]

time_feautres = ['last_time_days',
                 'next_time_days',
                 'cano_locdt_global_time_std']


change_card_feautres = ['diff_locdt_with_last_trans_cano',
                        'diff_locdt_of_two_card']

conam_feautres = [
                   'cano_locdt_conam_min',
                   'cano_locdt_conam_max',
                   'diff_gtime_with_conam_zero_trans_locdt',
                   'bacno_conam_mean',
                   'bacno_conam_median',
                   'diff_bacno_conam_mean',
                   'diff_bacno_conam_median',
                   'etymd_conam_mean',
                   'etymd_conam_median',
                   'diff_etymd_conam_mean',
                   'diff_etymd_conam_median',
                   'csmcu_conam_mean',
                   'csmcu_conam_median',
                   'diff_csmcu_conam_mean',
                   'diff_csmcu_conam_median',
                   'stocn_conam_mean',
                   'stocn_conam_median',
                   'diff_stocn_conam_mean',
                   'diff_stocn_conam_median'
                  ]

mchno_features = ['bacno_mchno_locdt_head_tail_diff',
                  'cano_days_mchno_index']

# special_feautures = ['mchno_in_normal_mchno_list',
#                     'mchno_in_fraud_mchno_list',
#                     'conam_in_fraud_conam_list',
#                     'diff_with_first_fraud_locdt']

base_features =  (    raw_bool_features 
                    + raw_categorial_features
                    + raw_contiuous_feautres
                    + transaction_frequency_feautres
                    + time_feautres
                    + change_card_feautres
                    + conam_feautres 
                    + mchno_features
                        )

# base_features =  (    raw_bool_features 
#                     + raw_categorial_features
#                     + raw_contiuous_feautres
#                         )

label = 'fraud_ind'

#%% preprocessing
preprocess = False
if preprocess:
    df = pd.read_csv('train.csv')
    # df = pre.preprocess_init(df)
    df = pre.preprocess_bool(df, raw_bool_features)
    df = pre.preprocess_time(df)
    df = pre.preprocess_transaction_frequency(df)
    df = pre.preprocess_change_card(df)
    df = pre.preprocess_mchno(df)
    df = pre.preprocess_conam(df)
    df.to_pickle('df_preprocessed.pkl')
else:
    df = pd.read_pickle('df_preprocessed.pkl')

# df = pd.read_csv('train.csv')
# df = pre.preprocess_init(df)
# df = pre.preprocess_bool(df, raw_bool_features)
train, dev, test = pre.preprocess_split_train_dev_test(df)
train_x, train_y = pre.generate_x_y(train, base_features, label)
dev_x, dev_y = pre.generate_x_y(dev, base_features, label)
test_x, test_y = pre.generate_x_y(test, base_features, label)

#%% model
## f1: 39 ~ 41 
# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier(max_depth=None, criterion='entropy', random_state=0)
# model = model.fit(train_x, train_y)

# ## f1: 52 ~ 53 
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
# model = model.fit(train_x, train_y)

# from sklearn.cluster import KMeans
# model = KMeans(n_clusters=2, random_state=0, verbose=1).fit(train_x)

def F1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

import lightgbm as lgb
Lgb_params = {
        'task': 'train',
        'is_unbalance': False,
        'num_leaves': 10000,
        'max_depth': None,
        'min_data_in_leaf': 3000,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'objective': 'cross_entropy',
        'metric': 'cross_entropy',
        'lambda_l1': 0.0,
        'lambda_l2': 0.0,
        }

trn_data = lgb.Dataset(train_x, label=train_y)
dev_data = lgb.Dataset(dev_x, label=dev_y)
test_data = lgb.Dataset(test_x, label=test_y)
evals_result = dict()
Lgb = lgb.train(params = Lgb_params,
                train_set = trn_data,
                valid_sets = [trn_data, dev_data, test_data],
                valid_names = ['train', 'dev', 'test'],
                categorical_feature = raw_categorial_features,
                # learning_rates=lambda iter: 0.05 * (0.99 ** iter),
                num_boost_round = 2000,
                early_stopping_rounds = 1000,
                verbose_eval = 10,                
                feval = F1_score,                
                evals_result=evals_result)

#%%
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(evals_result['train']['cross_entropy'], 'g-', label='train_cross_entropy')
ax.plot(evals_result['dev']['cross_entropy'],   'r-', label='dev_cross_entropy')
ax.plot(evals_result['test']['cross_entropy'],  'b-', label='test_cross_entropy')
ax.legend()
ax.set_xlabel('epoch', fontsize=20)
ax.set_ylabel('loss', fontsize=20)     
pylab.tick_params(which='major', width=4)         

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(evals_result['train']['f1'], 'g-', label='train_f1_score')
ax.plot(evals_result['dev']['f1'],   'r-', label='dev_f1_score')
ax.plot(evals_result['test']['f1'],  'b-', label='test_f1_score')
ax.legend()
ax.set_xlabel('epoch', fontsize=20)
ax.set_ylabel('f1', fontsize=20)     
pylab.tick_params(which='major', width=4)

#%% performance
# pred_y_train = model.predict(train_x)
# pred_y_dev = model.predict(dev_x)
# pred_y_test = model.predict(test_x)

# print (f'train acc: {model.score(train_x, train_y)}')
# print (f'train f1: {f1_score(train_y, pred_y_train)}')
# print (confusion_matrix(train_y, pred_y_train))
# print (classification_report(train_y, pred_y_train, digits=4))
# print ('-----------------------------')
# print (f'dev acc: {model.score(dev_x, dev_y)}')
# print (f'dev f1: {f1_score(dev_y, pred_y_dev)}')
# print (confusion_matrix(dev_y, pred_y_dev))
# print (classification_report(dev_y, pred_y_dev, digits=4))
# print ('-----------------------------')
# print (f'test acc: {model.score(test_x, test_y)}')
# print (f'test f1: {f1_score(test_y, pred_y_test)}')
# print (confusion_matrix(test_y, pred_y_test))
# print (classification_report(test_y, pred_y_test, digits=4))








