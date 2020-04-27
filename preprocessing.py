# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:35:41 2020

@author: Allenyummy
"""

import numpy as np
import pandas as pd

def generic_groupby(df, group, feature, agg_list):
    df_tem = df.groupby(group)[feature].agg(agg_list).reset_index()
    agg_list = ['std' if x==np.std else x for x in agg_list]                
    rename_dict = dict([(x,'{}_{}_{}'.format('_'.join(group), feature, x)) for x in agg_list])
    df_tem = df_tem.rename(columns = rename_dict)
    df = pd.merge(df, df_tem, how = 'left', on = group)
    return df

#########################################
### drop nan from flbmk and flg_3ds_mk
### lgbm can handle nan
#########################################
def preprocess_init(df):
    print ('##### drop nan #####')
    print (df.isnull().sum())
    df = df.dropna(subset=['flbmk', 'flg_3dsmk'])
    # df = df.sort_values(by = ['bacno','cano','locdt','loctm']).reset_index(drop = True)
    return df
#########################################


    
#########################################
### turn Yes/No to 1/0
#########################################
def preprocess_bool(df, bool_features):
    print ('##### preprocess bool features #####')
    print (f'{bool_features}')
    for feature in bool_features:
        df[feature] = np.select([df[feature]=='Y',df[feature]=='N'],[1,0])
    return df
#########################################



#########################################
### time
#########################################
def preprocess_time(df):
    print ('##### preprocess time #####')
    conditions = [[df['locdt']<=30], [(df['locdt']>30)&(df['locdt']<=60)], [(df['locdt']>60)&(df['locdt']<=90)]]
    choices = [30, 60, 90]
    df['days'] = np.select(conditions, choices)[0]
    df['global_time'] = loctm_to_global_time(df)
    df['last_time_days'] = df.groupby(['cano','days'])['global_time'].diff(periods = 1)
    df['next_time_days'] = df.groupby(['cano','days'])['global_time'].diff(periods = -1)
    groups = ['cano','locdt']
    feature = 'global_time'
    agg_list = [np.std]
    df = generic_groupby(df, groups, feature, agg_list)
    return df

def loctm_to_global_time(df):
    df = df.copy()
    df['loctm'] = df['loctm'].astype(str)
    df['loctm'] = df['loctm'].str[:-2]
    df['hours'] = df['loctm'].str[-6:-4]
    df['hours'] = np.where(df['hours']=='', '0', df['hours']).astype(int)
    df['minutes'] = df['loctm'].str[-4:-2]
    df['minutes'] = np.where(df['minutes']=='', '0', df['minutes']).astype(int)
    df['second'] = df['loctm'].str[-2:].astype(int)
    df['loctm'] = df['hours']*60*60 + df['minutes']*60 + df['second']
    df['global_time'] = df['locdt']*24*60*60 + df['hours']*60*60+df['minutes']*60+df['second']
    return df['global_time']
#########################################



#########################################
### transaction frequency
#########################################
def preprocess_transaction_frequency(df):
    print ('##### preprocess transaction frequency #####')
    feature = 'txkey'
    agg_list = ['count']
    groups_list = [['cano','days'], ['cano','locdt'], ['bacno','locdt','mchno']]
    for groups in groups_list:
        df = generic_groupby(df, groups, feature, agg_list)
    return df
#########################################




#########################################
### change card
#########################################
def preprocess_change_card(df):
    print ('##### preprocess change card #####')
    df_tem = df.groupby(['bacno','cano','days']).agg(['max','min'])['locdt'].reset_index().sort_values(by = ['cano','max'])
    df_tem['next_card_min'] = df_tem.groupby(['bacno','days'])['min'].shift(-1)
    df_tem['next_card_min'] = np.where(df_tem['max'] - df_tem['next_card_min']>=0, np.nan, df_tem['next_card_min'])
    df_tem['diff_locdt_of_two_card'] = df_tem['max'] - df_tem['next_card_min']
    df_tem = df_tem.rename(columns = {'max':'cano_last_trans_locdt'})
    df_tem = df_tem.iloc[:,list(range(1,7))]
    df = pd.merge(df,df_tem,how = 'left', on = ['cano','days'])
    df['diff_locdt_with_last_trans_cano'] = df['locdt'] - df['cano_last_trans_locdt']
    df['diff_locdt_with_last_trans_days_cano'] = df['days'] - df['cano_last_trans_locdt']
    return df
#########################################


#########################################
### mchno
#########################################
def preprocess_mchno(df):
    print ('##### preprocess mchno #####')
    df = bacno_mchno_locdt_head_tail_diff(df)
    df = cano_days_mchno_index(df)
    return df

def bacno_mchno_locdt_head_tail_diff(df):
    df_head = df.groupby(['bacno','mchno','days']).head(1)[['bacno','mchno','days','locdt']]
    df_head = df_head.rename(columns = {'locdt' : 'locdt_head'})
    df_tail = df.groupby(['bacno','mchno','days']).tail(1)[['bacno','mchno','days','locdt']]
    df_tail = df_tail.rename(columns = {'locdt' : 'locdt_tail'})
    df_head = pd.merge(df_head, df_tail, how = 'left', on = ['bacno','mchno','days'])
    df_head['bacno_mchno_locdt_head_tail_diff'] = df_head['locdt_tail'] - df_head['locdt_head']
    df = pd.merge(df,df_head, how = 'left', on =['bacno','mchno','days'])
    return df

def cano_days_mchno_index(df):
    df['cano_days_mchno_index'] = 1
    df['cano_days_mchno_index'] = df.groupby(['cano','days','mchno'])['cano_days_mchno_index'].cumsum()
    return df
#########################################



#########################################
### conam
#########################################
def preprocess_conam(df):
    print ('##### preprocess conam #####')
    df = preprocess_global_conam_max_min(df)
    df = diff_with_zero_conam_time(df)
    df = preprocess_mean_conam_bacno(df)
    df = preprocess_mean_conam_etymd(df)
    df = preprocess_mean_conam_csmcu(df)
    df = preprocess_mean_conam_stocn(df)
    return df

def preprocess_global_conam_max_min(df):
    groups = ['cano','locdt']
    agg_list = ['min','max']
    feature = 'conam'
    df = generic_groupby(df, groups, feature, agg_list)
    return df

def diff_with_zero_conam_time(df):
    df = df.copy()
    df_tem = df[df['conam']==0].drop_duplicates(subset = ['cano','locdt'],keep = 'first')
    df_tem = df_tem.rename(columns = {'global_time' : 'conam_zero_trans_global_time'})
    df = pd.merge(df, df_tem[['cano','locdt','conam_zero_trans_global_time']], how = 'left' , on = ['cano','locdt'])
    df['diff_gtime_with_conam_zero_trans_locdt'] = df['global_time'] - df['conam_zero_trans_global_time']
    return df

def preprocess_mean_conam_bacno(df):
    groups = ['bacno']
    feature = 'conam'
    agg_list = ['mean', 'median']
    df = generic_groupby(df, groups, feature, agg_list)
    df['diff_bacno_conam_mean'] = df['bacno_conam_mean'] - df['conam']
    df['diff_bacno_conam_median'] = df['bacno_conam_median'] - df['conam']
    return df
    
def preprocess_mean_conam_etymd(df):
    groups = ['etymd']
    feature = 'conam'
    agg_list = ['mean', 'median']
    df = generic_groupby(df, groups, feature, agg_list)
    df['diff_etymd_conam_mean'] = df['etymd_conam_mean'] - df['conam']
    df['diff_etymd_conam_median'] = df['etymd_conam_median'] - df['conam']
    return df
    
def preprocess_mean_conam_csmcu(df):
    groups = ['csmcu']
    feature = 'conam'
    agg_list = ['mean', 'median']
    df = generic_groupby(df, groups, feature, agg_list)
    df['diff_csmcu_conam_mean'] = df['csmcu_conam_mean'] - df['conam']
    df['diff_csmcu_conam_median'] = df['csmcu_conam_median'] - df['conam']
    return df

def preprocess_mean_conam_stocn(df):
    groups = ['stocn']
    feature = 'conam'
    agg_list = ['mean', 'median']
    df = generic_groupby(df, groups, feature, agg_list)
    df['diff_stocn_conam_mean'] = df['stocn_conam_mean'] - df['conam']
    df['diff_stocn_conam_median'] = df['stocn_conam_median'] - df['conam']
    return df
#########################################



#########################################
### split data set
######################################### 
def preprocess_split_train_dev_test(df):
    train = df.query("locdt <= 60")
    dev = df.query("locdt > 60 and locdt <= 75")
    test = df.query("locdt > 75 and locdt <= 90")
    return train, dev, test
#########################################


#########################################
### generate x y for supervised learning
#########################################
def generate_x_y(df, features, label):
    return df[features], df[label]
#########################################

    
    
# # 若在嫌疑商戶類別中消費，註記1，反之註記0
# mcc = data_dropna.groupby(by=['mcc', 'fraud_ind']).size().reset_index(name='counts')
# mcc_fraud_list = mcc.query('fraud_ind == 1 & counts >= 50')['mcc'].tolist()
# data_dropna['mcc_mk'] = np.where(data_dropna['mcc'].isin(mcc_fraud_list), 1, 0)

# # 若在嫌疑特店中消費，註記1，反之註記0
# mchno = data_dropna.groupby(by=['mchno', 'fraud_ind']).size().reset_index(name='counts')
# mchno_fraud_list = mchno.query('fraud_ind == 1 & counts >= 50')['mchno'].tolist()
# data_dropna['mchno_mk'] = np.where(data_dropna['mchno'].isin(mchno_fraud_list), 1, 0)

# # 若在嫌疑收單行中，註記1，反之註記0
# acqic = data_dropna.groupby(by=['acqic', 'fraud_ind']).size().reset_index(name='counts')
# acqic_fraud_list = acqic.query('fraud_ind == 1 & counts >= 50')['acqic'].tolist()
# data_dropna['acqic_mk'] = np.where(data_dropna['acqic'].isin(acqic_fraud_list), 1, 0)
    
    
    
    
    
    

