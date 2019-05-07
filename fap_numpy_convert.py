# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 00:15:56 2018

@author: User
"""
import pandas as pd
import numpy as np
import os

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, RFECV, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier

from fap_ml_preprocessing import DataImputation, DataEncoder

#feature_file = r'.\data\data20181205\feature_set8.csv' #feature_set.csv'   # or another subset of features
feature_file = 'feature_set.csv'   # or another subset of features
categorical_file = 'categorical_set.csv'

feature_set = pd.read_csv(feature_file, header = 0)
feature_set = [fe[0].strip() for fe in feature_set.values if len(fe[0].strip()) >0]

categorical_features = pd.read_csv(categorical_file, header = 0)
categorical_features = [fe[0].strip() for fe in categorical_features.values if len(fe[0].strip()) >0 and fe[0].strip() in feature_set]

datafile = os.path.join('..', 'data20181217', 'dataset_ca_e2_dtc.csv')
target = 'fail'
data_normalize = True   # normailze, Nov 27, 2018

#feature_ranking = 'stable' # rfe-lgr/kbest
#feature_ranking = 'rfe_lgr' #/kbest
#feature_ranking = 'stable' # rfe-lgr/kbest
feature_ranking = 'rfecv_lgr' #/kbest

#feature_scoring = chi2 #f_classif
feature_scoring = f_classif     # selectkbest

select_k = 30    # option

header = 0      # for datafile
index_col = 0
n_jobs = 10

sample_tag = 'ca_e2_dtc'

output_dir = os.path.join('..', 'data20181217')
    
print('feature sizes', len(feature_set), len(categorical_features))

#######################

if data_normalize:
    sample_tag += '_n'
    
if target in feature_set:
    feature_set.remove(target)

df = pd.read_csv(datafile, header = header, index_col = index_col)

fap_id = df.index.values

print(fap_id[:10])

data = df.loc[:, feature_set]
y = df.loc[:,target:target]
y = y[target].values

di = DataImputation(categorical_features, normalize = data_normalize)
de = DataEncoder(categorical_features)

data = di.fit_transform(data)
data = de.fit_transform(data)

if feature_ranking == 'kbest':
    select_k_best_classifier = SelectKBest(score_func=feature_scoring, k = select_k).fit(data, y)
elif feature_ranking == 'stable':
    select_k_best_classifier = RandomizedLasso(alpha='aic',scaling=0.5, random_state = 10).fit(data, y)
elif feature_ranking == 'rfe_lgr':
    select_k_best_classifier = RFE(LogisticRegression(class_weight = 'balanced', C = 0.1, random_state = 10), n_features_to_select=30).fit(data, y)
elif feature_ranking == 'rfecv_lgr':
    select_k_best_classifier = RFECV(LogisticRegression(class_weight = 'balanced', C = 0.1, random_state = 10), scoring = 'roc_auc', n_jobs = n_jobs).fit(data, y)

datak = select_k_best_classifier.transform(data)

print('sample sizes', len(y), data.shape)

# update
select_k = datak.shape[1]
sample_tag += str(data.shape[1])

#y = y.values

fap_id_pos = fap_id[y==1]
fap_id_neg = fap_id[y==0]

data_pos = data[y==1]
data_neg = data[y==0]
datak_pos = datak[y==1]
datak_neg = datak[y==0]

df_pos = pd.DataFrame(data_pos, index = fap_id_pos)
df_neg = pd.DataFrame(data_neg, index = fap_id_neg)
dfk_pos = pd.DataFrame(datak_pos, index = fap_id_pos)
dfk_neg = pd.DataFrame(datak_neg, index = fap_id_neg)

df = df.loc[:, feature_set+[target]]

print(data_pos.shape, data_neg.shape)

df_pos.to_csv(os.path.join(output_dir, 'dataset_pos_'+ sample_tag+'np.csv'), index_label = 'fap_id')
df_neg.to_csv(os.path.join(output_dir, 'dataset_neg_'+ sample_tag+'np.csv'), index_label = 'fap_id')
dfk_pos.to_csv(os.path.join(output_dir, 'dataset_pos_'+sample_tag+'np_k'+ str(select_k)+'_'+feature_ranking+ '.csv'), index_label = 'fap_id')
dfk_neg.to_csv(os.path.join(output_dir, 'dataset_neg_'+sample_tag+'np_k'+ str(select_k)+'_'+feature_ranking+ '.csv'), index_label = 'fap_id')

df.to_csv(os.path.join(os.path.dirname(datafile), os.path.basename(datafile).split('.')[0]+'_t.csv'), index_label = 'fap_id')
print(os.path.basename(datafile).split('.')[0])
print('done')