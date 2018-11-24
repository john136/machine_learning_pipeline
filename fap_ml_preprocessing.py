# -*- coding: utf-8 -*-
"""
fap_ml_preprocessing.py

a library of machine learning for data preprocessing: data imputation and categorical encoder

Created on Wed Nov  7 13:54:43 2018

johnli@bigr.io: John Li

"""
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.base import BaseEstimator, TransformerMixin, clone

import numpy as np
import pandas as pd


#from imblearn.over_sampling import SMOTE
#from imblearn.combine import SMOTETomek

class DataImputation(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, categoricalna = 'UNQ'):  # categoricalna from Tarun
        self.categorical_features = categorical_features
        self.categoricalna = categoricalna
        return
    
    def fit(self, X, y = None):
        imputation_dir = {}  # saved for future test data
        n_numerical = 0
        for column in X:
            #print(column)
            try:
                if column in self.categorical_features:
                    if X[column].isnull().values.any():
                        X[column].fillna(value=self.categoricalna,inplace=True)
                    continue
                
                X[column] = pd.to_numeric(X[column], errors = 'coerce') # Nov 9, 2018
                
                if(X[column].isnull().values.all()):
                    # X[column].fillna(value=0,inplace=True)
                    imputation_dir[column] = 0 #X_train[column].min()/6   # John
                elif(X[column].notnull().values.any() and V[column].min() != 0 ):
                    # X[column].fillna(value=X[column].min()/6,inplace=True)
                    imputation_dir[column] = X[column].min()/6    # John
                else:
                    # X[column].fillna(value=0,inplace=True)    
                    imputation_dir[column] = 0 #X_train[column].min()/6   # John
                n_numerical +=1
            except:
                raise Exception('column <%s> is not a numerical feature. Defining it as a categorical feature in *.categorical_features' % column)
        
        # save results
        
        self.imputation_dir = imputation_dir
                
        return self
    
    def transform(self, X, y = None):
        for column in X:
            #print(column)
            try:
                if column in self.categorical_features:
                    if X[column].isnull().values.any():
                        X[column].fillna(value=self.categoricalna,inplace=True)
                    continue
                
                X[column] = pd.to_numeric(X[column], errors = 'coerce') # Nov 10, 2018, John
                
                X[column].fillna(value=self.imputation_dir[column],inplace=True) 

                # if(X[column].isnull().values.all()):
                #     X[column].fillna(value=0,inplace=True)
                #     imputation_dir[column] = 0 #X_train[column].min()/6   # John
                # elif(X[column].notnull().values.any() and V[column].min() != 0 ):
                #     X[column].fillna(value=imputation_dir[column],inplace=True)
                #     # imputation_dir[column] = X[column].min()/6    # John
                # else:
                #     X[column].fillna(value=0,inplace=True)    
                #     # imputation_dir[column] = 0 #X_train[column].min()/6   # John
                # n_numerical +=1
            except:
                raise Exception('column <%s> is not a numerical feature or not an valid feature in the original training data. Defining it as a categorical feature in *.categorical_features; or removing the column' % column)        
        return X
        
    def fit_transform(self, X, y = None):
        imputation_dir = {}  # saved for future test data
        n_numerical = 0
        
        for column in X:
            #print(column)
#            if column == 'VERSIONSKENNUNGEN.VERSIONSKENNUNGEN_PARAMETERSATZ_DAEMPFERHYDRAULIK':
#                print(set(X[column]),777, column in self.categorical_features)
            
            try:
                #print(X[column], self.categorical_features, 999)
                if column in self.categorical_features:
                    if X[column].isnull().values.any():
                        X[column].fillna(value=self.categoricalna,inplace=True)                    
                    continue
                
                #print(X[pd.to_numeric(X[column], errors = 'coerce').isnull()])
                                
                X[column] = pd.to_numeric(X[column], errors = 'coerce')
                
                if(X[column].isnull().values.all()):
                    X[column].fillna(value=0,inplace=True)
                    imputation_dir[column] = 0 #X_train[column].min()/6   # John
                elif(X[column].notnull().values.any() and X[column].min() != 0 ):
                    X[column].fillna(value=X[column].min()/6,inplace=True)
                    imputation_dir[column] = X[column].min()/6    # John
                else:
                    X[column].fillna(value=0,inplace=True)    
                    imputation_dir[column] = 0 #X_train[column].min()/6   # John
                n_numerical +=1
            except:
                print(column in X)
                raise Exception('column <%s> is not a numerical feature. Defining it as a categorical feature in categorical_set' % column)
        
        self.imputation_dir = imputation_dir
        
        return X
        
    def get_params(self, deep = True):
        params = {}
        params['categorical_features'] = self.categorical_features
        params['categoricalna'] = self.categoricalna
        return params
    
    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)   
            
#from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#from sklearn.base import BaseEstimator, TransformerMixin, clone
#import numpy as np

class DataEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, missing_value = 'UNQ', unseen_missed = 'unseen_missed'):  # missing_value for na
        self.categorical_features = categorical_features
        self.unseen_missed = unseen_missed  # for categorical features
        self.missing_value = missing_value  # for categorical features
        return
    
    def fit(self, X, y = None):
        
        if self.categorical_features is None or len(self.categorical_features) == 0:
            return self

        # categorical_names = {}
        labelers = {}
        n_values=[] #np.array([0]*len(categorical_features))
        for feature in self.categorical_features:
#            print(feature, 999)
            if feature not in X:
                continue
            le = LabelEncoder()
            # categorical_feature = X[:, feature]
            m_categorical_feature = X[feature]

            #m_categorical_feature = list(m_categorical_feature) + [self.unseen_missed]
            #m_categorical_feature = np.array(m_categorical_feature, dtype = object)
            m_categorical_feature = np.append(m_categorical_feature, [self.unseen_missed]) # for unseen missed values

            #le.fit(X[:, feature])
            le.fit(m_categorical_feature)
            labelers[feature] = le
            # X[:, feature] = le.transform(X[:, feature])

            X[feature] = le.transform(X[feature].apply(str))  # John, Nov 14, 2018
            # categorical_names[feature] = le.classes_
            n_values.append(len(le.classes_)) # = len(le.classes_)+1
        
        # onehotencoder
        
        n_values = np.array(n_values)    
        # X = X.astype(dtype=np.float64)  
        
        encoder = OneHotEncoder(categorical_features=[X.columns.get_loc(feature) for feature in self.categorical_features], n_values = n_values, sparse = False, dtype = np.float64)
        #np.random.seed(1)
        
        encoder.fit(X)
        
        # save results
        
        self.encoder = encoder
        self.labelers = labelers
        
        # X = encoder.transform(X)
        
        return self
    
    def transform(self, X, y = None):
        for feature in self.categorical_features:
            m_missing_value = self.missing_value if self.missing_value in self.labelers[feature].classes_ else self.unseen_missed
            # categorical_feature = np.array([f if f in labelers[feature].classes_ else labelers[feature].classes_[labelers[feature].classes_.tolist().index(missing_value)] for f in X[feature].values])
            X[feature] = X[feature].apply(str) # John, Nov 14, 2018
            m_categorical_feature = np.array([f if f in self.labelers[feature].classes_ else m_missing_value for f in X[feature].values])
            
            X[feature] = self.labelers[feature].transform(m_categorical_feature)        
        return self.encoder.transform(X)

    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)
    
    def get_params(self, deep = True):
        params = {}
        params['categorical_features'] = self.categorical_features
        params['unseen_missed'] = self.unseen_missed
        params['missing_value'] = self.missing_value
        return params
    
    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

class SmoteSampler(BaseEstimator, TransformerMixin):
    # Smote sampler, John Li, Nov 11. 2018. BigRio
    def __init__(self):
        return
    
#    def fit(self, X, y):
#        smt = SMOTETomek(ratio='auto')
#        X_smt, y_smt = smt.fit_sample(X, y)
#        return X_smt, y_smt 
    
    def fit_transform(self, X, y):
        smt = SMOTETomek(ratio='auto')
        X_smt, y_smt = smt.fit_sample(X, y)
        return X_smt, y_smt 
#        return self.fit(X, y)
    
    def transform(self, X, y= None):
        return X #self.fit(X, y)
    
    def get_params(self, deep = True):
        params = {}
#        params['categorical_features'] = self.categorical_features
#        params['unseen_missed'] = self.unseen_missed
#        params['missing_value'] = self.missing_value
        return params
    
    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)        

def GetClassifier(clf_name = 'lgr', categorical_features = None, feature_range = [600, 700], refit = True, n_jobs = 1, verbose = 0, opt = False):
    # June 12, 2018, John
    
    # if not hasattr(None, 'categorical_features'): # June 28, 2018, John
    #     categorical_features = None
        
    selector_rig = Pipeline([("data_imputation", DataImputation(categorical_features)),
                            ("encoder", DataEncoder(categorical_features)),
                            ("selector", SelectKBest(f_classif)),
                            ('rig', RidgeClassifier(class_weight = 'balanced', random_state = 10) )])   
    
    selector_lgr = Pipeline([("data_imputation", DataImputation(categorical_features)),
                            ("encoder", DataEncoder(categorical_features)),
                            ("selector", SelectKBest(f_classif)),
                            ('lgr', LogisticRegression(class_weight = 'balanced', random_state = 10) )])
                            
    selector_lasso = Pipeline([("data_imputation", DataImputation(categorical_features)),
                        ("encoder", DataEncoder(categorical_features)),
                        ("selector", SelectKBest(f_classif)),
                        ('lasso', Lasso(random_state =10) )])     
    selector_svc = Pipeline([("data_imputation", DataImputation(categorical_features)),
                        ("encoder", DataEncoder(categorical_features)),
                        ("selector", SelectKBest(f_classif)),
                        ('svc', SVC(class_weight = 'balanced', random_state = 10, probability= True) )])

    selector_n_rig = Pipeline([("data_imputation", DataImputation(categorical_features)),
                            ("encoder", DataEncoder(categorical_features)),
                            ("selector", SelectKBest(f_classif)),
                            ("normalizer", Normalizer()), 
                            ('rig', RidgeClassifier(class_weight = 'balanced', random_state = 10) )])   
    
    selector_n_lgr = Pipeline([("data_imputation", DataImputation(categorical_features)),
                            ("encoder", DataEncoder(categorical_features)),
                            ("selector", SelectKBest(f_classif)),
                            ("normalizer", Normalizer()), 
                            ('lgr', LogisticRegression(class_weight = 'balanced', random_state = 10) )])

    
    clfs = {}
    if opt:
        clfs['rig'] = GridSearchCV(selector_rig, cv=10, n_jobs=n_jobs,  verbose = verbose, param_grid = {'selector__k':list(range(feature_range[0], feature_range[1]+1, 10)), 'rig__alpha':np.logspace(-2, 4, 15)}, scoring = "roc_auc")
        clfs['lgr'] = GridSearchCV(selector_lgr, cv=10, n_jobs=n_jobs,  verbose = verbose, param_grid = {'selector__k':list(range(feature_range[0], feature_range[1]+1, 20)), 'lgr__C':np.logspace(-5,2,15)}, scoring = "roc_auc")
        clfs['svc'] = GridSearchCV(selector_svc, cv=10, n_jobs=n_jobs,  verbose = verbose, param_grid = {'selector__k':list(range(feature_range[0], feature_range[1]+1, 20)), 'svc__C':np.logspace(-5,2,15)}, scoring = "roc_auc")
        clfs['lasso'] = GridSearchCV(selector_lasso, cv=10, n_jobs=n_jobs,  verbose = verbose, param_grid = {'selector__k':list(range(feature_range[0], feature_range[1]+1, 20)), 'lasso__alpha':np.logspace(-5,3,15)}, scoring = "roc_auc")        
        clfs['nrig'] = GridSearchCV(selector_n_rig, cv=10, n_jobs=n_jobs,  verbose = verbose, param_grid = {'selector__k':list(range(feature_range[0], feature_range[1]+1, 10)), 'normalizer__norm': ('l1', 'l2', 'max'), 'normalizer__axis':(0, 1), 'rig__alpha':np.logspace(-2, 5, 15)}, scoring = "roc_auc")
        clfs['nlgr'] = GridSearchCV(selector_n_lgr, cv=10, n_jobs=n_jobs,  verbose = verbose, param_grid = {'selector__k':list(range(feature_range[0], feature_range[1]+1, 20)), 'normalizer__norm': ('l1', 'l2', 'max'), 'normalizer__axis':(0, 1), 'lgr__C':np.logspace(-5,2,15)}, scoring = "roc_auc")
    else:
        clfs= {}                            
        clfs['rig'] = GridSearchCV(selector_rig, cv=10, n_jobs=n_jobs,  verbose = verbose, param_grid = {'selector__k':[680], 'rig__alpha':[3162.2776601683795]}, scoring = "roc_auc")
        clfs['lgr'] = GridSearchCV(selector_lgr, cv=10, n_jobs=n_jobs,  verbose = verbose, param_grid = {'selector__k':[680], 'lgr__C':[0.002]}, scoring = "roc_auc")
        clfs['svc'] = GridSearchCV(selector_svc, cv=10, n_jobs=n_jobs,  verbose = verbose, param_grid = {'selector__k':[680], 'svc__C':[0.002]}, scoring = "roc_auc")
        clfs['lasso'] = GridSearchCV(selector_lasso, cv=10, n_jobs=n_jobs,  verbose = verbose, param_grid = {'selector__k':[680], 'lasso__alpha':[1e-05]}, scoring = "roc_auc")

    #assert(clf_name in clfs)
    
    if clf_name not in clfs:
        raise Exception('alg <%s> is not defined. alg: lgr,  rig, svc, lasso' % clf_name)

    clf = clfs[clf_name]
    clf.refit = refit

    return clf

def RenameFeatures(X, feature_names, encoder, categorical_features = None, labelers = None):
#    feature_names = [dataset.feature_names[j] for j in dataset.selected_features] 
    #print(feature_names,encoder.feature_indices_,111)
    new_feature_names = []
    
    k = 0   # original features
    j = 0   # no. of original categorical features
    #print(encoder.feature_indices_, len(encoder.feature_indices_))
    for i in range(np.shape(X)[1]):

        if k == categorical_features[j]:
            #print(k, j, len(labelers[categorical_features[j]].classes_))
            if i-encoder.feature_indices_[j] < len(labelers[categorical_features[j]].classes_):
                # not clear why, May 8, 2018           
                cvalue = labelers[categorical_features[j]].classes_[i-encoder.feature_indices_[j]] # if labelers is not None else str(i-encoder.feature_indices_[j])
                new_feature_names.append(feature_names[k]+'_['+cvalue+']') #str(i-encoder.feature_indices_[j]))
                if i == encoder.feature_indices_[j+1]-1:
                    #print(k, j, len(labelers[categorical_features[j]].classes_))
                    k+=1
                    j = min(j+1, len(categorical_features)-1)
            else:
                k+=1
                j = min(j+1, len(categorical_features)-1)
        else:
            new_feature_names.append(feature_names[k]) #+'_'+str(dataset.selected_features[k]))
            k+=1
            
            # X shape  might be larger than what should be; not clear
            
            if k >= len(feature_names):
                break
          
    return new_feature_names

def FeatureRanking(scores, feature_names = None, topk = None):
    # not really helpful
    importances = scores #clf.feature_importances_
    importances = importances[np.logical_not(np.isnan(importances))] # remove some features without values; importence is nan
#    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
#                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    #print("Feature ranking:")
    
    if topk is None:
        topk = len(indices)
    
    if feature_names is not None:
        return [','.join(map(str, (f + 1, indices[f], feature_names[indices[f]], importances[indices[f]]))) for f in range(len(indices)) if f < topk and ('unseen_missed' not in feature_names[indices[f]] if (indices[f] < len(importances) and indices[f] < len(feature_names)) else False)] # unseen missed become importance features ! ignored for clarification
    
    return [(f + 1, indices[f], importances[indices[f]]) for f in range(len(indices)) if f < topk and indices[f] < len(importances)]
    
if __name__ == '__main__':
    
    print('test')
    