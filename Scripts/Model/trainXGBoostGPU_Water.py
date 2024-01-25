import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import sklearn
from sklearn.model_selection import GridSearchCV
import gc
from skopt import BayesSearchCV
from numba import jit
from pickle import dump
#### Import Training Data ####

floodedForestTrainingData = 'Flooded_Forest_Full_TrainingData.hdf'
backgroundTrainingData = 'Background_Full_TrainingData.hdf'
openWaterTrainingData = 'Open_Water_Full_TrainingData.hdf'

model = 'xgb'
ffDF = pd.read_hdf(floodedForestTrainingData)
bkDF = pd.read_hdf(backgroundTrainingData)
owDF = pd.read_hdf(openWaterTrainingData)

print(ffDF.shape)
print(owDF.shape)
print(bkDF.shape)

crash()
bandNames = [
  'HH',
  'HV',
  'NDPI',
  'Inc',
  'Slope',
  'Hand',
]

statsList = ['Mean', 'stdDev']

listVars = []
for bn in bandNames:
  for stat in statsList:
      listVars.append('{0}{1}'.format(bn, stat))
        


listVars = ['HHMean','HVMean','NDPIMean','NDPIstdDev','IncstdDev','SlopeMean','SlopestdDev','HandstdDev']

print(listVars)

#### Add Target Columns ####
ffDF['Class'] = 0
owDF['Class'] = 1
bkDF['Class'] = 0


#### Merge Datasets into Single DF ####
totalData = pd.concat([ffDF,owDF,bkDF])

del ffDF
del owDF
del bkDF
gc.collect()

targetData = totalData['Class']
totalData = totalData.drop(labels=['Class'],axis=1)

totalData = totalData[listVars]

# totalData = totalData.drop(labels=listVars,axis=1)
x_train, x_test, y_train, y_test = train_test_split(totalData, targetData, test_size=0.25)
print("Split Made")
del totalData
del targetData
gc.collect()
print(y_test.shape)

#### Perform Data Satndardisation ####

scaler = preprocessing.StandardScaler().fit(x_train)

x_scaled = scaler.transform(x_train)
dump(scaler,open('./Scaler_Water.pkl','wb'))
x_test_Scaled = scaler.transform(x_test).astype(np.float32)

del x_train
del x_test
gc.collect()

##### Perform Grid Search ####
seed = np.random.randint(1, 10000)

params = {
  'min_child_weight': [1, 2,3,4,5,6,7,8,9,10],
  'gamma': [0.5, 1, 1.5, 2,3,4,5],
  'subsample': [0.5,0.55,0.6,0.65,0.7,0.75, 0.8],
  'colsample_bytree': [0.2,0.4,0.6, 0.8, 1.0],
  'max_depth': [1,2,3, 4, 5],
  'learning_rate': [0.2, 0.4, 0.6, 0.8, 1],
  'booster': ['gbtree']
}

   # 'scale_pos_weight': [classWeight]
# 
#  gpu_id=0,
xgb_model = xgb.XGBClassifier(random_state=seed,
                              tree_method='gpu_hist', 
                              gpu_id=0, 
                              objective='binary:hinge',
                              eval_metric='logloss',
                              verbosity=0)

opt = BayesSearchCV(xgb_model,params)



opt.fit(x_scaled , y_train.astype(np.float32))

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(x_test_Scaled, y_test))

resultsDF = pd.DataFrame(opt.cv_results_)

resultsDF.to_csv('Model_02_HyperParameters.csv')

opt.best_estimator_.save_model('Trained_XGBoostModel_02_Amazon_Water.model')

feature_important = opt.best_estimator_.get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

data.to_csv('Feature_Importances')


