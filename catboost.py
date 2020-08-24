# -*- coding: utf-8 -*-
"""
Catboost
"""
import pandas as pd
import numpy as np
from sklearn import metrics,preprocessing
import dispatcher
from catboost import Pool

df=pd.read_csv('../inputs/train_final.csv')

features=[f for f in df.columns if f not in ['kfold','Class']]
num_feat=['Height', 'Diameter','Diameter_locality_max',
       'Diameter_locality_mean', 'Diameter_locality_std',
       'Diameter_locality_count', 'Diameter_region_min', 'Diameter_region_max',
       'Diameter_region_mean', 'Diameter_region_std', 'Diameter_region_count',]
cat_feat=['Area_Code', 'Locality_Code', 'Region_Code','Species', 'd_groups', 
          'h_groups']


##feature scaling
sc=preprocessing.MinMaxScaler()
sc.fit(df.loc[:,num_feat])
df.loc[:,num_feat]=sc.transform(df.loc[:,num_feat])

for f in cat_feat:
    df[f]=df[f].astype(str)

model=dispatcher.models['catboost']

#for evaluation    
fold=1    
df_train=df[df.kfold!=fold].reset_index(drop=True)
df_valid=df[df.kfold==fold].reset_index(drop=True)

x_train=df_train[features]
y_train=df_train.Class
x_valid=df_valid[features]
y_valid=df_valid.Class.values


    
train_pool=Pool(data=x_train,label=y_train,cat_features=cat_feat,feature_names=x_train.columns.tolist())
test_pool=Pool(data=x_valid,label=y_valid,cat_features=cat_feat,feature_names=x_valid.columns.tolist())

model.fit(train_pool,eval_set=test_pool)

# =============================================================================
# 
# #for final prediction
# X=df[features]
# y=df.Class
# train_pool=Pool(data=X,label=y,cat_features=cat_feat,feature_names=X.columns.tolist())
# 
# 
# 
# model.fit(train_pool)
# 
# test=pd.read_csv('../inputs/test_pr.csv').drop('Class',axis=1)
# test.loc[:,num_feat]=sc.transform(test.loc[:,num_feat])
# 
# for f in cat_feat:
#     test[f]=test[f].astype(str)
#     
# y_pred=model.predict_proba(test)
# y_pred=pd.DataFrame(y_pred)
# 
# y_pred.to_csv("../inputs/sub_cat1.csv",index=False)
# =============================================================================
