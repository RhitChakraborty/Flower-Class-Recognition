"""
Preprocessing
"""

import pandas as pd

train=pd.read_csv('../inputs/Train.csv')
test=pd.read_csv('../inputs/Test.csv')

#full Dataframe
df_full=pd.concat([train,test],axis=0).reset_index(drop=True)

# =============================================================================
# ##plotting a pairplot
# import seaborn as sns
# sns.pairplot(df_full,hue='Class')
# 
# =============================================================================

df_full['d_groups']=pd.cut(df_full['Diameter'],bins=[-1,5,10,13,24,31,70,112,387])
df_full['h_groups']=pd.cut(df_full['Height'],bins=[-1,2,3.1,5.5,7.6,8.9,12.5,25,60.1])


temp1=df_full.groupby('Locality_Code').agg({'Diameter':['max','mean','std','count']})
temp1.columns=['_locality_'.join(x) for x in temp1.columns]
df_full=pd.merge(df_full,temp1,on='Locality_Code',how='left')

temp2=df_full.groupby('Region_Code').agg({'Diameter':['min','max','mean','std','count']})
temp2.columns=['_region_'.join(x) for x in temp2.columns]
df_full=pd.merge(df_full,temp2,on='Region_Code',how='left')

df_full.loc[:,['Diameter_locality_std','Diameter_region_std']]=df_full.loc[:,['Diameter_locality_std','Diameter_region_std']].fillna(0)


df_test=df_full[df_full['Class'].isnull()]
df_train=df_full[df_full['Class'].notnull()]

df_train.to_csv('../inputs/train_pr.csv',index=False)
df_test.to_csv('../inputs/test_pr.csv',index=False)
