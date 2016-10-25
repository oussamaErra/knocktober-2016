import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
import xgboost as xgb
from datetime import datetime

def shuffle(df, n=1, axis=0):     
         df = df.copy()
         for _ in range(n):
             df.apply(np.random.shuffle, axis=axis)
         return df

def cam_type(x):
     if x <10:
         return 'short_camp'
     if (x<= 100) & (x >=10):
         return 'middle_camp'
     if (x<300) & (x>100) :
         return 'long_camp'
     if x>300 :
         return 'very_long_camp'
def map_class(a,b):
     if a >b :
         return 1
     else:
          return 0

train=pd.read_csv('C:/Users/oussama/Documents/Knocktober/data/train_first.csv',header=0)
test=pd.read_csv('C:/Users/oussama/Documents/Knocktober/test_first.csv',header=0)
test_predict=test.loc[:,['Patient_ID','Health_Camp_ID']]
submission= pd.read_csv('C:/Users/oussama/Documents/Knocktober/submissions/submission.csv',header=0)







train.drop(['City_Type','Employer_Category'],axis=1,inplace=True)
test.drop(['City_Type','Employer_Category'],axis=1,inplace=True)


train.drop(train[train.Registration_Date.isnull()].index,inplace=True )
train=train.reset_index(drop=True)

ntrain=train.shape[0]
ntest=test.shape[0]
target=train.target
train.drop('target',axis=1,inplace=True)


train_test=pd.concat([train,test]).reset_index(drop=True)



train_test['day_of_week_start']=train_test.loc[:,'Camp_Start_Date'].apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y').weekday())
train_test['day_of_week_registration']=train_test.loc[:,'Registration_Date'].apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y').weekday())



train_test['month_of_start']=train_test.loc[:,'Camp_Start_Date'].apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y').strftime('%b'))
train_test['month_of_registration']=train_test.loc[:,'Registration_Date'].apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y').strftime('%b'))






train_test.Registration_Date = train_test.Registration_Date.apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y'))
train_test.Camp_End_Date= train_test.Camp_End_Date.apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y'))
train_test.Camp_Start_Date= train_test.Camp_Start_Date.apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y'))

train_test['time_to_start']=((train_test.Camp_Start_Date - train_test.Registration_Date)/np.timedelta64(1,'D')).astype(int)
train_test['camp_duration']=((train_test.Camp_End_Date - train_test.Camp_Start_Date)/np.timedelta64(1,'D')).astype(int)
train_test['registred_after']=train_test.time_to_start.apply(lambda x : 1 if x >0 else 0)
train_test['registred_intime']=train_test.time_to_start.apply(lambda x : 1 if x ==0 else 0)
train_test['registred_befor']=train_test.time_to_start.apply(lambda x : 1 if x <0 else 0)
train_test['camp_type']=pd.Series(train_test.camp_duration.apply(dict(train_test.camp_duration.value_counts()).get))
train_test.camp_type=train_test.camp_type.apply(lambda x : cam_type(x))

drops=['Patient_ID','Health_Camp_ID','First_Interaction','Camp_Start_Date','Camp_End_Date','Registration_Date','Education_Score','Income','Age']
category=['month_of_start','month_of_registration','Category2','camp_type']
#category=['month_of_start','month_of_registration']
train_test.drop(drops,axis=1,inplace=True)
#one_hot=['month_of_registration','month_of_start','day_of_week_registration','day_of_week_start','Category3','Category2','Var4','camp_type']
#one_hot=['day_of_week_start','camp_type','Category3','Category2']
#dic_day = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
#train_test.day_of_week_registration=train_test.day_of_week_registration.apply(dic_day.get)
#train_test.day_of_week_start=train_test.day_of_week_start.apply(dic_day.get)
#train_test.Var4=train_test.Var4.astype(str)

#train_test=pd.concat([train_test,pd.get_dummies(train_test.loc[:,one_hot])],axis=1)
#train_test.drop(one_hot,axis=1,inplace=True)

for k in category:
     train_test.loc[:,k]=pd.factorize(train_test.loc[:,k])[0]
x_train =train_test.iloc[:ntrain,:]
x_test =train_test.iloc[ntrain:,:]
X=x_train.values
y_train=target.values

#model 1

predictions=np.zeros((x_test.shape[0],2))
splits = StratifiedKFold(y_train,n_folds=2,shuffle=True,random_state=0)
for i ,(train_index,test_index) in enumerate(splits):
    X_train,Y_train = X[train_index] , y_train[train_index]
   
    dtrain=xgb.DMatrix(X_train,Y_train)
    xgb_params={ 'seed': 0,'eval_metric':'auc','booster':'gbtree','objective': 'binary:logistic','max_depth ':6 , 'min_child_weight' : 3 , 'subsample': 0.8 , 'colsample_bytree':0.8 ,'learning_rate':0.01}
    clf=xgb.train(params=xgb_params,dtrain=dtrain,num_boost_round=173)
    predictions[:,i]=clf.predict(xgb.DMatrix(x_test))

test_predict['proba1']=np.mean(predictions,axis=1)
#test_predict.proba=test_predict.proba.apply(lambda x : map_class(x_test,0.7))




# model 2 
X=x_train.values
y_train=target.values

dtrain=xgb.DMatrix(X,y_train)
xgb_params={ 'seed': 0, 'eval_metric':'auc','booster':'gbtree','objective': 'binary:logistic','max_depth ':5 , 'min_child_weight' : 3 , 'subsample': 0.8 , 'colsample_bytree':0.8 ,'learning_rate':0.01}
clf=xgb.train(params=xgb_params,dtrain=dtrain,num_boost_round=173)
test_predict['proba2']=clf.predict(xgb.DMatrix(x_test))

# model 3

#x_train['target']=target

#train_1=x_train[x_train.target==1].reset_index(drop=True)
#train_0=x_train[x_train.target==0].reset_index(drop=True)


#predictions=np.zeros((x_test.shape[0],3))
#splits = KFold(train_0.shape[0],n_folds=3,shuffle=True,random_state=0)
#for i ,(_,test) in enumerate(splits):
#    X_0= train_0.ix[test,:]
#    X=pd.concat([X_0,train_1]).reset_index(drop=True)
#    X=shuffle(X)
#    y=X.target.values
#    X.drop('target',axis=1,inplace=True)
#    dtrain=xgb.DMatrix(X,y)
#    xgb_params={  'seed': 0,'eval_metric':'auc','booster':'gbtree','objective': 'binary:logistic','max_depth ':7 , 'min_child_weight' : 1 , 'subsample': 0.8 , 'colsample_bytree':0.8 ,'learning_rate':0.01}
#    clf=xgb.train(params=xgb_params,dtrain=dtrain,num_boost_round=973)
#    predictions[:,i]=clf.predict(xgb.DMatrix(x_test))

#test_predict['proba3']=np.mean(predictions,axis=1)

#test_predict['proba'] =(test_predict['proba1']+ test_predict['proba2']+test_predict['proba3']  )/3
#drops=['proba1','proba2','proba3']

drops=['proba1','proba2']
test_predict['proba'] =(test_predict['proba1']+ test_predict['proba2']  )/2

test_predict.drop(drops,axis=1,inplace=True)



submission=pd.merge(submission,test_predict,on=['Patient_ID','Health_Camp_ID'],how='left')








#2eme camp

train=pd.read_csv('C:/Users/oussama/Documents/Knocktober/data/train_second.csv',header=0)
test=pd.read_csv('C:/Users/oussama/Documents/Knocktober/test_second.csv',header=0)
test_predict=test.loc[:,['Patient_ID','Health_Camp_ID']]





train.drop(['City_Type','Employer_Category'],axis=1,inplace=True)
test.drop(['City_Type','Employer_Category'],axis=1,inplace=True)

train.drop(train[train.Registration_Date.isnull()].index,inplace=True )
train=train.reset_index(drop=True)

ntrain=train.shape[0]
ntest=test.shape[0]
target=train.target
train.drop('target',axis=1,inplace=True)

#people  have attended the first camp


train_cam1=pd.read_csv('C:/Users/oussama/Documents/Knocktober/data/train_first.csv',header=0)
test_cam1=pd.read_csv('C:/Users/oussama/Documents/Knocktober/test_first.csv',header=0)
train_cam2=pd.read_csv('C:/Users/oussama/Documents/Knocktober/data/train_second.csv',header=0)
test_cam2=pd.read_csv('C:/Users/oussama/Documents/Knocktober/test_second.csv',header=0)

peoples_train = [ k for k in train_cam2.Patient_ID.values if k in train_cam1.Patient_ID.values ]
peoples_test  = [k for k in test_cam2.Patient_ID.values if k in test_cam1.Patient_ID.values]

train['has_attended']=train.Patient_ID
train['has_attended']=train.has_attended.apply(lambda x: True if x in peoples_train else False )

test['has_attended']=test.Patient_ID
test['has_attended']= test.has_attended.apply(lambda x: True if x in peoples_test else False )





train_test=pd.concat([train,test]).reset_index(drop=True)





# new features

train_test['day_of_week_start']=train_test.loc[:,'Camp_Start_Date'].apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y').weekday())
train_test['day_of_week_registration']=train_test.loc[:,'Registration_Date'].apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y').weekday())



train_test['month_of_start']=train_test.loc[:,'Camp_Start_Date'].apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y').strftime('%b'))
train_test['month_of_registration']=train_test.loc[:,'Registration_Date'].apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y').strftime('%b'))






train_test.Registration_Date = train_test.Registration_Date.apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y'))
train_test.Camp_End_Date= train_test.Camp_End_Date.apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y'))
train_test.Camp_Start_Date= train_test.Camp_Start_Date.apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y'))

train_test['time_to_start']=((train_test.Camp_Start_Date - train_test.Registration_Date)/np.timedelta64(1,'D')).astype(int)
train_test['camp_duration']=((train_test.Camp_End_Date - train_test.Camp_Start_Date)/np.timedelta64(1,'D')).astype(int)
train_test['registred_after']=train_test.time_to_start.apply(lambda x : 1 if x >0 else 0)
train_test['registred_intime']=train_test.time_to_start.apply(lambda x : 1 if x ==0 else 0)
train_test['registred_befor']=train_test.time_to_start.apply(lambda x : 1 if x <0 else 0)
train_test['camp_type']=pd.Series(train_test.camp_duration.apply(dict(train_test.camp_duration.value_counts()).get))
train_test.camp_type=train_test.camp_type.apply(lambda x : cam_type(x))

drops=['Patient_ID','Health_Camp_ID','First_Interaction','Camp_Start_Date','Camp_End_Date','Registration_Date','Education_Score','Income','Age']
category=['month_of_start','month_of_registration','Category2','camp_type']
#category=['month_of_start','month_of_registration']
train_test.drop(drops,axis=1,inplace=True)
#one_hot=['month_of_registration','month_of_start','day_of_week_registration','day_of_week_start','Category3','Category2','Var4','camp_type']
#one_hot=['day_of_week_start','camp_type','Category3','Category2']
#dic_day = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
#train_test.day_of_week_registration=train_test.day_of_week_registration.apply(dic_day.get)
#train_test.day_of_week_start=train_test.day_of_week_start.apply(dic_day.get)
#train_test.Var4=train_test.Var4.astype(str)

#train_test=pd.concat([train_test,pd.get_dummies(train_test.loc[:,one_hot])],axis=1)
#train_test.drop(one_hot,axis=1,inplace=True)

for k in category:
     train_test.loc[:,k]=pd.factorize(train_test.loc[:,k])[0]
x_train =train_test.iloc[:ntrain,:]
x_test =train_test.iloc[ntrain:,:]

x_test=x_test.values
X=x_train.values
y_train=target.values

#model 1

predictions=np.zeros((x_test.shape[0],2))
splits = StratifiedKFold(y_train,n_folds=2,shuffle=True,random_state=0)
for i ,(train_index,test_index) in enumerate(splits):
    X_train,Y_train = X[train_index] , y_train[train_index]
   
    dtrain=xgb.DMatrix(X_train,Y_train)
    xgb_params={ 'seed': 0, 'eval_metric':'auc','booster':'gbtree','objective': 'binary:logistic','max_depth ':3, 'min_child_weight' : 2, 'subsample': 0.6 , 'colsample_bytree':0.5 ,'learning_rate':0.01}
    clf=xgb.train(params=xgb_params,dtrain=dtrain,num_boost_round=164)
    predictions[:,i]=clf.predict(xgb.DMatrix(x_test))

test_predict['proba1']=np.mean(predictions,axis=1)
#test_predict.proba=test_predict.proba.apply(lambda x : map_class(x_test,0.7))




# model 2 
X=x_train.values
y_train=target.values

dtrain=xgb.DMatrix(X,y_train)
xgb_params={ 'seed': 0, 'eval_metric':'auc','booster':'gbtree','objective': 'binary:logistic','max_depth ':3, 'min_child_weight' : 2, 'subsample': 0.6 , 'colsample_bytree':0.5 ,'learning_rate':0.01}
clf=xgb.train(params=xgb_params,dtrain=dtrain,num_boost_round=164)
test_predict['proba2']=clf.predict(xgb.DMatrix(x_test))



test_predict['proba'] =(test_predict['proba1']+ test_predict['proba2'] )/2
drops=['proba1','proba2']
test_predict.drop(drops,axis=1,inplace=True)


submission=pd.merge(submission,test_predict,on=['Patient_ID','Health_Camp_ID'],how='left')





#3eme camp


train=pd.read_csv('C:/Users/oussama/Documents/Knocktober/data/train_third.csv',header=0)
test=pd.read_csv('C:/Users/oussama/Documents/Knocktober/test_third.csv',header=0)
test_predict=test.loc[:,['Patient_ID','Health_Camp_ID']]





train.drop(['City_Type','Employer_Category'],axis=1,inplace=True)
test.drop(['City_Type','Employer_Category'],axis=1,inplace=True)

train.drop(train[train.Registration_Date.isnull()].index,inplace=True )
train=train.reset_index(drop=True)

ntrain=train.shape[0]
ntest=test.shape[0]
target=train.target
train.drop('target',axis=1,inplace=True)


train_test=pd.concat([train,test]).reset_index(drop=True)





#new features

train_test['day_of_week_start']=train_test.loc[:,'Camp_Start_Date'].apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y').weekday())
train_test['day_of_week_registration']=train_test.loc[:,'Registration_Date'].apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y').weekday())



train_test['month_of_start']=train_test.loc[:,'Camp_Start_Date'].apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y').strftime('%b'))
train_test['month_of_registration']=train_test.loc[:,'Registration_Date'].apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y').strftime('%b'))






train_test.Registration_Date = train_test.Registration_Date.apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y'))
train_test.Camp_End_Date= train_test.Camp_End_Date.apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y'))
train_test.Camp_Start_Date= train_test.Camp_Start_Date.apply(lambda x : datetime.strptime(str(x) ,'%d-%b-%y'))

train_test['time_to_start']=((train_test.Camp_Start_Date - train_test.Registration_Date)/np.timedelta64(1,'D')).astype(int)
train_test['camp_duration']=((train_test.Camp_End_Date - train_test.Camp_Start_Date)/np.timedelta64(1,'D')).astype(int)
train_test['registred_after']=train_test.time_to_start.apply(lambda x : 1 if x >0 else 0)
train_test['registred_intime']=train_test.time_to_start.apply(lambda x : 1 if x ==0 else 0)
train_test['registred_befor']=train_test.time_to_start.apply(lambda x : 1 if x <0 else 0)
train_test['camp_type']=pd.Series(train_test.camp_duration.apply(dict(train_test.camp_duration.value_counts()).get))
train_test.camp_type=train_test.camp_type.apply(lambda x : cam_type(x))

drops=['Patient_ID','Health_Camp_ID','First_Interaction','Camp_Start_Date','Camp_End_Date','Registration_Date','Education_Score','Income','Age']
category=['month_of_start','month_of_registration','Category2','camp_type']
#category=['month_of_start','month_of_registration']
train_test.drop(drops,axis=1,inplace=True)
#one_hot=['month_of_registration','month_of_start','day_of_week_registration','day_of_week_start','Category3','Category2','Var4','camp_type']
#one_hot=['day_of_week_start','camp_type','Category3','Category2']
#dic_day = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
#train_test.day_of_week_registration=train_test.day_of_week_registration.apply(dic_day.get)
#train_test.day_of_week_start=train_test.day_of_week_start.apply(dic_day.get)
#train_test.Var4=train_test.Var4.astype(str)

#train_test=pd.concat([train_test,pd.get_dummies(train_test.loc[:,one_hot])],axis=1)
#train_test.drop(one_hot,axis=1,inplace=True)

for k in category:
     train_test.loc[:,k]=pd.factorize(train_test.loc[:,k])[0]
x_train =train_test.iloc[:ntrain,:]
x_test =train_test.iloc[ntrain:,:]





x_test=x_test.values
X=x_train.values
y_train=target.values

#model 1

predictions=np.zeros((x_test.shape[0],2))
splits = StratifiedKFold(y_train,n_folds=2,shuffle=True,random_state=0)
for i ,(train_index,test_index) in enumerate(splits):
    X_train,Y_train = X[train_index] , y_train[train_index]
   
    dtrain=xgb.DMatrix(X_train,Y_train)
    xgb_params={  'seed': 0,'eval_metric':'auc','booster':'gbtree','objective': 'binary:logistic','max_depth ':3, 'min_child_weight' : 2, 'subsample': 0.6 , 'colsample_bytree':0.5 ,'learning_rate':0.01}
    clf=xgb.train(params=xgb_params,dtrain=dtrain,num_boost_round=150)
    predictions[:,i]=clf.predict(xgb.DMatrix(x_test))

test_predict['proba1']=np.mean(predictions,axis=1)
#test_predict.proba=test_predict.proba.apply(lambda x : map_class(x_test,0.7))




# model 2 
X=x_train.values
y_train=target.values

dtrain=xgb.DMatrix(X,y_train)
xgb_params={ 'seed': 0, 'eval_metric':'auc','booster':'gbtree','objective': 'binary:logistic','max_depth ':3, 'min_child_weight' : 2, 'subsample': 0.6 , 'colsample_bytree':0.5 ,'learning_rate':0.01}
clf=xgb.train(params=xgb_params,dtrain=dtrain,num_boost_round=150)
test_predict['proba2']=clf.predict(xgb.DMatrix(x_test))



test_predict['proba'] =(test_predict['proba1']+ test_predict['proba2'] )/2
drops=['proba1','proba2']
test_predict.drop(drops,axis=1,inplace=True)

submission=pd.merge(submission,test_predict,on=['Patient_ID','Health_Camp_ID'],how='left')
submission.fillna(0,inplace=True)
submission['Outcome']=submission.proba_x+submission.proba_y+submission.proba
submission.drop(['proba_x','proba_y','proba'],axis=1,inplace=True)
submission.to_csv('C:/Users/oussama/Documents/Knocktober/submissions/last_day_syb/The_Final_Submission.csv',index=None)


