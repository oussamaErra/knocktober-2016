import pandas as pd


#import the files and create 6 separated data sets , 3 for training 3 models and 3 test data sets then merge the 3 predicions in one submission file and make the submission



train=pd.read_csv('C:/Users/oussama/Documents/Knocktober/train.csv',header=0)
test=pd.read_csv('C:/Users/oussama/Documents/Knocktober/test.csv',header=0)

first=pd.read_csv('C:/Users/oussama/Documents/Knocktober/First_Health_Camp_Attended.csv',header=0)
second=pd.read_csv('C:/Users/oussama/Documents/Knocktober/Second_Health_Camp_Attended.csv',header=0)
third=pd.read_csv('C:/Users/oussama/Documents/Knocktober/Third_Health_Camp_Attended.csv',header=0)
patient_profile=pd.read_csv('C:/Users/oussama/Documents/Knocktober/Patient_Profile.csv',header=0)
healt_camps=pd.read_csv('C:/Users/oussama/Documents/Knocktober/Health_Camp_Detail.csv',header=0)

first_final=pd.merge(first,healt_camps,on ='Health_Camp_ID' , how='left')
second_final=pd.merge(second,healt_camps,on ='Health_Camp_ID' , how='left')
third_final=pd.merge(third,healt_camps,on ='Health_Camp_ID' , how='left')

train_test=pd.concat([train,test]).reset_index(drop=True)
train_test_profile=pd.merge(train_test,patient_profile,on='Patient_ID',how='left')
data_profil_healt=pd.merge(train_test_profile, healt_camps,on='Health_Camp_ID',how='left')



ntrain=train.shape[0]
ntest=test.shape[0]

train =data_profil_healt.iloc[:ntrain,:]
test =data_profil_healt.iloc[ntrain:,:]



first.drop('Unnamed: 4',inplace=True,axis=1)



train_camps=pd.merge(train,healt_camps,on='Health_Camp_ID',how='left')
test_camps=pd.merge(test,healt_camps,on='Health_Camp_ID',how='left')

train_test=pd.concat([train_camps,test_camps]).reset_index(drop=True)

train_test1=pd.merge(train_test,first,on='Patient_ID',how='left')
train_test2=pd.merge(train_test1,second ,on='Patient_ID',how='left')
train_test3=pd.merge(train_test2,third,on='Patient_ID',how='left')

x_train=pd.merge(train,train_test3,on =[k for k in train.columns.values],how='left')
x_test=pd.merge(test,train_test3,on =[k for k in train.columns.values],how='left')



train_1=train.loc[train.Category1=='First',:]
train_1.drop('Category1',axis=1,inplace=True)

train_first=pd.merge(train_1,first,on=['Patient_ID','Health_Camp_ID'],how='left')

train_2=train.loc[train.Category1=='Second',:]
train_2.drop('Category1',axis=1,inplace=True)

train_second=pd.merge(train_2,second,on=['Patient_ID','Health_Camp_ID'],how='left')
train_second.rename(columns= {'Health Score':'Health_Score'},inplace=True)

train_3=train.loc[train.Category1=='Third',:]
train_3.drop('Category1',axis=1,inplace=True)

train_third=pd.merge(train_3,third,on=['Patient_ID','Health_Camp_ID'],how='left')



test_first=test.loc[test.Category1=='First',:]
test_first.drop('Category1',axis=1,inplace=True)

test_second=test.loc[test.Category1=='Second',:]
test_second.drop('Category1',axis=1,inplace=True)


test_third=test.loc[test.Category1=='Third',:]
test_third.drop('Category1',axis=1,inplace=True)


train_first['target']=train_first.Health_Score
train_first['target']=train_first.target.apply(lambda x :0 if np.isnan(x) else 1 )

train_second['target']=train_second.Health_Score
train_second['target']=train_second.target.apply(lambda x :0 if np.isnan(x) else 1 )

train_third['target']=train_third.Number_of_stall_visited
train_third['target']=train_third.target.apply(lambda x :0 if np.isnan(x) else 1 )



#save the 6 data sets for further modeling


train_first.to_csv('C:/Users/oussama/Documents/Knocktober/data/train_first.csv',index=None)
train_second.to_csv('C:/Users/oussama/Documents/Knocktober/data/train_second.csv',index=None)
train_third.to_csv('C:/Users/oussama/Documents/Knocktober/data/train_third.csv',index=None)

test_first.to_csv('C:/Users/oussama/Documents/Knocktober/test_first.csv',index=None)
test_second.to_csv('C:/Users/oussama/Documents/Knocktober/test_second.csv',index=None)
test_third.to_csv('C:/Users/oussama/Documents/Knocktober/test_third.csv',index=None)




