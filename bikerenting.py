#!/usr/bin/env python
# coding: utf-8

# In[91]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[92]:


# load dataset 
bike_count=pd.read_csv("/home/akshay/Downloads/day.csv")


# In[93]:


# check dimensions 
bike_count.shape


# In[94]:


bike_count.nunique()


# In[95]:


# lets have a look at data 
bike_count.head(6)


# In[96]:


# removing instant-just a index and dteday-day,month,yr as its not relevant  
predict_count=bike_count.drop(columns=["instant","dteday"],axis=1)


# In[97]:


# lets have a look at stastical information
predict_count.describe()


# In[100]:


#bike count
distribution=["season","yr","mnth","holiday","weekday","workingday","weathersit"]
for i in distribution:
    sns.catplot(x=i,y="cnt",data=predict_count,kind='bar')


# In[102]:


#missing value analysis
predict_count.isnull().sum()


# In[103]:


#outlier analysis
continuosvar= ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
categoricalvar = ["season","yr","mnth","holiday","weekday","workingday","weathersit"]



# In[104]:


for i in continuosvar:
    print(i)
    sns.boxplot(y=predict_count[i])
    plt.xlabel(i)
    plt.ylabel("values")
    plt.title("Boxplot of "+i)
    plt.show()


# In[105]:


#windspeed and hum is having outlier
for i in continuosvar:
    print(i)
    q75,q25= np.percentile(predict_count.loc[:,i],[75,25])
    iqr= q75-q25
    minimum= q25-(iqr*1.5)
    maximum= q75+(iqr*1.5)
    print("min= "+str(minimum))
    print("max= "+str(maximum))
    print("IQR= "+str(iqr))
    
#replace outliers with NA-
    predict_count.loc[predict_count[i]<minimum,i]=np.nan
    predict_count.loc[predict_count[i]>maximum,i]=np.nan


# In[106]:


#imputing NA with median-
predict_count['hum']=predict_count['hum'].fillna(predict_count['hum'].mean())
predict_count['windspeed']=predict_count['windspeed'].fillna(predict_count['windspeed'].mean())


# In[107]:


#check NA in data-
print(predict_count.isnull().sum())


# In[108]:


## Correlation 
sns.heatmap(predict_count[continuosvar].corr())
predict_count[continuosvar].corr()


# In[109]:


#feature scaling-data is uniformly distributed
for i in continuosvar:
    print(i)
    sns.distplot(predict_count[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()






# In[110]:


sns.heatmap(predict_count.corr())
predict_count.corr()


# In[111]:


#dropping atemp as it is highly correlated with temp
#dropping holiday,weekday,working day as they don't contribute much to cnt
predict=bike_count.drop(columns=["instant","dteday","atemp","holiday","weekday","workingday","casual","registered"],axis=1)
predict.shape
predict.head()


# In[112]:


#model development
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# In[113]:


train,test = train_test_split(predict, test_size = 0.2,random_state = 166)


# In[114]:


train.shape,test.shape


# In[115]:


# decision tree 
decisiontreemodel = DecisionTreeRegressor(random_state=166).fit(train.iloc[:,0:7], train.iloc[:,7])

print(decisiontreemodel)


# In[116]:


decisiontreemodeltpredict = decisiontreemodel.predict(test.iloc[:,0:7])
print(decisiontreemodeltpredict)


# In[117]:


predict_result = pd.DataFrame({'actual': test.iloc[:,7], 'pred': decisiontreemodeltpredict})
predict_result.head()


# In[118]:


#Calculate MAPE
def MAPE(y_true, y_pred): 
    MAE = np.mean(np.abs((y_true - y_pred)))
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    print("MAPE is: ",mape)
    print("MAE is: ",MAE)              
    return mape

def RMSE(y_test,y_predict):
    mse = np.mean((y_test-y_predict)**2)
    print("Mean Square : ",mse)
    rmse=np.sqrt(mse)
    print("Root Mean Square : ",rmse)
    return rmse

MAPE(test.iloc[:,7], decisiontreemodeltpredict)
RMSE(test.iloc[:,7], decisiontreemodeltpredict)


# In[119]:


#randomforest
randommodel = RandomForestRegressor(n_estimators = 1000, random_state = 166).fit(train.iloc[:,0:7], train.iloc[:,7])


# In[120]:


print(randommodel)


# In[121]:


randomforest = randommodel.predict(test.iloc[:,0:7])
print(randomforest)


# In[122]:


MAPE(test.iloc[:,7], randomforest)
RMSE(test.iloc[:,7], randomforest)


# In[123]:


pd.DataFrame(test).to_csv('Inputdata.csv', index = False)
pd.DataFrame(randomforest, columns=['predictions']).to_csv('outputRandomForestPy.csv')


# In[124]:


#linear regression
from sklearn.linear_model import LinearRegression


# In[125]:


lm = LinearRegression().fit(train.iloc[:,0:7], train.iloc[:,7])


# In[126]:


print(lm)


# In[127]:


linearmodeltpredict = lm.predict(test.iloc[:,0:7])


# In[128]:


print(linearmodeltpredict)


# In[129]:


MAPE(test.iloc[:,7], linearmodeltpredict)
RMSE(test.iloc[:,7], linearmodeltpredict)

