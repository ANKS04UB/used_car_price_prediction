#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


cars=pd.read_csv("cars_sampled.csv")


# In[3]:


cars.head()


# In[4]:


cars.shape


# In[5]:


data=cars.copy()


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


# conversion into readable format
pd.set_option('display.float_format',lambda x: '%.3f'% x)


# In[9]:


data.describe()


# In[10]:


# to display maximum set of columns
pd.set_option('display.max_columns',500)
data.describe()


# In[11]:


# dropping unwanted data
data.columns


# In[12]:


unwanted_col=['name','dateCrawled','dateCreated','postalCode','lastSeen']


# In[13]:


data=data.drop(columns=unwanted_col,axis=1)


# In[14]:


data.head()


# In[15]:


# removing duplicate data
data.drop_duplicates(keep='first',inplace=True)


# In[16]:


data.shape


# In[17]:


data.info()


# In[18]:


data.isnull().sum()


# In[19]:


# variable year of registration
yearwise_count=data['yearOfRegistration'].value_counts().sort_index()


# In[20]:


sum(data['yearOfRegistration']>2018)


# In[21]:


sum(data['yearOfRegistration']<1950)


# In[22]:


sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=False,data=data)


# #we have to working on yearofregistration of working range is 1950 and 2018

# In[23]:


# variable price
price_count=data['price'].value_counts().sort_index()


# In[24]:


sns.distplot(data['price'])


# In[25]:


data['price'].describe()


# In[26]:


sns.boxplot(y=data['price'])


# In[27]:


sum(data['price']>150000)


# In[28]:


sum(data['price']<100)


# # working on range 100 and 150000

# In[29]:


# Variable power ps
power_count=data['powerPS'].value_counts().sort_index()


# In[30]:


sns.distplot(data['powerPS'])


# In[31]:


data['powerPS'].describe()


# In[32]:


sns.boxplot(y=data['powerPS'])


# In[33]:


sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=data)


# In[34]:


sum(data['powerPS']>500)


# In[35]:


sum(data['powerPS']<10)


# ## working range 10 to 500

# In[36]:


# working on range of data


# In[37]:


data=data[(data.yearOfRegistration <= 2018)&(data.yearOfRegistration >= 1950) & (data.price >= 100)& (data.price <= 150000)& (data.powerPS >= 10)&(data.powerPS<=500)]
 


# In[38]:


data.shape


# In[39]:


data['monthOfRegistration']/=12


# In[40]:


data['age']=(2018-data['yearOfRegistration'])+data['monthOfRegistration']


# In[41]:


data['age']=round(data['age'],2)


# In[42]:


data['age'].describe()


# In[43]:


# drop them
data=data.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)


# In[44]:


# visulise parameter


# In[45]:


sns.distplot(data['age'])


# In[46]:


sns.boxplot(y=data['age'])


# In[47]:


# price
sns.distplot(data['price'])


# In[48]:


sns.boxplot(y=data['price'])


# In[49]:


# powerps
sns.distplot(data['powerPS'])


# In[50]:


sns.boxplot(y=data['powerPS'])


# In[51]:


sns.regplot(x='age',y='price',scatter=True,fit_reg=False,data=data)


# In[52]:


sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=data)


# In[53]:


# variable seller
data['seller'].value_counts()


# In[54]:


pd.crosstab(data['seller'],columns='count',normalize=True)


# In[55]:


sns.countplot(x='seller',data=data)


# In[56]:


# commercials are insignificance


# In[57]:


# variable offertype
data['offerType'].value_counts()


# In[58]:


sns.countplot(x='offerType',data=data)


# In[59]:


# all cars have only 1 offertype so it insignificant


# In[60]:


# variable abtest
data['abtest'].value_counts()


# In[61]:


pd.crosstab(data['abtest'],columns='count',normalize=True)


# In[62]:


sns.countplot(x='abtest',data=data)


# In[63]:


# it is equally distributed


# In[64]:


sns.boxplot(x='abtest',y='price',data=data)


# In[65]:


# for every price value there is 50-50 distribution it is does not affect price >= insingificant


# In[66]:


# variable vehicletype
data['vehicleType'].value_counts()


# In[67]:


pd.crosstab(data['vehicleType'],columns='count',normalize=True)


# In[68]:


sns.countplot(x='vehicleType',data=data)


# In[69]:


sns.boxplot(x='vehicleType',y='price',data=data)


# In[70]:


# it affect on price 


# In[71]:


# variable gearbox
data['gearbox'].value_counts()


# In[72]:


pd.crosstab(data['gearbox'],columns='count',normalize=True)


# In[73]:


sns.countplot(x='gearbox',data=data)


# In[74]:


sns.boxplot(x='gearbox',y='price',data=data)


# In[75]:


# gearbox affect on price


# In[76]:


# variable model
data['model'].value_counts()


# In[77]:


pd.crosstab(data['model'],columns='count',normalize=True)


# In[78]:


sns.countplot(x='model',data=data)


# In[79]:


sns.boxplot(x='model',y='price',data=data)


# In[80]:


# cars are distributed over many models, considered in modeling


# In[81]:


# variable kilometer
data['kilometer'].value_counts().sort_index()


# In[82]:


pd.crosstab(data['kilometer'],columns='count',normalize=True)


# In[83]:


sns.countplot(x='kilometer',data=data)


# In[84]:


sns.boxplot(x='kilometer',y='price',data=data)


# In[85]:


data['kilometer'].describe()


# In[86]:


# considered in modeling


# In[87]:


# variable fueltype
data['fuelType'].value_counts()


# In[88]:


pd.crosstab(data['fuelType'],columns='count',normalize=True)


# In[89]:


sns.countplot(x='fuelType',data=data)


# In[90]:


sns.boxplot(x='fuelType',y='price',data=data)


# In[91]:


# fueltype affect to price


# In[92]:


# variable brand
data['brand'].value_counts()


# In[93]:


pd.crosstab(data['brand'],columns='count',normalize=True)


# In[94]:


sns.countplot(x='brand',data=data)


# In[95]:


sns.boxplot(x='brand',y='price',data=data)


# In[96]:


# cars also distributing over brands, it affect price 


# In[97]:


# variable notRepairedDamage yes-car is damaged but not rectified no- car was damaged but has been rectified


# In[98]:


data['notRepairedDamage'].value_counts()


# In[99]:


pd.crosstab(data['notRepairedDamage'],columns='count',normalize=True)


# In[100]:


sns.countplot(x='notRepairedDamage',data=data)


# In[101]:


sns.boxplot(x='notRepairedDamage',y='price',data=data)


# In[102]:


# the cars that require damages to be repaired fall under lower price ranges


# In[103]:


# Removing insignificance variables


# In[104]:


column=['seller','offerType','abtest']


# In[105]:


data=data.drop(columns=column,axis=1)


# In[106]:


cars_copy=data.copy()


# In[107]:


cars_copy.head()


# In[108]:


# correlation 


# In[109]:


cars_select1=data.select_dtypes(exclude=['object'])


# In[110]:


correlation=cars_select1.corr()


# In[111]:


round(correlation,3)


# In[112]:


cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]


# In[113]:


# ommited missing values
data_ommit=data.dropna(axis=0)


# In[114]:


# converting categorical into numerical
data_ommit=pd.get_dummies(data_ommit,drop_first=True)


# In[115]:


# importing neccesary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[116]:


# model building with ommited data
x1=data_ommit.drop(['price'],axis='columns',inplace=False)


# In[117]:


y1=data_ommit['price']


# In[118]:


# plotting variable price
prices=pd.DataFrame({'1.Before':y1,'2.After':np.log(y1)})
prices.hist()


# In[119]:


# Transforming price as a logarithmic value
y1=np.log(y1)


# In[120]:


# splitting data
X_train, X_test, y_train, y_test = train_test_split(x1,y1,test_size=0.3, random_state=3)


# In[121]:


# baseline model for ommited data
base_pred=np.mean(y_test)


# In[122]:


base_pred


# In[123]:


# repeating same value till lenght of test data
base_pred=np.repeat(base_pred,len(y_test))


# In[124]:


# finding the RMSE
base_root_mean_square_error = np.sqrt(mean_squared_error(y_test, base_pred))


# In[125]:


base_root_mean_square_error


# In[126]:


# model with ommited data
lg = LinearRegression(fit_intercept=True)


# In[127]:


model_lin = lg.fit(X_train, y_train)


# In[128]:


# predicting model on test set
cars_prediction_lin1= lg.predict(X_test)


# In[129]:


# computing MSE and RMSE
lin_mse = mean_squared_error(y_test,cars_prediction_lin1)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[130]:


# r square value
r2_lin_test = model_lin.score(X_test, y_test)
r2_lin_train = model_lin.score(X_train, y_train)
r2_lin_test, r2_lin_train


# In[132]:


# Random forest with omited data


# In[135]:


rf = RandomForestRegressor(n_estimators = 100, max_features='auto',max_depth= 100, min_samples_split=10, min_samples_leaf= 4,random_state=1)


# In[136]:


model_rf = rf.fit(X_train, y_train)


# In[137]:


# predicting model on test set
cars_prediction_rf1 = rf.predict(X_test)


# In[138]:


# computing MSE and RMSE
rf_mse = mean_squared_error(y_test,cars_prediction_rf1 )
rf_rmse = np.sqrt(rf_mse)
rf_mse,rf_rmse


# In[139]:


# r_square value
r2_rf_test1 = model_rf.score(X_test, y_test)
r2_rf_train1 = model_rf.score(X_train,y_train)
r2_rf_test1, r2_rf_train1


# In[ ]:





# In[ ]:




