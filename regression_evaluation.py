#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

path_test ='../datasets_coursework/Wine/wine_test.csv'
path_train ='../datasets_coursework/Wine/wine_train.csv'
train_set = pd.read_csv(path_train,sep=';') 
test_set = pd.read_csv(path_test,sep=';') 


# In[2]:


y = train_set.quality
x = train_set.drop('quality',axis = 1)


# In[3]:


from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(x,y)
#train


# In[4]:


test_x = test_set.drop('quality',axis = 1)
test_y = test_set.quality


# In[5]:


y_prediction = elastic_net.predict(test_x)


# In[6]:


MSE = mean_squared_error(test_y, y_prediction)
RMSE = np.sqrt(MSE)
RMSE


# In[7]:


from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(x, y)
y_prediction = lasso_reg.predict(test_x)

#y_prediction = sgd_reg.predict(test_x)


# In[8]:


MSE = mean_squared_error(test_y, y_prediction)
RMSE = np.sqrt(MSE)
RMSE


# In[ ]:





# In[ ]:




