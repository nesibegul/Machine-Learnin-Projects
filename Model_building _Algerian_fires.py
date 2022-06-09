#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import missingno as msno
import pickle
#import warnings
#warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# ### Data reading and feature scaling with MinMaxScaler()

# In[2]:


data = pd.read_csv('df_final_with10.csv')


# In[3]:


data.drop('Unnamed: 0', axis = 1, inplace = True)


# In[12]:


data


# In[13]:


X_columns  = ['Humidity', 'Wind', 'Rain', 'DMC', 'DC', 'Spread']
X_scale = data[X_columns]


# In[14]:


#scaling data
df = data.copy()
from sklearn.preprocessing import MinMaxScaler
Mim_max_X = MinMaxScaler()


X_part = pd.DataFrame(Mim_max_X.fit_transform(X_scale), columns = X_columns)


# In[15]:


X_part.head()


# In[16]:


df = data.copy()
df.drop(X_columns,axis = 1, inplace = True)


# In[17]:


df_new = df.join(X_part)


# In[18]:


col_order = data.columns
df_new = df_new[col_order]


# In[19]:


df_new


# ### Train and test data splitting

# In[20]:


X = df_new.drop('Classes', axis = 1)
y = df_new['Classes']


# In[43]:


#importing train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)


# ### KNN Classifier with GridSearchCV

# In[22]:


from sklearn.neighbors import KNeighborsClassifier
#import GridSearchCV
from sklearn.model_selection import GridSearchCV
#In case of classifier like knn the parameter to be tuned is n_neighbors
param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X_train,y_train)


# In[23]:


knn_cv.predict(X_test)


# In[24]:


print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))


# ### KNN searching best n_neighbor value manually

# In[25]:


test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))


# In[26]:


print(train_scores)
print(test_scores)


# #### n_neighbor value is same as gridsearchcv

# max_train_score = max(train_scores)
# train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
# print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))

# In[27]:


## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))


# In[28]:


y_pred = knn.predict(X_test)
y_pred


# ### Confusion matrix and Classification report

# In[29]:


#import confusion_matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[30]:


# Creating a Heatmap for the confusion matrix. 
y_pred = knn.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[46]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# ### Logistic Regression with GridSearchCV for parameter tuning

# In[35]:


##Appy Logistic regression
from sklearn.linear_model import LogisticRegression
lg_reg = LogisticRegression(random_state = 2)

#import GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = {
    'penalty' : ['l1','l2', 'elasticnet'], 
    'C'       : np.logspace(-3,3,7),
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
}
log_reg = LogisticRegression(random_state = 2)
logreg_cv= GridSearchCV(log_reg,parameters, cv=5)
logreg_cv.fit(X_train,y_train)


# In[36]:


logreg_cv.predict(X_test)


# ### Accuracy, classification report and best parameters 

# In[37]:


print("Best Score:" + str(logreg_cv.best_score_))
print("Best Parameters: " + str(logreg_cv.best_params_))


# In[38]:


logreg_best = LogisticRegression(C = 10, 
                            penalty = 'l1', 
                            solver = 'liblinear')
logreg_best.fit(X_train,y_train)
y_pred = logreg_best.predict(X_test)
y_pred


# In[40]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# ### SVM Model with GridSearchCV and Evaluation

# In[45]:


### Apply SVM model

from sklearn.svm import SVC
clssfr = SVC(random_state = 2)
parameters = {
    'kernel'  : ['linear', 'poly', 'rbf', 'sigmoid'], 
    'C'       : np.logspace(-3,3,7),
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
}
clssfr_cv= GridSearchCV(clssfr,parameters, cv=5)
clssfr_cv.fit(X_train,y_train)


# In[46]:


print(clssfr_cv.predict(X_test))
print("Best Score:" + str(clssfr_cv.best_score_))
print("Best Parameters: " + str(clssfr_cv.best_params_))


# In[47]:


y_pred = clssfr_cv.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# ### Decision tree and parameter tuning and evaluation

# In[63]:


### Apply Decision tree model

from sklearn.tree import DecisionTreeClassifier
clssfr_dcsn_tree = DecisionTreeClassifier(random_state = 2)

parameters = {
    'criterion' : ['gini', 'entropy', 'log_loss'],
    'min_samples_leaf'       : np.arange(1,10)
    
}
clssfr_dcsn_tree_cv= GridSearchCV(clssfr_dcsn_tree, parameters, cv=5)
clssfr_dcsn_tree_cv.fit(X_train,y_train)


# In[64]:


print(clssfr_dcsn_tree_cv.predict(X_test))
print("Best Score:" + str(clssfr_dcsn_tree_cv.best_score_))
print("Best Parameters: " + str(clssfr_dcsn_tree_cv.best_params_))


# In[65]:


y_pred = clssfr_dcsn_tree_cv.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# ### NaiveBayes and Evaluation

# In[56]:


### Apply Naive Bayes model
from sklearn.naive_bayes import BernoulliNB
clssfr = BernoulliNB()

#import GridSearchCV
from sklearn.model_selection import GridSearchCV

parameters = {
    'alpha':[1,0]
    
    
}
clssfr_cv= GridSearchCV(clssfr,parameters, cv=5)
clssfr_cv.fit(X_train,y_train)


# In[57]:


print(clssfr_cv.predict(X_test))
print("Best Score:" + str(clssfr_cv.best_score_))
print("Best Parameters: " + str(clssfr_cv.best_params_))


# In[58]:


y_pred = clssfr_cv.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# ### RandomForest with best parameters and evaluation

# In[60]:


### Apply Random Forest model

from sklearn.ensemble import RandomForestClassifier
clssfr = RandomForestClassifier(random_state = 2)

parameters = {
    
 'bootstrap': [True, False],
 'max_depth': np.arange(4,10),
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'n_estimators': [200, 400, 600]

}

clssfr_cv= GridSearchCV(clssfr, parameters, cv=5)
clssfr_cv.fit(X_train,y_train)


# In[61]:


print(clssfr_cv.predict(X_test))
print("Best Score:" + str(clssfr_cv.best_score_))
print("Best Parameters: " + str(clssfr_cv.best_params_))


# In[62]:


y_pred = clssfr_cv.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# ### Select the best classification model based on accuracy-precision-recall and dump to use later

# In[67]:


#best model can be selected as decision tree classifier

pickle.dump(clssfr_dcsn_tree_cv, open("class_model.pkl", 'wb'))
##my_model = pickle.load(open("class_model.pkl", "rb"))


# # Regression modeling 

# In[68]:


#dependent variable will be rain column and remaining columns except categoricals go through minmax scale


# In[69]:


X_columns  = ['Humidity', 'Wind',  'DMC', 'DC', 'Spread']
X_scale = data[X_columns]


# In[70]:


#scaling data
df = data.copy()
from sklearn.preprocessing import MinMaxScaler
Mim_max_X = MinMaxScaler()


X_part = pd.DataFrame(Mim_max_X.fit_transform(X_scale), columns = X_columns)


# In[71]:


X_part.head()


# In[72]:


df = data.copy()


# In[73]:


df.drop(X_columns,axis = 1, inplace = True)
df_new = df.join(X_part)


# In[76]:


col_order = ['Day', 'Month', 'Humidity', 'Wind', 'DMC', 'DC', 'Spread',
       'Region', 'Classes', 'Rain']


# In[95]:



df_new = df_new[col_order]
df_new


# In[104]:


X = df_new.drop('Rain', axis = 1)
y = df_new['Rain']
y.values


# ### Linear Regression and evaluation with mse and r2

# In[105]:


#importing train_test_split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[106]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

lin_regressor=LinearRegression()

mse=cross_val_score(lin_regressor,X_train,y_train,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse) #train data result
print(mean_mse)


# In[107]:


mse=cross_val_score(lin_regressor,X_test,y_test,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse) #test data result
print(mean_mse)


# In[108]:


from sklearn.metrics import r2_score, mean_squared_error
lin_regressor.fit(X_train, y_train)
y_pred= lin_regressor.predict(X_test)
print("r2_score:", r2_score(y_test, y_pred))
print("mean_squared_error: ",mean_squared_error(y_test, y_pred))


# ### Ridge regression with hyperparameter tuning and evaluation

# In[109]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,y_train)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[110]:


from sklearn.metrics import r2_score, mean_squared_error
ridge_regressor.fit(X_train, y_train)
y_pred= ridge_regressor.predict(X_test)
print("r2_score:", r2_score(y_test, y_pred))
print("mean_squared_error: ",mean_squared_error(y_test, y_pred))


# ### Lasso with GridSearchCV and evaluation with r2 and mse

# In[111]:


from sklearn.linear_model import Lasso
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[112]:


lasso_regressor.fit(X_train, y_train)
y_pred= lasso_regressor.predict(X_test)
print("r2_score:", r2_score(y_test, y_pred))
print("mean_squared_error: ",mean_squared_error(y_test, y_pred))


# ### ElasticSearch and Evaluation

# In[113]:


from sklearn.linear_model import ElasticNet
regress=ElasticNet()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
regressor_cv=GridSearchCV(regress,parameters,scoring='neg_mean_squared_error',cv=5)

regressor_cv.fit(X,y)
print(regressor_cv.best_params_)
print(regressor_cv.best_score_)


# In[114]:


from sklearn.metrics import r2_score, mean_squared_error
regressor_cv.fit(X_train, y_train)
y_pred= regressor_cv.predict(X_test)
print("r2_score:", r2_score(y_test, y_pred))
print("mean_squared_error: ",mean_squared_error(y_test, y_pred))


# ### SVR and r2-mse values

# In[117]:


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
regress=SVR()
parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.5,0.3]}
regressor_cv=GridSearchCV(regress,parameters,scoring='neg_mean_squared_error',cv=5)

regressor_cv.fit(X,y)
print(regressor_cv.best_params_)
print(regressor_cv.best_score_)
print(regressor_cv.score)


# In[118]:


from sklearn.metrics import r2_score, mean_squared_error
regressor_cv.fit(X_train, y_train)
y_pred= regressor_cv.predict(X_test)
print("r2_score:", r2_score(y_test, y_pred))
print("mean_squared_error: ",mean_squared_error(y_test, y_pred))


# ### DecisionTree Regression and parameter tuning

# In[119]:


from sklearn.tree import DecisionTreeRegressor
regress=DecisionTreeRegressor()
parameters = {'criterion': ("squared_error", "friedman_mse", "absolute_error", "poisson"), 'min_samples_split':np.arange(2,10)}
regressor_cv=GridSearchCV(regress,parameters,scoring='neg_mean_squared_error',cv=5)

regressor_cv.fit(X,y)
print(regressor_cv.best_params_)
print(regressor_cv.best_score_)
print(regressor_cv.score)


# In[120]:


from sklearn.metrics import r2_score, mean_squared_error
regressor_cv.fit(X_train, y_train)
y_pred= regressor_cv.predict(X_test)
print("r2_score:", r2_score(y_test, y_pred))
print("mean_squared_error: ",mean_squared_error(y_test, y_pred))


# ### RandomForest and evaluation

# In[128]:


from sklearn.ensemble import RandomForestRegressor
regress=RandomForestRegressor()
parameters = {
    'max_depth': [80,  110],
    'min_samples_leaf': [3, 4, 5],
    'n_estimators': [100, 200, 300, 1000]
              
             }
regressor_cv=GridSearchCV(regress,parameters,scoring='neg_mean_squared_error',cv=5)

regressor_cv.fit(X,y)
print(regressor_cv.best_params_)
print(regressor_cv.best_score_)
print(regressor_cv.score)


# In[129]:


from sklearn.metrics import r2_score, mean_squared_error
regressor_cv.fit(X_train, y_train)
y_pred= regressor_cv.predict(X_test)
print("r2_score:", r2_score(y_test, y_pred))
print("mean_squared_error: ",mean_squared_error(y_test, y_pred))


# ###  Best model is random forest, dump for prediction of rain 

# In[130]:


pickle.dump(regressor_cv, open("regression_model.pkl", 'wb'))
##my_model = pickle.load(open("class_model.pkl", "rb"))


# In[ ]:




