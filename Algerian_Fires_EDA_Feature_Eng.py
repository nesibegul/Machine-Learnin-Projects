#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import missingno as msno
import warnings
warnings.filterwarnings('ignore')


# ### Import data and handle with trim space in values and column names

# In[2]:


pd.set_option('display.max_columns', None)


# In[3]:


data1 = pd.read_csv('Algerian_forest_fires_dataset_UPDATE.csv', header = 1, nrows = 122)


# In[4]:


data1['Region'] = 'Bejaia'


# In[5]:


data1.tail()


# In[6]:


data2 = pd.read_csv('Algerian_forest_fires_dataset_UPDATE.csv', header = 125, nrows = 122)


# In[7]:


data2['Region'] = 'Sidi-Bel Abbes'


# In[8]:


data2.tail()


# In[9]:


data1.append(data2)


# In[10]:


data = data1.append(data2)
data.reset_index(inplace = True, drop = True )
data.tail()


# In[11]:


pd.DataFrame({'nulls': data.isnull().sum(), 'data_types': data.dtypes, 'unique_variables': [len(data[feature].unique()) for feature in data.columns]})


# ### Change column data types

# In[12]:


#Change the columns data types

'''print(pd.DataFrame(data['DC'].str.replace(' ', '')) )          
x = pd.DataFrame(data['DC'].str.replace(' ', ''))
x.head(30)
pd.DataFrame(data['DC'])
'''
data.loc[data['DC']=='14.6 9','DC']='14.69'
data['DC'].apply(lambda x: float(x))


# In[13]:


#Change the columns data types

data.loc[data['FWI']=='fire   ', 'FWI']= np.NaN
data.loc[data['FWI']=='fire   ', 'Classes  ']='fire'
data['FWI'].apply(lambda x: float(x))


# In[14]:


#Change the columns data types

data['DC'] = data['DC'].astype('float64')
data['FWI'] = data['FWI'].astype('float64')
print(data[['FWI']].info())
print(data[['DC']].info())


# In[15]:


column_ord = data.columns 
column_ord
data.info()


# In[16]:


column_ord = list(map(lambda x: x.strip(), column_ord))
column_ord


# In[17]:



data.columns = column_ord


# In[18]:


data.columns


# In[19]:


data.rename(columns={'day':'Day', 'month':'Month', 'year':'Year', 'RH':'Humidity', 'Ws': 'Wind', 'ISI':'Spread', 'BUI': 'Buildup', 'FWI':'Weather' }, inplace = True)


# In[20]:


dt_obj = data.select_dtypes(['object'])

data[dt_obj.columns] = dt_obj.apply(lambda x: x.str.rstrip())


# In[21]:


column_ord = data.columns
column_ord = ['Day', 'Month', 'Year', 'Temperature', 'Humidity', 'Wind', 'Rain',
       'FFMC', 'DMC', 'DC', 'Spread', 'Buildup', 'Weather',
       'Region', 'Classes']
data = data[column_ord]


# In[22]:


pd.DataFrame({'nulls': data.isnull().sum(), 'data_types': data.dtypes, 'unique_variables': [len(data[feature].unique()) for feature in data.columns]})


# ### Exploratory Data Analysis

# In[23]:


#Drop the year column which has just year 2012
df = data.copy()
data.drop(columns = 'Year', axis = 1, inplace = True)


# In[24]:


data['Classes'].value_counts()


# In[391]:


df = data.copy()
for feature in df.columns:
    sns.displot(df[feature])
    plt.title(feature)
    plt.show()


# In[412]:





# In[402]:


numeric_features = [feature for feature in data.columns if data[feature].dtypes != 'object']


# In[413]:


for feature in numeric_features:
    sns.catplot( feature, kind = 'box',  data = df)
    plt.title(feature)
    plt.show()


# In[410]:


for feature in numeric_features:
    sns.catplot(y = feature, x = 'Classes',  data = df)
    plt.title(feature)
    plt.show()


# In[466]:


for feature in numeric_features:
    sns.catplot(y = feature, x = 'Classes', col = 'Month',  data = df)
    plt.title(feature)
    plt.show()


# In[454]:


sns.catplot(y = 'Day', x = 'Classes', col = 'Month', kind ='violin',  data = df)


# In[467]:


df =data.copy()
p=sns.pairplot(df, hue = 'Classes')


# ### Encode the data and handle the outliers

# In[25]:


##Encode Classes and Region columns 
df = data.copy()
df['Classes'].unique()


# In[26]:


data['Classes'] = data['Classes'].replace(('fire', 'not fire'), (1, 0))


# In[27]:


data['Classes'] = data['Classes'].replace(np.NaN, 1)
data['Classes'] = data['Classes'].astype('int')


# In[28]:


data['Region'] = data['Region'].replace(('Bejaia', 'Sidi-Bel Abbes'), (1, 0))


# In[29]:


data.Classes.unique()


# In[30]:


##Detecting outliers
df = data.copy()

def detect_outliers(data):
    outliers=[]
    threshold=3
    mean = np.mean(data)
    std =np.std(data)
       
    for i in data:
        z_score= (i - mean)/std 
        if np.abs(z_score) > threshold:
            outliers.append(i)
    print( 'critical values are ', mean+3*std, mean-3*std )
    return outliers

print(detect_outliers(df['Wind']))
print(detect_outliers(df['Rain']))
print(detect_outliers(df['DMC']))
print(detect_outliers(df['FFMC']))


# In[31]:


#Impute the outliers
data.loc[df['Wind']>=24, 'Wind'] = 24
data.loc[df['Wind']<=7, 'Wind'] = 7


# In[32]:


data.loc[df['Rain']>=7, 'Rain'] = 7
data.loc[df['DMC']>=52, 'DMC'] = 52
data.loc[df['FFMC']<=35, 'FFMC'] = 35


# In[489]:


df = data.copy()
for feature in ['Wind', 'Rain', 'DMC', 'FFMC']:
    sns.displot(df[feature])
    plt.title(feature)
    plt.show()


# ### Imbalance data check

# In[33]:


#imbalanced data control
data['Classes'].value_counts()
#The class in an imbalanced classification predictive modeling problem that has slightly more examples. 
#thus, there is no imbalanced data problem, but we can add some sampling in data get equal sample


# In[34]:


from sklearn.utils import resample
df = data.copy()
#create two different dataframe of majority and minority class 
df_majority = df[(df['Classes']==1)] 
df_minority = df[(df['Classes']==0)] 
# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples= 138, # to match majority class
                                 random_state=42)  # reproducible results
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])


# In[35]:


df_upsampled.reset_index(drop = True, inplace = True)
df_upsampled.head()


# ### Correlation and Multicollinearity control

# In[500]:


#Multicollinearity check
plt.figure(figsize= (12,10))
sns.heatmap(df_upsampled.corr(), annot = True, cmap = 'YlGnBu')


# ### Dropping multicollinear columns based on VIF value which goes high with such data

# In[36]:


df_upsampled.drop(index = 219, inplace = True)


# In[38]:



column_inc =  ['Day', 'Month',  'Humidity', 'Wind', 'Rain', 
       'DMC', 'DC', 'Spread',   'Region']


# In[39]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


X = df_upsampled.iloc[:, :-1]
X = X[column_inc]
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] =column_inc

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)


# In[40]:


df_upsampled.corr()


# In[41]:


#final columns are 
column_inc = ['Day', 'Month', 'Humidity', 'Wind', 'Rain', 'DMC', 'DC', 'Spread', 'Region', 'Classes']


# In[42]:


#Check point and update  our dataset
df_final = df_upsampled.copy()
df_final = df_final[column_inc]


# ### Save the final datasets

# In[43]:


df_final


# In[ ]:


df_final.to_csv('df_final_with10.csv') #with multicollinearity check
df_upsampled.to_csv('df_upsampledwith14.csv') #with whole dataset

