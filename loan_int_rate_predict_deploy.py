#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas.api.types as ptypes
import pickle
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
import math
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
regex = re.compile(r"\[|\]|<", re.IGNORECASE)


# ##### User Defined Functions

# In[2]:


def treat_num_columns(data, col_list):
    
    for i in col_list:
        col_name = i
        
        if (data[col_name].isnull().any().any() == True):
            
            if ptypes.is_numeric_dtype(cc_data[col_name]):
                # Replacing missing values with Median
                data[col_name].fillna(data[col_name].median(), inplace=True)
                print('Done')
            else:
                data[col_name].fillna(data[col_name].mode()[0], inplace=True)
                
   
# In[3]:


def find_outliers(get_data):
    
    for i in range(get_data.shape[1]):
        
        col_name = get_data.columns[i]
        
        q1 = get_data[col_name].quantile(0.25)
        q2 = get_data[col_name].quantile(0.50)
        q3 = get_data[col_name].quantile(0.75)
        iqr = q3 - q1
        LL = q1 - (1.5*iqr)
        UL = q3 + (1.5*iqr)
        

# In[5]:


def treat_outliers(get_data, col_list):
    for i in col_list:
        col_name = i
        
        q1 = get_data[col_name].quantile(0.25)
        q2 = get_data[col_name].quantile(0.50)
        q3 = get_data[col_name].quantile(0.75)
        iqr = q3 - q1
        low_limit = q1 - (1.5*iqr)
        upper_limit = q3 + (1.5*iqr)
        
    
        get_data[col_name] = np.where(get_data[col_name] > upper_limit, upper_limit, get_data[col_name])  ### values more than Upper Limit value, are replaced by Upper Limit value
        get_data[col_name] = np.where(get_data[col_name] < low_limit, low_limit, get_data[col_name])    ### values lower than Lower Limit value, are replaced by Lower Limit value
        
        sns.distplot(get_data[col_name], bins=15)
        #plt.show()


# In[6]:


loan_intrt_data = pd.read_csv(r'C:\Users\RONALD\Desktop\IMS-Classroom\Python Code\Resume Project - ML Algo\Loan Interest Rate prediction\Deployment\train_data.csv)


# In[7]:


loan_intrt_data.columns


# #### Rename columns

# In[8]:


#rename columns
loan_intrt_data.columns = ['ID','Amount_Requested','Amount_Funded_By_Investors','Interest_Rate','Loan_Length','Loan_Purpose',
                          'Debt_To_Income_Ratio','Home_Ownership','Monthly_Income','Open_Credit_Lines','Revolving_Credit_Balance',
                          'Inquiries_in_the_last_6months','Employment_Length','FICO_Score']

loan_intrt_data.columns


# In[9]:


loan_intrt_data.head(15)


# #### Correction in the data (if any)

# In[10]:


loan_intrt_data.groupby("Loan_Length")['ID'].nunique()


# In[11]:


loan_intrt_data['Loan_Length'].replace(['.'],['36 months'],inplace=True)


# In[12]:


loan_intrt_data.groupby('Loan_Length')['ID'].nunique()


# ### 4. DATA CLEANING & FORMATTING

# #### A. Missing Values Identification & Treatment

# In[13]:


loan_intrt_data.isnull().sum(axis=0)


# In[14]:


loan_data = loan_intrt_data.copy()     #------ copy dataset 
loan_data.shape


# In[15]:


loan_data.isnull().sum(axis=0)


# In[16]:


loan_data = loan_data.drop(['ID'], axis=1)  #--- drop ID
loan_data.columns


# In[17]:


#Splitting Numerical & Categorical columns

with_num_cols_data = loan_data[loan_data.select_dtypes(include=[np.number]).columns.tolist()]
with_cat_cols_data = loan_data[loan_data.select_dtypes(exclude=[np.number]).columns.tolist()]


# In[18]:


with_num_cols_data.columns  #numerical columns


# In[19]:


with_cat_cols_data.columns   #categorical columns


# ##### Checking for missing value in Numerical features and treating them

# In[22]:


num_col_name =  with_num_cols_data.columns


# In[23]:


treat_num_columns(with_num_cols_data, num_col_name)


# In[24]:


with_num_cols_data.isnull().sum(axis=0)


# ##### Checking for missing value in Categorical features and treating them

# In[25]:


cat_col_name = with_cat_cols_data.columns


# In[26]:


treat_cat_columns(with_cat_cols_data, with_cat_cols_data.columns)


# In[27]:


with_cat_cols_data.isnull().sum(axis=0)


# In[28]:


with_cat_cols_data.head(15)


# loan_data.isnull().sum(axis=0)

# #### B. Outlier Identification & Treatment

# In[30]:


find_outliers(with_num_cols_data)


# plot_scatter(with_num_cols_data)

# In[31]:


col_list = ['Amount_Funded_By_Investors', 'Amount_Requested', 'Open_Credit_Lines']


# In[32]:


treat_outliers(with_num_cols_data, col_list)


# #### C. Feature Engineering & Feature Scaling

# #### Numeric Columns

# In[33]:


with_num_cols_data.head()


# tarin

# #### Feature Engineering - Power Transformation

# In[34]:


cols_to_transform = ['Amount_Requested','Amount_Funded_By_Investors','Open_Credit_Lines','Monthly_Income', 'Revolving_Credit_Balance']

# from sklearn.preprocessing import PowerTransformer
# 
# pt = PowerTransformer()
# 
# with_num_cols_data[cols_to_transform] = pd.DataFrame(pt.fit_transform(with_num_cols_data[cols_to_transform]), columns=cols_to_transform)
# 
# #with_num_cols_data[cols_to_transform].hist(figsize=(14, 5))
# 

# In[35]:


with_num_cols_data[cols_to_transform].var()


# In[36]:


with_num_cols_data[cols_to_transform].agg(['skew']).transpose()


# ##### Overall Numeric Column Distribution

# In[37]:


with_num_cols_data.hist(figsize=(14, 5))


# ##### Feature scaling

# #### Categorical Columns: Label Encoding

# In[38]:


from sklearn.preprocessing import LabelEncoder

#with_cat_cols_data['Employment_Length'] = with_cat_cols_data['Employment_Length'].apply(LabelEncoder().fit_transform)

with_cat_cols_data.Loan_Length=(LabelEncoder().fit_transform(with_cat_cols_data.Loan_Length))
with_cat_cols_data.Loan_Purpose=(LabelEncoder().fit_transform(with_cat_cols_data.Loan_Purpose))
with_cat_cols_data.Home_Ownership=(LabelEncoder().fit_transform(with_cat_cols_data.Home_Ownership))
with_cat_cols_data.Employment_Length=(LabelEncoder().fit_transform(with_cat_cols_data.Employment_Length))


# In[39]:


with_cat_cols_data.head()


# ##### Column: FICO Score

# In[40]:


mapping = {'670-674':'Good', '675-679':'Good', '680-684':'Good',
           '695-699' : 'Good', '665-669' : 'Fair', '690-694' : 'Good',
           '685-689' : 'Good', '705-709' : 'Good', '700-704' : 'Good',
           '660-664' : 'Fair', '720-724' : 'Good', '710-714' : 'Good',
           '725-729' : 'Good', '730-734' : 'Good', '715-719' : 'Good',
           '735-739' : 'Good', '750-754' : 'Very Good', '745-749' : 'Very Good',
           '740-744' : 'Very Good', '755-759' : 'Very Good', '760-764' : 'Very Good',
           '765-769' : 'Very Good', '780-784' : 'Very Good', '775-779' : 'Very Good',
           '790-794' : 'Very Good', '785-789' : 'Very Good', '770-774' : 'Very Good',
           '795-799' : 'Very Good', '800-804' : 'Exceptional', '805-809' : 'Exceptional',
           '810-814' : 'Exceptional', '815-819' : 'Exceptional', '640-644' : 'Fair',
           '655-659' : 'Fair', '645-649' : 'Fair', '830-834' : 'Exceptional',
           '820-824' : 'Exceptional', '650-654' : 'Fair'

}


# In[41]:


with_cat_cols_data['FICO_Score_mapped'] = with_cat_cols_data['FICO_Score'].map(mapping)


# In[42]:


with_cat_cols_data.columns


# In[43]:


scale_mapper = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Exceptional': 3}


# In[44]:


with_cat_cols_data['FICO_Score_Scaled'] = with_cat_cols_data['FICO_Score_mapped'].replace(scale_mapper)


# cat_features_data_dummies['FICO_Score_Scaled'] = with_cat_cols_data['FICO_Score_Scaled']

# In[45]:


with_cat_cols_data.columns


# In[46]:


cat_features = ['Loan_Length','Loan_Purpose','Home_Ownership','Employment_Length','FICO_Score_Scaled']


# In[47]:


cat_cols_final = with_cat_cols_data[cat_features]


# In[48]:


cat_cols_final.columns

# #### Combining Numerical Columns and Categorical Columns into an updated dataset

# In[50]:


with_num_cols_data.columns


# In[51]:


updated_loan_data = pd.concat([with_num_cols_data, cat_cols_final], axis=1)
updated_loan_data.columns


# ##### Chec for any missing values

# In[52]:


updated_loan_data.isnull().sum()


# In[53]:


updated_loan_data.head()


# ### 5. DATA PARTITION

# In[54]:


X = updated_loan_data.drop(['Interest_Rate'], axis=1)
Y = updated_loan_data[['Interest_Rate']]


# In[55]:


X.columns


# In[56]:


Y.columns


# In[57]:


df_X = X.values


# In[58]:


from sklearn.ensemble import GradientBoostingRegressor

gbr_model = GradientBoostingRegressor()
gbr_model.fit(df_X, Y)


# In[59]:


gbr_model2 = GradientBoostingRegressor()
gbr_model2.fit(df_X, Y)


# In[60]:


pickle.dump(gbr_model, open('model3.pkl', 'wb'))


# In[61]:


model2 = pickle.load(open('model3.pkl', 'rb'))



