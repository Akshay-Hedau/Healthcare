#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


health = pd.read_excel('healthcare.xlsx')


# In[3]:


health.head()


# In[4]:


health.tail()


# 1(a)-Perform preliminary data inspection and report the findings on the structure of the data, missing values, duplicates, etc.

# In[5]:


health.info()


# In[6]:


health.shape


# In[7]:


health.describe().T


# In[8]:


health.isna().sum()


# In[9]:


health[health.duplicated(keep= False)]


# Our Data has no Missing Values, but it has a Duplicate ROW.

# 1(b) Based on these findings, remove duplicates (if any) and treat missing values using an appropriate strategy.

# In[10]:


#Drop the Duplicate Row.
health = health.drop(164)


# In[11]:


#Check size after dropping the duplicate row.
health.shape


# In[12]:


health[health.duplicated(keep= False)]


# NOW no Duplicate Values.

# CONCLUSION-
# 1. There are no missing values in our DataFrame.
# 2. Thare are no Duplicates in our DataFrame.
# 3. Our data has 302 ROWS and 14 COLUMNS.

# 2. Prepare a report about the data explaining the distribution of the disease and the related factors using the steps listed below:

#   2.(a)Get a preliminary statistical summary of the data and explore the measures of central tendencies and spread of the data.

# In[13]:


#Check the Distribution of Age in our Data.

plt.figure(figsize= (8,6))

sns.displot(data= health, x= "age")

plt.show()


# * Age is Continuous Feature and Normally Distributed. 

# In[14]:


#Check Sex column usng Count Plot.

plt.figure(figsize= (8,6))

sns.countplot(data= health, x= "sex")

plt.show()


# In[15]:


health["sex"].value_counts()


# * 0 is for FEMALE and 1 is for MALE
# * Here we can see that we have twice the number of observations for MALE than FEMALE. 

# In[16]:


#Check CP column (Chest Pain) using Count Plot.

plt.figure(figsize= (8,6))

sns.countplot(data= health, x= "cp")

plt.show()


# In[17]:


health["cp"].value_counts()


# * Chest Pain (cp): seems to be ordinal Categorical Variable.

# In[18]:


#Check trestbps column using Distribution Plot.

plt.figure(figsize= (8,6))

sns.displot(data= health, x= "trestbps")

plt.show()


# * Resting Blood Pressure(trestbps) is Continuous and seems to be Normally Distributed with Some Outliers at Right Tail.

# In[19]:


#Check chol column using Distribution Plot.

plt.figure(figsize= (8,6))

sns.displot(data= health, x= "chol")

plt.show()


# * Here we can see that Cholestrol(chol) is Continuous and Normally Distributed with some Outliers on the Right.

# In[20]:


#Check fbs column using Count Plot.

plt.figure(figsize= (8,6))

sns.countplot(data= health, x= "fbs")

plt.show()


# In[21]:


health["fbs"].value_counts()


# * Fasting Blood Sugar "fbs" is Ordinal Categorical Feature.

# In[22]:


#Check column restecg using Count Plot.

plt.figure(figsize= (8,6))

sns.countplot(data= health, x= "restecg")

plt.show()


# In[23]:


health["restecg"].value_counts()


# * Resting electrocardiographic results "restecg" is Ordinal Categorical Feature.

# In[24]:


#Check thalac column using Distribution Plot.

plt.figure(figsize= (8,6))

sns.displot(data= health, x= "thalach")

plt.show()


# * Maximum Heart Rate Achieved "thalach" is Continuous Feature and it is Left Skewed.

# In[25]:


#Check for column exang using Count Plot.

plt.figure(figsize= (8,6))

sns.countplot(data= health, x= "exang")

plt.show()


# In[26]:


health["exang"].value_counts()


# * Exercise Induced Enigma(exang) is Categorical Feature.

# In[27]:


#Check for column oldpeak using Distribution Plot.

plt.figure(figsize= (8,6))

sns.displot(data= health, x= "oldpeak")

plt.show()


# * ST depression induced by exercise relative to rest "oldpeak" is Continuous Feature and it is Highly Right Skewed.

# In[28]:


#Check for column slope using Count Plot.

plt.figure(figsize= (8,6))

sns.countplot(data= health, x= "slope")

plt.show()


# In[29]:


health["slope"].value_counts()


# * Slope of the peak exercise ST segment "slope" is Ordinal Categorical Feature.

# In[30]:


#Check for column 'ca' using Count Plot.

plt.figure(figsize= (8,6))

sns.countplot(data= health, x= "ca")

plt.show()


# In[31]:


health["ca"].value_counts()


# * Number of major vessels (0-3) colored by fluoroscopy "ca" is Ordinal Categorical Feature.

# In[32]:


#Check for column 'thal' using Count Plot.

plt.figure(figsize= (8,6))

sns.countplot(data= health, x= "thal")

plt.show()


# In[33]:


health["thal"].value_counts()


# * Thalassaemia "thal" is Nominal Categorical Variable.

# In[34]:


#Check for column 'target' using Count Plot.

plt.figure(figsize= (8,6))

sns.countplot(data= health, x= "target")

plt.show()


# In[35]:


health["target"].value_counts()


# * "target" is our Target Variable and we have No Class Imbalance here.

# 2.(b).	Identify the data variables which are categorical and describe and explore these variables using the appropriate tools, such as count plot.

# In[36]:


health.dtypes


# In[37]:


for col in health.columns:
    print(f"Number of Unique Values in {col} : {health[col].nunique()}")


# In[38]:


for col in health.columns:
   
    if health[col].nunique() <= 5:
        
        plt.figure(figsize= (6,4))

        sns.countplot(data= health, x= col)
        
        plt.title(f"Countplot of {col}")

        plt.show()


# 2.(c).	Study the occurrence of CVD across the Age category

# In[39]:


plt.figure(figsize= (8,6))

sns.displot(data= health, x= "age", col= "target")

plt.show()


# * 40-70 seems to be the Age range Where there are more chances of Cardiovascular Diseases.
# * Although, looking at target= 0 graph, 55-62 seems to be the Age Range in which Amny Observations from Our Data have no CVD.
# * Also, CVD seems to be present in all Age Ranges in our Data, which can be a Cause of Concern.

# 2.(d). Study the composition of all patients with respect to the Sex category.

# In[40]:


# We will Compare Features of all Observations with respect to Gender.


# In[41]:


for cols in health.drop("sex",axis= 1).columns:
    
    if health[cols].nunique() <= 5:
        
        plt.figure(figsize= (6,4))

        sns.countplot(data= health, x= cols, hue= "sex")
        
        plt.title(f"Countplot of {cols}")

        plt.show()
        
    else:
        
        plt.figure(figsize= (6,4))

        sns.displot(data= health, x= cols, col= "sex")
        
        plt.title(f"Distribution of {cols} by Gender:")

        plt.show()
        


# 2.(e).Study if one can detect heart attacks based on anomalies in the resting blood pressure (trestbps) of a patient

# In[42]:


plt.figure(figsize= (6,4))

sns.displot(data= health, x= "trestbps", col= "target")

plt.show()


# * We have some observations with very High Resting Blood Pressure values without occurence of CVD.
# * In general, we can see that Resting Blood Pressure values from 120-160 has more chances of CVD.
# * Still, This feature alone can not be said to be conclusive of CVD.

# 2.(f).Describe the relationship between cholesterol levels and a target variable.

# In[43]:


plt.figure(figsize= (6,4))

sns.displot(data= health, x= "chol", col= "target")

plt.show()


# * Here too, No considerable conclusion can be made about CVD by Cholesterol Levels alone.

# 2.(g). State what relationship exists between peak exercising and the occurrence of a heart attack.

# In[44]:


plt.figure(figsize= (6,4))

sns.displot(data= health, x= "oldpeak", col= "target")

plt.show()


# * As can be seen above, Lower Values of ST Depression Induced by Exercise relative to Rest clearly has more chances of CVD Occurence.

# In[45]:


plt.figure(figsize= (6,4))

sns.countplot(data= health, x= "slope", hue= "target")

plt.show()


# * Clear Relationship Between Slope of the Peak Exercise ST Segment and Occurence of CVD, having more value of "slope" clearly has more chances of CVD Occurence.

# 2.(h). Check if thalassemia is a major cause of CVD.

# In[46]:


plt.figure(figsize= (6,4))

sns.countplot(data= health, x= "thal", hue= "target")

plt.show()


# * As can be seen clearly, Thalassemia seems to be major Factor in Occurence of CVD.

# 2.(i). List how the other factors determine the occurrence of CVD.

# In[47]:


plt.figure(figsize= (18,8), dpi= 200)

sns.heatmap(health.corr(), annot= True)

plt.show()


# * Chest Pain (cp), Maximum Heart Rate Achieved (thalach), Slope of the peak exercise ST segment (slope) have Decently High Positive Correlation with Occurence of CVD.
# 
# * Exercise Induced Enigma (exang), ST depression induced by exercise relative to rest (oldpeak), Number of major vessels (0-3) colored by fluoroscopy (ca) and 
#   Thalassemia (thal) have Decently High Negative Correlation with Occurence of CVD.
# 
# * Cholesterol (chol) and Fasting Blood Sugar (fbs) have Very Low Correlation to Heart Disease.

# 2.(j). Use a pair plot to understand the relationship between all the given variables.

# In[48]:


plt.figure(dpi= 200)

sns.pairplot(health, hue= "target")

plt.show()


# * There aren't any Clearly Discernible Relationship Between any of the Features.

# 3.	Build a baseline model to predict the risk of a heart attack using a logistic regression and random forest and explore the results while using correlation analysis and logistic regression (leveraging standard error and p-values from statsmodels) for feature selection.

# In[49]:


#Separate Features and Target into diffrent DataFrame.
x = health.drop("target", axis= 1)


# In[50]:


x.head()


# In[51]:


x.shape


# In[52]:


y = health["target"]


# In[53]:


y.head()


# In[54]:


y.shape


# Using Generalized Linear Model from statsmodel library to determine which Features are Significant in Decidind Target Variable.

# In[55]:


from statsmodels.api import GLM
glm_model = GLM(y, x)


# In[56]:


glm_results = glm_model.fit()


# In[57]:


glm_results.summary()


# * There are Some Features which Have p-Value > 0.05.
# 
# * Those Features are not Significant in Predicting Target Variable.
# 
# * We will Build our Model Twice, once Using all The Features and Once Using Only Those Features deemed Significant by GLM.

# Creating new Data Frame with Feature deemed Significan by GLM.

# In[58]:


glm_results.pvalues


# In[59]:


glm_results.pvalues[glm_results.pvalues < 0.05]


# In[60]:


significant_cols = list(glm_results.pvalues[glm_results.pvalues < 0.05].index)


# In[61]:


significant_cols


# In[62]:


x_glm = x[significant_cols].copy()


# In[63]:


x_glm.head()


# Train Test Split of Datafrmae with All Features:

# In[64]:


from sklearn.model_selection import train_test_split, GridSearchCV


# In[65]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)


# In[66]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# Train Test Split of Datafrmae with GLM Features:

# In[67]:


x_glm_train, x_glm_test, y_train, y_test = train_test_split(x_glm, y, test_size= 0.2, random_state= 42)


# In[68]:


print(x_glm_train.shape)
print(x_glm_test.shape)
print(y_train.shape)
print(y_test.shape)


# Scalling of Datafrmae with All Features:

# In[69]:


from sklearn.preprocessing import StandardScaler


# In[70]:


sc_all = StandardScaler()


# In[71]:


temp = sc_all.fit_transform(x_train)
x_train = pd.DataFrame(temp, columns= x_train.columns)
x_train.head()


# In[72]:


temp = sc_all.transform(x_test)
x_test = pd.DataFrame(temp, columns= x_test.columns)
x_test.head()


# Scalling of Datafrmae with GLM Features:

# In[73]:


sc_glm = StandardScaler()


# In[74]:


temp = sc_glm.fit_transform(x_glm_train)
x_glm_train = pd.DataFrame(temp, columns= x_glm_train.columns)
x_glm_train.head()


# In[75]:


temp = sc_glm.transform(x_glm_test)
x_glm_test = pd.DataFrame(temp, columns= x_glm_test.columns)
x_glm_test.head()


# ** Building Logistic Regression Model and Random Forest Model:

# * Logistic Regression Model Using All Features:

# In[76]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix


# In[77]:


log_model_all = LogisticRegression()


# In[78]:


log_model_all.fit(x_train, y_train)


# In[79]:


preds = log_model_all.predict(x_test)


# In[80]:


print(classification_report(y_test, preds))


# In[81]:


plot_confusion_matrix(log_model_all, x_test, y_test)
plt.show()


# * Logistic Regression Model Using GLM Features:

# In[82]:


log_model_glm = LogisticRegression()


# In[83]:


log_model_glm.fit(x_glm_train, y_train)


# In[84]:


preds = log_model_glm.predict(x_glm_test)


# In[85]:


print(classification_report(y_test, preds))


# In[86]:


plot_confusion_matrix(log_model_glm, x_glm_test, y_test)
plt.show()


# * Random Forest Classifier Using All Features:

# In[87]:


from sklearn.ensemble import RandomForestClassifier


# In[88]:


rf_model_all = RandomForestClassifier()

rf_model_all.fit(x_train, y_train)

preds = rf_model_all.predict(x_test)

print(classification_report(y_test, preds))


# In[89]:


plot_confusion_matrix(rf_model_all, x_test, y_test)
plt.show()


# * Random Forest Classifier Using GLM Features:

# In[90]:


rf_model_glm = RandomForestClassifier()

rf_model_glm.fit(x_glm_train, y_train)

preds = rf_model_glm.predict(x_glm_test)

print(classification_report(y_test, preds))


# In[91]:


plot_confusion_matrix(rf_model_glm, x_glm_test, y_test)
plt.show()


# * We should use Significant Features Found using GLM to Train and Build Model to Predict CVD as it uses less features to Provide same Rate of Accuracy.
