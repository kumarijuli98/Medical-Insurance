#!/usr/bin/env python
# coding: utf-8

# # Medical Cost Personal

# Table of Contents
# 
# 1 [Problem Statement](#section1)<br>
#  1.1 [Introduction](#section101)<br/>
#  1.2 [Data source and data set](#section102)<br/>
# 2. [Load the Packages and Data](#section2)<br/>
# 3. [Data Profiling](#section3)<br/>
#  3.1 [Understanding the Dataset](#section301)<br/>
#  3.2 [Pre Profiling](#section302)<br/>
#  3.3 [Preprocessing](#section303)<br/>
#  3.4 [Post Profiling](#section304)<br/>
#  4. Data Exploration

#  1.Problem Statement:
# 
#  1.1 Introduction
#  
#  E-commerce (electronic commerce) is the buying and selling of goods and
#  services, or the transmitting of funds or data, over an electronic network, primarily
#  the internet. These business transactions occur either as business-to-business
#  
#  (B2B), business-to-consumer (B2C), consumer-to-consumer or consumer-to-
#  business.
# 
#  
# 

# 1.2 Data source and data set
# ##You can find the dataset on the given link
# 
#  https://www.kaggle.com/srolka/ecommerce-customer
#     

# #2. Load the Packages and Data
# 

# __Importing Packages__

# In[1]:


pip install pandas_profiling


# ## first Import the Important Libraries

# In[3]:


import numpy as np                                                 # Implemennts milti-dimensional array and matrices
import pandas as pd                                                # For data manipulation and analysis
import ydata_profiling
import matplotlib.pyplot as plt                                    # Plotting library for Python programming language and it's numerical mathematics extension NumPy
import seaborn as sns                                              # Provides a high level interface for drawing attractive and informative statistical graphics
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
from subprocess import check_output 


# # Loading Data set

# In[4]:


medi = pd.read_csv(r'D:\work\python\Projects\Medical Cost Personal Datasets\archive\insurance.csv')


# In[5]:


medi


# 
# # # 3.Data Profiling

#  3.1 Understanding the Dataset

# In[6]:


medi.shape



# In[7]:


medi.columns


# In[8]:



medi.info()


# In[9]:


medi.head()


# In[10]:


medi.describe()


# In[11]:



medi.describe(include="all")


# In[12]:


medi.sort_values(by=['charges'],ascending= False).head(10)


# In[20]:


# Checking the total number of missing values

medi.isnull().sum()


# There is no null values in this data set

# In[22]:


# We have checked the raw data and there aren't any missing values!

medidata = medi.copy()


# # 4. Data Exploration

# Displaying the probability distribution function (PDF) of a variable is a fantastic data exploration step. We will see how that variable is spread in the PDF. This makes it very simple to identify outliers and other irregularities. Frequently, the PDF also serves as the foundation for our decision over whether to alter a feature.

# In[23]:


# distribution of age variable

sns.displot(medidata['age'])
plt.title('Distribution of Age')
plt.show()


# As the figure represents that the highest density of people is of age 20-23. From age 24 to 70, the distribution is almost equal.

# In[24]:


# plot of sex variables

sns.countplot(x = 'sex', data = medidata)
plt.title('Distribution of sex')
plt.show


# Firstly, we are using 'sns.countplot' because 'sex' is a categorical variables and to represent in a better way. The figure shows that the number of male and female is almost equal.

# In[25]:


# distribution of bmi variable

sns.displot(medidata['bmi'])
plt.title('Distribution of BMI')
plt.show()


# This kind of distribution is normal distribution. The figure shows that we have an gradual increase from 15 to reach the peak values of 30. Then there is a gradual decrease. We may also notice very few outliers and we will take care of them later.
# 
# According to the research, Normal BMI range is 18.5 to 24.9. A person exceeding the limit is overweight and the person below this limit is underweight. We could see that there are more number of people in this dataset that are overweight!

# In[26]:


# plot of children variable

sns.countplot(x = 'children', data = medidata)
plt.title('Children')
plt.show()


# According to the figure, there are more number of people with no children. Then there are people having 1-3 children and there are very less people having 4-5 children.

# In[27]:


# plot of smoker variable

sns.countplot(x = 'smoker', data = medidata)
plt.title('Smoker')
plt.show()


# In this dataset, there are more non-smokers than smokers.

# In[28]:


# plot of region variable

sns.countplot(x = 'region', data = medidata)
plt.title('Region')
plt.show()


# We have four regions: Southwest, Southeast, Northwest, and Northeast. People are equally distributed in all the regions with southeast having slightly more number of people than other regions.

# In[29]:


# distribution of charges variable

sns.displot(medidata['charges'])
plt.title('Distribution of Charges')
plt.show()


# Mostly, the charges are around 1000-10,000 dollars.

# # 5. Data Pre-Processing

# In[30]:


# Categorical features: Sex, Smoker, and Region.

data = medidata.copy()

# Assigning values for 'smoker' feature

data['smoker'] = data['smoker'].map({'yes':1, 'no':0})

data.head()


# In[31]:


# As we know that 'sex' and 'region' are nominal categorical variables
# We will create dummy variable

dummies = pd.get_dummies(data['sex'])


# In[32]:


dummies


# In[34]:


from sklearn.preprocessing import OneHotEncoder


# In[35]:


# We will use this method to feed this feature to the machine

ohe = OneHotEncoder()

feature_array = ohe.fit_transform(data[['region']]).toarray()


# In[36]:


# Let's see the following categories in column region

feature_labels = ohe.categories_

print(feature_labels)


# In[37]:


# Creating one array

feature_labels = np.array(feature_labels).ravel()

print(feature_labels)


# In[38]:


# We are now making a data frame of these labels

features = pd.DataFrame(feature_array, columns = feature_labels)

features.head()


# In[39]:


# We will now join the dummy variable and OHE columns to original dataset

data_new = pd.concat([data, dummies, features], axis=1)

data_new = data_new.drop(columns='region', axis=1)
data_new = data_new.drop(columns='sex', axis=1)

data_new.head()


# # Splitting the Features and Targets

# In[40]:


# declare the variables

y = data_new.charges
x = data_new.drop(columns='charges', axis=1)


# In[41]:


# let's have a look at the target variables

print(y)


# In[42]:


# let's have a look at the features

print(x)


# # Splitting the data into training data & testing data

# In[43]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)


# In[44]:


# By looking at the shape, we could see the number of observations which are training and testing

print(x.shape, x_train.shape, x_test.shape)


# # 6. data profiling

# In[46]:


profile = pandas_profiling.ProfileReport(medidata)
profile


# In[ ]:




