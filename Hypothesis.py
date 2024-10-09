#!/usr/bin/env python
# coding: utf-8

# Exploratory Data Analisys of Books.

# Libraries installation with shell commands

# In[1]:


get_ipython().run_line_magic('pip', 'install pandas numpy seaborn matplotlib')


# Libraries importation:

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Dataset Selection  
# For this Exploratory data Analisys i will be utilizing a kindle ebook dataset obtained from kaggle.  
# This particular dataset contains an array of kindle books with their respective atributes and we will be focusing on the price, reviews and stars along with other boolean type information like if it is on kindle unlimited subscription, if it is BestSeller or if it is a GoodReadersChoice.  
# This dataset was scraped on October 2023 and it was uploaded by user asaniczka and it can be accesed via the following link:  
# https://www.kaggle.com/datasets/asaniczka/amazon-kindle-books-dataset-2023-130k-books  
# 
# I chose this dataset because i am an avid reader and i find it really interesting finding the relations between the different atributes and how that impacts the book reviews and priceing.
#   
# In the following cells we will be showing the shape of the dataset along with the information about it includeing the descriptions of its atributes  

# Atributes description (These descriptions where obtained directly from the dataset source):  
#   
# asin: Product ID from Amazon.  
# title: Title of the book.  
# author: Author(s) of the book.  
# soldBy: Seller(s) of the book.  
# imgUrl: URL of the book.  
# productURL: URL to the publication on wich the ebook is sold.  
# stars: Average rating of the book. If 0, no ratings were found.  
# reviews: Number of reviews. If 0, no reviews were found.  
# price: Price of the book. If 0, price was unavailable.  
# isKindleUnlimited: Whether the book is available through Kindle Unlimited.  
# category_id: Serial id assigned to the category this book belong to.  
# isBestSeller: Whether the book had the Amazon Best Seller status or not.  
# isEditorsPick: Whether the book had the Editor's Pick status or not.  
# isGoodReadsChoice: Whether the book had the Goodreads Choice status or not.  
# publishedDate: Publication date of the book.  
# category_name: Name of the book category.

# CSV File loading  
# For working with this dataset i will be using the pandas library.  
# In the next cell it will be loaded and we can apriciate the shape of the dataset.  

# In[3]:


df = pd.read_csv('data/kindle_data-v2.csv')
df.shape


# Atributes of the dataset  
# Here we can see all the features of this dataset as described above:

# In[4]:


print(df.columns)


# Following the later here is a example of how the dataset is composed using the head() function included with pandas

# In[5]:


df.head()


# After loading the dataset into our environment i will aproach it by looking at its data types using the info() function.  
# This i think is one of the best aproaches to a dataset because i can instantly identify which of my columns have null values and the data type of each colum which will allow me to devise how to adress them.  
# We can also see how many entries (rows) and feaures (columns) we have to work with.

# In[6]:


df.info()


# As shown before the dataset contains almost non-null information wich can be deceving because as described in the atributes section in most of the numerical attributes the null data is saved as 0.  
# With that information in mind i will be utilizing the funtions value_counts() and the fucntion get() to se how many null information we have in the columns og price, reviews and stars. 

# In[7]:


print("Price null values:",df['price'].value_counts().get(0, 0))
print("Reviews null values:",df['reviews'].value_counts().get(0, 0))
print("Stars null values:",df['stars'].value_counts().get(0, 0))


# As seen on the shape of the dataset we have 133102 rows in the dataset making the 64670 missing values in reviews a concenring amount for the data clensing and processing.  
# And even with that we have several missing values in the price and stars fields that will requiere diffenrent considerations.  
# Now because we can follow several paths to deal with this missing values for startes i will be proposing my hypothesys right now based on a simple overview of the data for it to be processed in the pursuit of the analaysis to be made to prove or discard them.  

# So for now i will do several pairPlot using seaborn to try and understand how the missing data is distributed and ow it affects the dataset analysis.

# Pair plot beteween all the features:

# In[52]:


sns.pairplot(df)


# Based on the last pairplot i deduce that there are some relations betweeen the data with the missing values and other categorical features such as category.  
# Because of that i will create a pair plot of the dataset based on category_name.  

# In[53]:


sns.pairplot(df, hue = 'category_name')


# With this plot i see how the category may affect some of the other values such as price and maybe even identify some outliers wich i will deal with later.  
# As for now i will plot the average price per category to dig deeper in this relationship to decide what how to proceed with the data clensing.

# In[8]:


avg_price_per_category = df.groupby('category_name')['price'].mean().reset_index()
avg_price_per_category = avg_price_per_category.sort_values(by='price', ascending=False)

plt.figure(figsize=(10, 6))

sns.barplot(x='category_name', y='price', data=avg_price_per_category)

plt.title('Average Price per Category')
plt.xlabel('Category Name')
plt.ylabel('Average Price (in $)')
plt.xticks(rotation=45)  
plt.tight_layout()

plt.show()                           


# Based on the later i deduce that there is a relationship between the category and the price and that there could be other similar relationships with price and that they are affected too by the missing values. 
# Therefore given that the missing values in price are not a substancial amount of the dataset i decided to ignore them and i will be eliminating them from a copy of the dataset to work with.
# With this i only have to decide what to do with the other missing values in reviews and stars.

# As for stars since there is not a substancial amount of missing values and that most rated book are more likely to not be very poorly reviewed i decided to impute them with the median.

# Finaly for reviews since there is a lot of missing data and that the relationships seems minimal with the features other than stars i will ignore the whole column for the rest of this analysis.

# Actual data clensing and feature engeneering  
# Here a will create a smaller dataset without the ignored features described above and without some other columns that we wont be using like the urls.

# In[18]:


# Creation of a smaller dataset without the columns we wont be using
work_df = df.loc[:,['author','soldBy','stars','price','isKindleUnlimited','category_name','isBestSeller','isEditorsPick','isGoodReadsChoice','publishedDate']]
work_df.head()


# For easier handling of the missing values i will be replaceing all the ceroes on price and stars with pandas not a number NaN

# In[19]:


work_df['price'] = work_df['price'].replace(0, pd.NA)
work_df['stars'] = work_df['stars'].replace(0, pd.NA)
work_df.info()


# Eliminating missing values rows from price:

# In[20]:


work_df =work_df.dropna(subset=['price'])
work_df.info()


# Next i will be imputing the stars values.

# In[21]:


# Step 1: Calculate the median for non-zero stars
median_stars = work_df.loc[work_df['stars'] > 0, 'stars'].median()

# Step 2: Replace missing values (stars == 0) with the median
work_df['stars'] = work_df['stars'].replace(pd.NA, median_stars)

# Verify the changes
work_df.info()


# Following the feature engeneering i will transform the date column into three separate ones.

# In[22]:


work_df['publishedDate'] = pd.to_datetime(df['publishedDate'], format='%Y-%m-%d')
work_df['year'] = work_df['publishedDate'].dt.year
work_df['month'] = work_df['publishedDate'].dt.month
work_df['day'] = work_df['publishedDate'].dt.day

work_df.head()


# To finish with the data clensing and feature engenieering in will change all the boolean rows of the dataset to 0 and 1.
# 

# In[23]:


work_df = work_df.replace({True: 1, False: 0}).infer_objects(copy=False)
work_df.head()


# Data insights:

# In[24]:


avg_price_per_category = work_df.groupby('category_name')['price'].mean().reset_index()
avg_price_per_category = avg_price_per_category.sort_values(by='price', ascending=False)

plt.figure(figsize=(10, 6))

sns.barplot(x='category_name', y='price', data=avg_price_per_category)

plt.title('Average Price per Category')
plt.xlabel('Category Name')
plt.ylabel('Average Price (in $)')
plt.xticks(rotation=45)  
plt.tight_layout()

plt.show()      


# In[26]:


avg_price_per_stars_rating = work_df.groupby('stars')['price'].mean().reset_index()
avg_price_per_stars_rating = avg_price_per_stars_rating.sort_values(by='price', ascending=False)

plt.figure(figsize=(10, 6))

sns.barplot(x='stars', y='price', data=avg_price_per_stars_rating)

plt.title('Average Price per Stars rating')
plt.xlabel('Category Name')
plt.ylabel('Average Price (in $)')
plt.xticks(rotation=45)  
plt.tight_layout()

plt.show()   

