import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as ans

from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("C:/Users/BHANU PRASAD/Downloads/New folder/Datafiniti_Hotel_Reviews.csv")
print(df.shape)
print(df.info())
print(df.head())
print(df.describe(include='all')) 
print(df.isnull().sum())

# Removing unnecessary columns  (for text (senti) analysis ,you need only reviews.txt and reviews.text )

df=df[['reviews.text','reviews.rating']]
print(df)
print("after new dataset")
df = df.dropna(subset=['reviews.text', 'reviews.rating'])

print("hello i am from duplicate",df.duplicated().sum())
df = df.drop_duplicates()
print("hello iam from duplicate2",df.duplicated().sum())

# rating 4 or 5 → Positive  
# rating 3 → Neutral  
# rating 1 or 2 → Negative

def rating_to_sentiment(r):
    if r >= 4:
        return "positive"
    elif r == 3:
        return "neutral"
    else:
        return "negative"

df["sentiment"] = df["reviews.rating"].apply(rating_to_sentiment)

print("breakpoint - 1")
print(df.shape)
print(df.isnull().sum())
print(df.head())
print(df.tail()) 


