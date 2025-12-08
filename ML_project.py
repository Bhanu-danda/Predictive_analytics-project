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

