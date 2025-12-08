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

#Clean the Text (Lowercase, remove symbols)

import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['clean_text'] = df['reviews.text'].apply(clean_text)

print("breakpoint -2")
print(df)

# 🌟 What is NLTK? (Super Simple Explanation)
# NLTK = Natural Language Toolkit
# It is a Python library that helps you work with text data.
#👉 A toolbox for cleaning, processing, and understanding text


#🔥 STEP 6: Remove Stopwords + Lemmatize (NLTK)



import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('wordnet')

stop = set(stopwords.words('english'))
lemm = WordNetLemmatizer()

def process_text(text):
    words = text.split()
    words = [lemm.lemmatize(w) for w in words if w not in stop]
    return " ".join(words)

print("breakpoint - 3")
df['clean_text'] = df['clean_text'].apply(process_text)    
print(df)

# Train - test split

from sklearn.model_selection import train_test_split
X = df['clean_text']
Y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

#Convert Text → Numbers (TF-IDF Vectorizer)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# Step-6: Train ML Model (Start with Logistic Regression)
 
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# evalute accuracy
from sklearn.metrics import accuracy_score, classification_report
pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))


print("breakpoint-4")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

pred_nb = nb.predict(X_test_tfidf)

print("Naive Bayes Accuracy:", accuracy_score(y_test, pred_nb))
print("\nClassification Report:\n", classification_report(y_test, pred_nb))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred_nb))


print("breakpoint - 5")
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)

pred_svm = svm.predict(X_test_tfidf)

print("SVM Accuracy:", accuracy_score(y_test, pred_svm))
print("\nClassification Report:\n", classification_report(y_test, pred_svm))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred_svm))





















































