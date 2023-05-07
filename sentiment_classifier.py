import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import streamlit as st

def train_model():
    reviews=pd.read_csv("reviews.csv")
    reviews=reviews.rename(columns={"text":"review"},inplace=False)

    X= reviews.review
    y=reviews.polarity
    X_train, X_test, y_train,y_test = train_test_split(X,y,train_size=0.6,random_state=1 )

    vector = CountVectorizer(stop_words = "english",lowercase= False)
    vector.fit(X_train)
    X_transormed=vector.transform(X_train)
    X_test_transformed= vector.transform(X_test)

    nb=MultinomialNB()
    nb.fit(X_transormed,y_train)

    s=pickle.dumps(nb)
    model = pickle.loads(s)
    return model, vector

model,vector=train_model()
def predict(input):
    vec= vector.transform([input]).toarray()
    print(vec)
    category=str(list(model.predict(vec))[0]).replace('0', 'NEGATIVE').replace('1', 'POSITIVE')
    return category

st.header("Review Classifier")
input=st.text_area("Enter the review")
if st.button("Classify"):
    st.write(predict(input))




