import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

nltk.download('stopwords')

print("Loading data...")
data = pd.read_csv("complaints.csv")

stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stopwords.words('english')]
    return " ".join(words)

print("Cleaning text...")
data['cleaned'] = data['Complaint'].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['cleaned'])
y = data['Category']

print("Training model...")
model = MultinomialNB()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")
