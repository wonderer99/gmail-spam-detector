import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("spam.csv", encoding = "latin-1")

#print(df.head())
#print(df.shape)
print(df.describe())
#print(df.columns)
print(df.isnull().sum())

df = df[["v1","v2"]]
df.columns = ['label','message']
print(df.head())

df['label'] = df['label'].map({'ham':0,'spam':1})
print(df['label'].value_counts())

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['message'])
y= df['label']

print(X.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)

print("Model trained!")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred))

result = model.predict(tfidf.transform(["i miss you babe"]))
print("Spam" if result[0] == 1 else "Not Spam")

with open("spam_model.pkl","wb") as f:
    pickle.dump(model,f)
    
with open("tfidf_vectorizer.pkl","wb") as f:
    pickle.dump(tfidf,f)
    
print("the model was saved!")