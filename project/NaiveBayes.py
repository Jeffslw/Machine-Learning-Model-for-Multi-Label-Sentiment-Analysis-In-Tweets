#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the original dataset
data = pd.read_csv('redditinput.csv')
X = data['text']
y = data['emotion']

# Preprocess the data using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)

# Evaluate the model on the test set
y_test_pred = nb_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification report:\n", classification_report(y_test, y_test_pred))

# Load the new dataset
new_data = pd.read_csv('tweetspredit.csv')
X_new = new_data['text']

# Preprocess the new text data using the same TfidfVectorizer
X_new_vectorized = vectorizer.transform(X_new)

# Make predictions on the new dataset
y_new_pred = nb_clf.predict(X_new_vectorized)

# Print the predicted emotions for the new dataset
#print("Predicted emotions:", y_new_pred)

# Load the true labels (emotions) for the new dataset
y_new_true = new_data['emotion']

# Calculate the accuracy
accuracy = accuracy_score(y_new_true, y_new_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate a classification report
report = classification_report(y_new_true, y_new_pred)
print("Classification report:\n", report)


# In[ ]:




