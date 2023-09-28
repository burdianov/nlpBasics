import pandas as pd
import numpy as np
import spacy

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

from utils import read_zipped_data

nlp = spacy.load("en_core_web_lg")

df = read_zipped_data("fake_real_data.zip", "csv")
df.head()
df.shape

df.columns = df.columns.str.lower()

df.label.value_counts()

df["label_num"] = df["label"].map({"Fake": 0, "Real": 1})

df["vector"] = df["text"].apply(lambda x: nlp(x).vector)

X_train, X_test, y_train, y_test = train_test_split(
    df.vector.values,
    df.label_num,
    test_size=0.2,
    random_state=42,
)

X_train_2D = np.stack(X_train)
X_test_2D = np.stack(X_test)

scaler = MinMaxScaler()
scale_train_embed = scaler.fit_transform(X_train_2D)
scale_test_embed = scaler.transform(X_test_2D)

# MultinomialNB
clf = MultinomialNB()
clf.fit(scale_train_embed, y_train)

y_pred = clf.predict(scale_test_embed)
print(classification_report(y_test, y_pred))

# KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")

clf.fit(X_train_2D, y_train)

y_pred = clf.predict(X_test_2D)

print(classification_report(y_test, y_pred))
