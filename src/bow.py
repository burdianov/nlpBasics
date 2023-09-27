import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

df = pd.read_csv("../data/spam.csv")
df.head()

df.columns = df.columns.str.lower()

df.category.value_counts()

df["spam"] = df["category"].apply(lambda x: 1 if x == "spam" else 0)

X_train, X_test, y_train, y_test = train_test_split(
    df.message, df.spam, test_size=0.2, random_state=42
)

X_train.shape
X_test.shape

type(X_train)
X_train[:4]
X_train[1978]

type(y_train)
y_train[:4]

v = CountVectorizer()

X_train_cv = v.fit_transform(X_train.values)
X_train_cv.toarray()
X_train_cv.shape

v.get_feature_names_out()[1000:1050]
v.get_feature_names_out().shape

dir(v)

v.vocabulary_
v.get_feature_names_out()[1335]

X_train_np = X_train_cv.toarray()
X_train_np[:4]
X_train_np[:4][0]

np.where(X_train_np[0] != 0)

X_train[:4]
X_train[:4][1978]
X_train_np[0][6888]
v.get_feature_names_out()[6888]

model = MultinomialNB()

model.fit(X_train_cv, y_train)

X_test_cv = v.transform(X_test)

y_pred = model.predict(X_test_cv)

print(classification_report(y_test, y_pred))

emails = [
    "Hey mohan, can we get together to watch the football game tomorrow?",
    "Up to 20 percent discount on parking, exclusive offer just for you. Don't miss the reward!",
]

emails_count = v.transform(emails)
model.predict(emails_count)

# Now the same as all the above using a pipeline

clf = Pipeline(
    [
        ("vectorizer", CountVectorizer()),
        ("nb", MultinomialNB()),
    ]
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
