from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from utils import read_zipped_data
from utils import preprocess

df = read_zipped_data("ecommerce_data.zip", "csv")

df.shape
df.head()
df.label.value_counts()

df["label_num"] = df.label.map(
    {
        "Books": 0,
        "Clothing & Accessories": 1,
        "Electronics": 2,
        "Household": 3,
    }
)
df.head()

df.columns = df.columns.str.lower()

X_train, X_test, y_train, y_test = train_test_split(
    df.text,
    df.label_num,
    test_size=0.2,
    random_state=42,
    stratify=df.label_num,
)
y_train.value_counts()
y_test.value_counts()


# using KNeighborsClassifier
clf = Pipeline(
    [
        ("vectorizer_tfidf", TfidfVectorizer()),
        ("KNN", KNeighborsClassifier()),
    ]
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

X_test[:5]
y_test[:5]
y_pred[:5]

# using MultinomialNB
clf = Pipeline(
    [
        ("vectorizer_tfidf", TfidfVectorizer()),
        ("MultinomialDB", MultinomialNB()),
    ]
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

# using RandomForestClassifier
clf = Pipeline(
    [
        ("vectorizer_tfidf", TfidfVectorizer()),
        ("RandomForestClassifier", RandomForestClassifier()),
    ]
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

# using the preprocessed text

df["preprocessed_text"] = df["text"].apply(preprocess)

df.text[0]
df.preprocessed_text[0]

X_train, X_test, y_train, y_test = train_test_split(
    df.preprocessed_text,
    df.label_num,
    test_size=0.2,
    random_state=42,
    stratify=df.label_num,
)

clf = Pipeline(
    [
        ("vectorizer_tfidf", TfidfVectorizer()),
        ("RandomForestClassifier", RandomForestClassifier()),
    ]
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
