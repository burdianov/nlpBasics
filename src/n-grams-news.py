import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from utils import read_zipped_data
from utils import preprocess

df = read_zipped_data("news_dataset.zip", "json")

print(df.shape)
df.head()

df.category.value_counts()

# to balance the dataset, we'll use undersampling, which is not a practical solution; we use it for the sake of brevity


min_samples = 1381

df_business = df[df.category == "BUSINESS"].sample(min_samples, random_state=42)
df_sports = df[df.category == "SPORTS"].sample(min_samples, random_state=42)
df_crime = df[df.category == "CRIME"].sample(min_samples, random_state=42)
df_science = df[df.category == "SCIENCE"].sample(min_samples, random_state=42)

df_balanced = pd.concat([df_business, df_sports, df_crime, df_science], axis=0)
df_balanced.category.value_counts()

df_balanced["category_num"] = df_balanced.category.map(
    {
        "BUSINESS": 0,
        "SPORTS": 1,
        "CRIME": 2,
        "SCIENCE": 3,
    }
)

X_train, X_test, y_train, y_test = train_test_split(
    df_balanced.text,
    df_balanced.category_num,
    test_size=0.2,
    random_state=42,
    stratify=df_balanced.category_num,
)
# the effect of stratify param:
y_train.value_counts()

# model training - 1-gram
clf = Pipeline(
    [
        ("vectorizer_bow", CountVectorizer()),
        ("multi_db", MultinomialNB()),
    ]
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

# model training - 1-gram and 2-gram
clf = Pipeline(
    [
        ("vectorizer_bow", CountVectorizer(ngram_range=(1, 2))),
        ("multi_db", MultinomialNB()),
    ]
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

# with preprocessing

df_balanced["preprocessed_txt"] = df_balanced.text.apply(preprocess)
df_balanced.head()

X_train, X_test, y_train, y_test = train_test_split(
    df_balanced.preprocessed_txt,
    df_balanced.category_num,
    test_size=0.2,
    random_state=42,
    stratify=df_balanced.category_num,
)

# 1. Create a pipeline object
clf = Pipeline(
    [
        ("vectorizer_bow", CountVectorizer(ngram_range=(1, 2))),
        ("multi_db", MultinomialNB()),
    ]
)

# 2. fit X_train and y_train
clf.fit(X_train, y_train)

# 3. get the predictions for X_test and store them in y_pred
y_pred = clf.predict(X_test)

# 4. print the classification report
print(classification_report(y_test, y_pred))
