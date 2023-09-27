import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Thor eating pizza, Loki is eating pizza, Ironman ate pizza already",
    "Apple is announcing new iphone tomorrow",
    "Tesla is announcing new model-3 tomorrow",
    "Google is announcing new pixel-6 tomorrow",
    "Microsoft is announcing new surface tomorrow",
    "Amazon is announcing new eco-dot tomorrow",
    "I am eating biryani and you are eating grapes",
]

v = TfidfVectorizer()
transformed_output = v.fit_transform(corpus)

print(v.vocabulary_)
dir(v)

all_feature_names = v.get_feature_names_out()

for word in all_feature_names:
    index = v.vocabulary_.get(word)
    print(f"{word}: {v.idf_[index]}")

corpus[:2]

transformed_output.toarray()[:2]

trans_df = pd.DataFrame(transformed_output.toarray(), columns=v.get_feature_names_out())
