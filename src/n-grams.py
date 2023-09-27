from sklearn.feature_extraction.text import CountVectorizer
import spacy

nlp = spacy.load("en_core_web_sm")

v = CountVectorizer(ngram_range=(1, 3))
v.fit(["Thor Hathodawala is looking for a job"])
v.vocabulary_

corpus = [
    "Thor ate pizza",
    "Loki is tall",
    "Loki is eating pizza",
]


def preprocess(text: str):
    doc = nlp(text)

    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    return " ".join(filtered_tokens)


filtered_tokens = preprocess("Thor ate pizza")

corpus_processed = [preprocess(text) for text in corpus]

v = CountVectorizer(ngram_range=(1, 2))
v.fit(corpus_processed)
v.vocabulary_

v.transform([preprocess("Thor ate pizza")]).toarray()
v.transform([preprocess("Hulk ate pizza")]).toarray()
