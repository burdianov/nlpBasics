import spacy
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_lg")

doc = nlp("dog cat banana afskfsd")

for token in doc:
    print(f"{token.text} -> Vector: {token.has_vector}; OOV: {token.is_oov}")

doc[0].vector.shape

base_token = nlp("bread")
base_token.vector.shape

doc = nlp("bread sandwich burger car tiger human wheat")

for token in doc:
    print(f"{token.text} <-> {base_token.text}: {token.similarity(base_token)}")


def print_similarity(base_word: str, words_to_compare: str):
    base_token = nlp(base_word)
    doc = nlp(words_to_compare)

    for token in doc:
        print(f"{token.text} <-> {base_token.text}: {token.similarity(base_token)}")


print_similarity("iphone", "apple samsung iphone dog kitten")

king = nlp.vocab["king"].vector
man = nlp.vocab["man"].vector
woman = nlp.vocab["woman"].vector
queen = nlp.vocab["queen"].vector

result = king - man + woman

cosine_similarity([result], [queen])
