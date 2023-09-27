import nltk
from nltk.stem import PorterStemmer
import spacy

stemmer = PorterStemmer()

words = [
    "eating",
    "eats",
    "eat",
    "ate",
    "adjustable",
    "rafting",
    "ability",
    "meeting",
    "better",
]

for word in words:
    print(word, "\t|", stemmer.stem(word))

nlp = spacy.load("en_core_web_sm")

print("-------------------------")

doc = nlp("eating eats eat ate adjustable rafting ability meeting better")
for token in doc:
    print(token, "\t|", token.lemma_, "\t|", token.lemma)

nlp.pipe_names

ar = nlp.get_pipe("attribute_ruler")
ar.add([[{"TEXT": "Bro"}], [{"TEXT": "Brah"}]], {"LEMMA": "brother"})

doc = nlp("Bro, you wanna go? Brah, don't say no! I am exhausted")
for token in doc:
    print(token.text, "\t|", token.lemma_)
