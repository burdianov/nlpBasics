import spacy

nlp = spacy.blank("en")

doc = nlp("Dr. Jones loves pizza. In Dubai it costs only 2$ per plate.")

for token in doc:
    print(token)

doc = nlp("Let's go to N.Y.")
for token in doc:
    print(token)

with open("../data/students.txt") as f:
    text = f.readlines()

text = " ".join(text)

doc = nlp(text)

emails = []
for token in doc:
    if token.like_email:
        emails.append(token.text)
