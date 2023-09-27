import spacy
from spacy import displacy

nlp = spacy.blank("en")

doc = nlp("Captain America ate 100$ of orange. Then he said I can do this all day.")

for token in doc:
    print(token)

nlp.pipe_names

nlp = spacy.load("en_core_web_sm")

nlp.pipe_names

doc = nlp("Captain America ate 100$ of orange. Then he said I can do this all day.")

for token in doc:
    print(token, " \t| ", token.pos_, " \t| ", token.lemma_)

doc = nlp("Tesla Inc is going to acquire twitter for $45 billion")

for ent in doc.ents:
    print(ent.text, "\t|", ent.label_)

displacy.render(doc, style="ent")

source_nlp = spacy.load("en_core_web_sm")

nlp = spacy.blank("en")
nlp.pipe_names
nlp.add_pipe("ner", source=source_nlp)
nlp.pipe_names
