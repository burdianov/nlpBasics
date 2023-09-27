import spacy
from spacy import displacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")

nlp.pipe_names

doc = nlp("Tesla Inc is going to acquire twitter for $45 billion")

for ent in doc.ents:
    print(ent.text, "\t|", ent.label_, "\t|", spacy.explain(ent.label_))

displacy.render(doc, style="ent")

nlp.pipe_labels["ner"]

doc = nlp("Michael Bloomberg founded Bloomberg in 1982")

for ent in doc.ents:
    print(ent.text, "\t|", ent.label_, "\t|", spacy.explain(ent.label_))

doc = nlp("Tesla is going to acquire twitter for $45 billion")
for ent in doc.ents:
    print(ent.text, "|", ent.label_)

type(doc[2:5])

s1 = Span(doc, 0, 1, label="ORG")
s2 = Span(doc, 5, 6, label="ORG")

doc.set_ents([s1, s2], default="unmodified")
for ent in doc.ents:
    print(ent.text, "|", ent.label_)
