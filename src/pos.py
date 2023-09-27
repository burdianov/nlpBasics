import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("Elon flew to mars yesterday. He carried pizza with him")

for token in doc:
    print(token, "\t|", token.pos_)


count = doc.count_by(spacy.attrs.POS)
doc.vocab[96].text

for k, v in count.items():
    print(doc.vocab[k].text, "\t|", v)
