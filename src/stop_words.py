import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from utils import read_zipped_data

print(STOP_WORDS)

nlp = spacy.load("en_core_web_sm")


def preprocess(text: str):
    doc = nlp(text)
    no_stop_words = [
        token.text for token in doc if not token.is_stop and not token.is_punct
    ]
    return " ".join(no_stop_words)


preprocess("We just opened our wings, thy flying part is coming soon.")

df = read_zipped_data("doj_press.zip", "json")

# df = pd.read_json("../data_ex/doj_press.json", lines=True)
df.shape
df.head()

df = df[df["topics"].str.len() != 0]
df.head()
df.shape

df = df.head(100)
df.shape

len(df["contents"].iloc[4])

df["contents_new"] = df["contents"].apply(preprocess)
df.head(5)

len(df["contents_new"].iloc[4])

df["contents"].iloc[4][:300]
df["contents_new"].iloc[4][:300]
