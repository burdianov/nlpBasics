import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from utils import read_zipped_data
from utils import preprocess

print(STOP_WORDS)

nlp = spacy.load("en_core_web_sm")

preprocess("We just opened our wings, thy flying part is coming soon.")

df = read_zipped_data("doj_press.zip", "json")

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
