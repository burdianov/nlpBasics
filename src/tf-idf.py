import pandas as pd

from utils import read_zipped_data

df = read_zipped_data("ecommerce_data.zip", "csv")

df.head()
df.label.value_counts()
