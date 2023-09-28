import gensim.downloader as api

wv = api.load("word2vec-google-news-300")
glv = api.load("glove-twitter-25")

wv.similarity(w1="income", w2="revenue")

wv.most_similar("good")

wv.most_similar(positive=["France", "Berlin"], negative=["Paris"])

wv.doesnt_match(["facebook", "cat", "google", "microsoft", "bloomberg"])
