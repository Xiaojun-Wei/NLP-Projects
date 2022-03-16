from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup",
]

vectorizer = TfidfVectorizer()
tf_idf = vectorizer.fit_transform(docs)
# 打印词和它对应的idf值
print("idf: ", [(n,idf) for idf, n in zip(vectorizer.idf_, vectorizer.get_feature_names_out())])
# 打印词的索引
print("v2i: ", vectorizer.vocabulary_)

query = "I get a coffee cup"
qtf_idf = vectorizer.transform([query])
res = cosine_similarity(tf_idf, qtf_idf)
res = res.ravel().argsort()[-3:]
print("\ntop 3 docs for '{}':\n{}".format(query, [docs[i] for i in res[::-1]]),
      "tf_idf_sklearn_matrix")