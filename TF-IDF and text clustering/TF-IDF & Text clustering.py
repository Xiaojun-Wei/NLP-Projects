import re
import math

import nltk.tokenize
import pandas as pd

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


""" Compute TF—IDF"""


corpus1 = [
    'this is the first document',
    'this is the second second document', 'and the third one',
    'is this the first document',
]

"""方法一：调用sklearn包"""
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(corpus1)

print("Bag of words: ", tfidf_vec.get_feature_names_out())
print("Word frequency: ", tfidf_vec.vocabulary_)
print("\n")
print("Solution1 for question 2: \n", tfidf_matrix.toarray())
print("\n")

"""方法二： 分布计算"""
# 分词
word_list1 = []
for i in range(len(corpus1)):
    word_list1.append(corpus1[i].split(' '))

# 统计词频
count_list1 = []
for i in range(len(word_list1)):
    count = dict(Counter(word_list1[i]))
    print(count)
    count_list1.append(count)
print("\n")

# 计算tf, tf = 词频 / 该文件单词总数
def compute_tf(word, count):
    return count[word] / sum(count.values())


# 计算idf，idf = log(语料库文件数 / 包含该词文件数+1)
def compute_idf(word, count_list1):
    doc_num = sum(1 for count in count_list1 if word in count)
    return math.log((len(count_list1) / (doc_num + 1)))


# 计算tf-idf, tf-idf = tf * idf
def tf_idf(word, count, count_list):
    return compute_tf(word, count) * compute_idf(word, count_list)


print("Solution2 for question 2: ")
for i, count in enumerate(count_list1):
    print('\n')
    print(f"tf-idf of words in document {i+1}")
    tfidf_dict = {word: tf_idf(word, count, count_list1) for word in count}
    # 降序排列
    sorted_list = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)
    for word, tfidf in sorted_list:
        print(word, ' ', tfidf)

print("-" * 50)

"""------------------------------------------------------------------"""


"""Text Clustering"""

text = open("corpus.txt", 'r', encoding="utf-8").read()

# 分词
stemmer = PorterStemmer()
stop_words = stopwords.words("english")
text = re.sub(r"it's", 'it is', text)  # 拓展缩写词
text = re.sub(r"\w\.(?!\n$)", '', text)  # 删除名字缩写
text = text.lower()  # 转换成小写
pattern = "[a-zA-Z]+"
tokens = nltk.tokenize.regexp_tokenize(text, pattern)

tfidf = TfidfVectorizer(stop_words=stop_words)  # 删去停用词
X = pd.DataFrame(tfidf.fit_transform(tokens).toarray(),  # 创建dataframe
                 index=tokens, columns=tfidf.get_feature_names_out())

km = KMeans(n_clusters=3)
y_predited = km.fit_predict(X)
print("Solution for question 3: ", y_predited)