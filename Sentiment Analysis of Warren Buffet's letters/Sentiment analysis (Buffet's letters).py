import string
import matplotlib.pyplot as plt

from collections import Counter
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# reading text file
text = open("letters 2001-2010 utf-8.txt", 'r', encoding="utf-8").read()

# converting to lowercase
lower_case = text.lower()

# Removing punctuations
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

# splitting text into words
tokens = word_tokenize(cleaned_text, "english")

# Removing stop words from the tokenized words list
tokens_no_sw = [word for word in tokens if not word in stopwords.words("english")]

# Next Emotion Algorithm:
# Check if the word in the final word list is also present in emotion.txt

# lemmatization
lemma_words = []
for word in tokens_no_sw:
    word = WordNetLemmatizer().lemmatize(word)
    lemma_words.append(word)


emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')

        if word in lemma_words:
            emotion_list.append(emotion)


# This analyser request the whole text
def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    if score['neg'] > score['pos']:
        print("Negative Sentiment")
    elif score['neg'] < score['pos']:
        print("Positive Sentiment")
    else:
        print("Neutral Sentiment")


sentiment_analyse(cleaned_text)


# counting emotions
w = Counter(emotion_list)

# Plotting the emotions on the graph
fig, ax = plt.subplots()  # minimize graph
ax.bar(w.keys(), w.values())
fig.autofmt_xdate(rotation=50)  # adjust the axis
plt.title(label="Emotions")
plt.savefig('Emotions.png')
plt.show()

