# coding=utf-8
# prepare data set
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

# read csv files
import csv
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

# Importing Gensim
import gensim
from gensim import corpora

# directory
from os import listdir
from os.path import isfile, join

no_topics = 12
no_top_words = 50

corpus = []
pos_tags = []
files = "users/"
for f in listdir(files):
    if isfile(join(files, f)):
        print(f)
        with open(join(files, f), 'r') as csv_file:
            for line in csv.reader(csv_file):
                # compile corpus
                tweet = line[3].decode('utf8')
                # print(tweet)
                pos = 0
                while pos > -1:
                    pos = max(tweet.find("http"), tweet.find("https"))
                    url = ""
                    if pos > -1:
                        count_whitespace = 0
                        for i in range(pos, len(tweet)):
                            if tweet[i] in " ":
                                count_whitespace = count_whitespace + 1
                            if count_whitespace is 3:
                                break
                            url += tweet[i]
                        tweet = tweet.replace(url, "")  # re.sub(r"\s", "", url))
                tweet = re.sub(r"pic\.twitter\.com\/\w*", "", tweet)  # remove pic urls
                tweet = re.sub(r"[^@]+@[^@]+\.[^@]+", "", tweet)  # remove email addresses
                tweet = re.sub(r"\@\w+", "", tweet)  # remove user names
                # tweet = re.sub(r"n't", " not", tweet)
                # tweet = re.sub(r"'ll", " will", tweet)
                # tweet = re.sub(r"[!\?,\"><()…’‘”“]", "", tweet)
                # tweet = re.sub(r"\s-\s", " ", tweet)
                # tweet = re.sub(r"\s\.\s|\.\s|\s\.|\.{3}", " ", tweet)
                # tweet = re.sub(r"\.$", "", tweet)
                # print(tweet)

                tags = word_tokenize(tweet)
                pos_tags.append(nltk.pos_tag(tags))

                corpus.append(tweet)

print("----------------")

print(len(corpus))

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


doc_clean = [clean(doc).split() for doc in corpus]

# Creating the term dictionary of our corpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % topic_idx
        print " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])


# no_features = 1000

print('nmf')
# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(corpus)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

display_topics(nmf, tfidf_feature_names, no_top_words)
