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

no_topics = 20
no_top_words = 10

corpus = []
usernames = []
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
                tweet = re.sub(r"[!\?,\"><()…’‘”“]", "", tweet)
                tweet = re.sub(r"\s-\s", " ", tweet)
                tweet = re.sub(r"\s\.\s|\.\s|\s\.|\.{3}", " ", tweet)
                tweet = re.sub(r"\.$", "", tweet)
                # print(tweet)

                tags = word_tokenize(tweet)
                pos_tags.append(nltk.pos_tag(tags))

                corpus.append(tweet)
                usernames.append(line[2])

print("----------------")

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def frequency_user(seq, idfun=None):
    # order preserving
    if idfun is None:
        def idfun(x): return x
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in result:
            result[marker] += 1
            continue
        result[marker] = 1
    return result


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

print(dictionary[:5])
print(frequency_user(usernames)[:5])

exit(1)

print('gensim 1')
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

print('gensim 2')
# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=no_topics, id2word=dictionary, passes=50)
print(ldamodel.print_topics(num_topics=no_topics, num_words=no_top_words))
print("----------------")


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % topic_idx
        print " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])


no_features = 1000

print('sklearn 1')
# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(corpus)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

print('sklearn 2')
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50., random_state=0).fit(tf)

display_topics(nmf, tfidf_feature_names, no_top_words)
print("----------------")
display_topics(lda, tf_feature_names, no_top_words)
