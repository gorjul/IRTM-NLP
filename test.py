from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

ps = PorterStemmer()

example_words = ["watch","watches","reality","realities","essence", "essences", "immigration"]

for w in example_words:
    print(ps.stem(w))

for w in example_words:
    print(wordnet_lemmatizer.lemmatize(w, pos='v'))

new_text = "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."

words = word_tokenize(new_text)

for w in words:
    print(ps.stem(w))
    
for w in words:
    print(wordnet_lemmatizer.lemmatize(w, pos='v'))
