import re
import nltk
nltk.download(['punkt','stopwords','wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

def tokenize(text):
    ''' Natural Language Processing: Normalize, Tokenize, Stem/Lemmatize
    - Normalize (lowercase, punctuation)
    - Tokenize words
    - Remove stop words (a, the ...)
    - Lemmatize
    '''
    # Convert text to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)    
    # Tokenize words
    tokens = word_tokenize(text)    
    # Remove Stop words, Stem & Lemmed words
    stop_word = stopwords.words("english")
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()    
    #stemmed = [stemmer.stem(w) for w in tokens if w not in stop_word]
    lemmed = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_word]
    return lemmed 