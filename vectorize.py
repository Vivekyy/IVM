from utils import getXy

from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

import spacy

def splitData():
    X,y = getXy()
    X,y = shuffle(X, y)

    X_train = X.iloc[:int(.8*len(X))]
    X_test = X.iloc[int(.8*len(X)):]

    y_train = y.iloc[:int(.8*len(y))]
    y_test = y.iloc[int(.8*len(y)):]

    return X_train, X_test, y_train, y_test

def vectorize(X_train, X_test, vocab_size=800, model_type = 'TFIDF'):
    if model_type.upper() == "BOW":
        vectorizer = CountVectorizer(analyzer=tokenize, max_features=vocab_size) #Tokenize is callable, so used to process the raw input
    elif model_type.upper() == "TFIDF":
        vectorizer = TfidfVectorizer(analyzer=tokenize, max_features=vocab_size)
    
    train_vecs = vectorizer.fit_transform(X_train)
    test_vecs = vectorizer.transform(X_test)

    return train_vecs, test_vecs

def tokenize(line):
    tokenizer = spacy.load("en_core_web_sm")
    clean_tokens = []
    for token in tokenizer(line):
        if (not token.is_stop) & (token.lemma_ != '-PRON-') & (not token.is_punct):
            clean_tokens.append(token.lemma_)
    return clean_tokens

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = splitData()
    train_vecs, test_vecs = vectorize(X_train, X_test, model_type = 'TFIDF')

    print(test_vecs)