from utils import getXy
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy

def splitData():
    X,y = getXy()
    X,y = shuffle(X, y)

    X_train = X[0:.8*len(X)]
    X_test = X[.8*len(X), len(X)]

    y_train = y[0:.8*len(y)]
    y_test = y[.8*len(y), len(y)]

    return X_train, X_test, y_train, y_test

def vectorize(X_train, X_test, model_type = 'TFIDF'):
    if model_type.upper() == "BOW":
        vectorizer = CountVectorizer(analyzer=tokenize) #Callable, so tokenize is used to process the raw input
    elif model_type.upper() == "TFIDF":
        vectorizer = TfidfVectorizer(analyzer=tokenize)
    
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