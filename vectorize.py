from utils import getXy

import pandas as pd

#Cuml can replace sci-kit learn for potential speedup
#Not using cuml due to buggy import issues
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import spacy

def splitData(split=.8, dataset_path='IntegratedValueModelrawdata.xlsx'):
    X,y,y_map = getXy(split, dataset_path)

    if split=='debug': #for quick debugging
        X_train = X.iloc[:int(.05*len(X))]
        X_test = X.iloc[:int(.05*len(X))]

        y_train = y.iloc[:int(.05*len(y))]
        y_test = y.iloc[:int(.05*len(y))]
    else:
        X_train = X.iloc[:int(split*len(X))]
        X_test = X.iloc[int(split*len(X)):]

        y_train = y.iloc[:int(split*len(y))]
        y_test = y.iloc[int(split*len(y)):]

    return X_train, X_test, y_train, y_test, y_map

def vectorize(X_train, X_test, vocab_size=800, model_type = 'BOW'):
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
    train_vecs, test_vecs = vectorize(X_train, X_test, model_type = 'BOW')

    print(test_vecs)