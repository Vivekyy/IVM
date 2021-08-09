from vectorize import splitData, vectorize

def train(model_type, num_epochs):
    X_train, X_test, y_train, y_test = splitData()
    train_vecs, test_vecs = vectorize(X_train, X_test, model_type = model_type)