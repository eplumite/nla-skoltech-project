from sklearn import preprocessing

def preprocess_dataset(X):
    return preprocessing.scale(X)