import numpy as np
from sklearn import datasets
from scipy.stats import ortho_group

def generate_synthetic(spectrum='linear'):
    v = ortho_group.rvs(dim=50)
    if spectrum == 'linear':
        d_elems = np.linspace(1000,1,50)
    elif spectrum == 'log':
        d_elems = np.logspace(3,0,50)
    else:
        raise Exception(spectrum)
    Lambda = np.diag(d_elems)
    M = v.dot(Lambda).dot(np.linalg.inv(v))
    return M

def load_iris_dataset():
    iris = datasets.load_iris()
    X = iris['data']
    return X