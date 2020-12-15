import numpy as np
from datasets import load_iris_dataset
from tools import create_plot

class BaseEstimator():
    def __init__(self, k=None):
        self.k=None

    def _iterate():
        pass

    def _measure_angle():
        pass

    def _measure_subspace_distance():
        pass

    def estimate():
        pass

class EigenGameEstimator(BaseEstimator):
    def __init__(self):
        BaseEstimator.__init__(self)

    def grad(self, V, v0, X):
        M = X.T@X
        if V is None:
            return 2*M.dot(v0)
        else:
            # du = v0.reshape(-1)
            du = np.zeros(V.shape[0])
            columns = V.shape[1]
            for j in range(columns):
            # du -= V[:,j]*np.dot(v0.reshape(-1), np.dot(M,V[:,j]))/np.dot(V[:,j].reshape(-1), np.dot(M,V[:,j]))
                du -= V[:,j]*np.dot(v0, np.dot(M,V[:,j]))
            du /= np.dot(V[:,j], np.dot(M,V[:,j]))
            du += v0
            du = 2*M.dot(du)
            return du

    def eigen_game_partial(self, X, v0=None, V=None, tol=1e-4, alpha=1e-3):
        v = v0
        dim = X.shape[1]
        # key = jax.random.PRNGKey(0)
        # v = jax.random.normal(key, (dim,))
        v = np.random.random((dim,))
        v /= np.linalg.norm(v)
        du = self.grad(V, v, X)
        t_max = np.ceil((5/4)*min(np.linalg.norm(du)*0.5,tol)**(-2)).astype(int)
        # for t in range(min(t_max, 100)):
        # print(t_max)
        convergence = []
        t_max = min(t_max, 10)
        for t in range(t_max):
            du = self.grad(V, v, X)
            du = du - np.dot(du, v)*v
            v_new = v + alpha*du
            v_new /= np.linalg.norm(v_new)
            convergence.append(np.linalg.norm(v-v_new))
            v = v_new
        return v.reshape(-1,1), convergence

    def eigen_game_sequential(self, X, tol=1e-4, alpha=1e-3, k=None):
        dim = X.shape[1]
        if k:
            dim = min(dim, k)
        V, conv = self.eigen_game_partial(X, v0=None, V=None, tol=1e-4, alpha=1e-3)
        convergence = []
        convergence.append(conv)
        for i in range(1, dim):
            component, conv = self.eigen_game_partial(X=X, v0=None, V=V, tol=tol, alpha=alpha)
            convergence.append(conv)
            V = np.hstack((V, component))
        return V, convergence

X = load_iris_dataset()
eigen_game = EigenGameEstimator()
V, conv = eigen_game.eigen_game_sequential(X)
