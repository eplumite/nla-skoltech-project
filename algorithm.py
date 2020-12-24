import numpy as np
from functools import reduce


class BaseEstimator():
    def __init__(self, n_components=None, n_iter=100, random_state=None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state

    def _measure_angular_error(self, v, v_true):
        scalar_product = np.dot(v, v_true)
        scalar_product /= np.linalg.norm(v)
        scalar_product /= np.linalg.norm(v_true)
        # scalar_product = np.round(scalar_product, 12)
        # if np.abs(scalar_product) > 1.0:
        #     raise ValueError(f"Incorrect scalar product: {scalar_product}")
        sinus = np.abs(1 - scalar_product ** 2) ** 0.5
        return np.arcsin(sinus)

    def _measure_subspace_distance(self, V, V_true):
        try:
            U_star = V.dot(np.linalg.pinv(V))
            P = V_true.dot(np.linalg.pinv(V_true))
        except:
            raise Exception
        return 1 - np.trace(U_star.dot(P))/V.shape[1]

    def _find_longest_strike(self, strikes):
        strikes_to_string = ''.join(strikes.astype(str))
        return reduce(max, map(lambda x: len(x), strikes_to_string.split('0')))

    def _estimate(self, X, V_true):
        pass

    def estimate(self, X, V_true=None):
        self._estimate(X, V_true)
        return self


class EigenGameEstimator(BaseEstimator):
    def __init__(self, n_components=None, n_iter=10, tol=1e-4, alpha=1e-3):
        BaseEstimator.__init__(self, n_components, n_iter)
        self.tol = tol
        self.alpha = alpha

    def _grad(self, V, v, X):
        M = X.T@X
        if V is None:
            return 2*M.dot(v)
        else:
            du = v
            columns = V.shape[1]
            for j in range(columns):
              du -= V[:,j]*np.dot(v, np.dot(M,V[:,j]))/np.dot(V[:,j], np.dot(M,V[:,j]))
            du = 2*M.dot(du)
            return du

    def _estimate_next_component(self, X, k):

        n = X.shape[1]
        v = np.random.random((n,))
        v /= np.linalg.norm(v)
        self.V[:,k] = v

        du = self._grad(self.V[:,:k], v, X)
        t_max = np.ceil((5/4)*min(np.linalg.norm(du)*0.5,self.tol)**(-2)).astype(int)

        if k == 0:
            self.convergence = np.zeros((n, self.n_iter))
            self.subspace_distance = np.zeros((n, self.n_iter))
            self._streak = np.zeros((n, self.n_iter), dtype=np.int)

        t_max = min(t_max, self.n_iter)

        for t in range(t_max):
            du = self._grad(self.V[:,:k], v, X)
            du = du - np.dot(du, v)*v
            v_new = v + self.alpha*du
            v_new /= np.linalg.norm(v_new)
            self.V[:,k] = v_new

            self.convergence[k, t] = np.linalg.norm(v-v_new)
            v = v_new

            if self.V_true is not None:
                self.subspace_distance[k, t] = self._measure_subspace_distance(self.V_true[:, :k+1], self.V[:,:k+1])

            if self._measure_angular_error(v, self.V_true[:,k]) < np.pi/8:
                self._streak[k, t] = 1

    def _estimate(self, X, V_true=None):
        # Handle number of components to obtain
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = min(X.shape[1], self.n_components)

        self.V = np.zeros((X.shape[1], n_components))
        self.V_true = V_true

        for k in range(n_components):
            self._estimate_next_component(X, k)

        self.longest_streak = np.apply_along_axis(self._find_longest_strike, 0, self._streak)


class OjaEstimator(BaseEstimator):
    def __init__(self, n_components=None, n_iter=10, lr=1e-1):
        BaseEstimator.__init__(self, n_components, n_iter)
        self.lr = lr

    def _estimate(self, X, V_true):
        # Handle number of components to obtain
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = min(X.shape[1], self.n_components)

        self.subspace_distance = np.zeros(self.n_iter)
        self.convergence = np.zeros((X.shape[1], self.n_iter))
        self._streak = np.zeros((X.shape[1], self.n_iter), dtype=np.int)

        #generate initial matrix  
        self.V = np.random.normal(size=(X.shape[1], X.shape[1]))
        #normirating
        for k in range(self.V.shape[1]):
            self.V[:,k] = self.V[:,k]/np.linalg.norm(self.V[:,k])

        #oja algorithm
        for t in range(self.n_iter):
            #oja step
            V_old = self.V.copy()
            self.V = self.V +  (X.T).dot(X.dot(self.V)) * self.lr / np.sqrt(t+1)
            #orthogonalization
            Q, R = np.linalg.qr(self.V)
            S = np.diag(np.sign( np.sign( np.diagonal(R) ) + 0.5 ) )
            self.V = Q.dot(S)
            #normirating
            for k in range(self.V.shape[1]):
                self.V[:,k] = self.V[:,k]/np.linalg.norm(self.V[:,k])
                self._streak[k,t] = self._measure_angular_error(self.V[:,k], V_true[:,k])
                self.convergence[k,t] = np.linalg.norm(self.V[:,k] - V_old[:,k])

            if V_true is not None:
                self.subspace_distance[t] = self._measure_subspace_distance(self.V[:, :n_components], V_true[:, :n_components])

        self.longest_streak = np.apply_along_axis(self._find_longest_strike, 0, self._streak[:n_components,:])
        self.convergence = self.convergence[:n_components,:]
        self.V = self.V[:, :n_components]

