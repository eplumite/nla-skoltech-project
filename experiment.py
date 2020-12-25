from algorithm import EigenGameEstimator, OjaEstimator
from sklearn.decomposition import PCA

import datasets as data
import tools

pca = PCA(svd_solver='auto', iterated_power=1)
eigen_game = EigenGameEstimator(n_iter=1000)
oja = OjaEstimator(n_iter=1000)

# Iris dataset
X = data.load_iris_dataset()
X = tools.preprocess_dataset(X)

pca.fit(X)
eigen_game.estimate(X, V_true=pca.components_.T)
tools.create_plot(eigen_game.longest_streak, xlabel='Iterations, n', ylabel='Longest streak', 
                    title='Iris dataset', filename='eigen_game_iris_longest_streak')
oja.estimate(X, V_true=pca.components_.T)
tools.create_plot(oja.longest_streak, xlabel='Iterations, n', ylabel='Longest streak', 
                    title='Iris dataset', filename='oja_iris_longest_streak')

# Synthetic data with linear spectrum 
X = data.generate_synthetic(spectrum='linear')
X = tools.preprocess_dataset(X)

pca.fit(X)
eigen_game.estimate(X, V_true=pca.components_.T)
tools.create_plot(eigen_game.longest_streak, xlabel='Iterations, n', ylabel='Longest streak', 
                    title='Synthetic data with linear spectrum', filename='eigen_game_synth_linear_longest_streak')
oja.estimate(X, V_true=pca.components_.T)
tools.create_plot(oja.longest_streak, xlabel='Iterations, n', ylabel='Longest streak', 
                    title='Synthetic data with linear spectrum', filename='oja_synth_linear_longest_streak')

# Synthetic data with log spectrum
X = data.generate_synthetic(spectrum='log')
X = tools.preprocess_dataset(X)

pca.fit(X)
eigen_game.estimate(X, V_true=pca.components_.T)
tools.create_plot(eigen_game.longest_streak, xlabel='Iterations, n', ylabel='Longest streak', 
                    title='Synthetic data with log spectrum', filename='eigen_game_synth_log_longest_streak')
oja.estimate(X, V_true=pca.components_.T)
tools.create_plot(oja.longest_streak, xlabel='Iterations, n', ylabel='Longest streak', 
                    title='Synthetic data with log spectrum', filename='oja_synth_log_longest_streak')

# Digits dataset
X = data.load_digits_dataset()
X = tools.preprocess_dataset(X)

pca.fit(X)
eigen_game.estimate(X, V_true=pca.components_.T)
tools.create_plot(eigen_game.longest_streak, xlabel='Iterations, n', ylabel='Longest streak', 
                    title='Digits dataset', filename='eigen_game_digits_longest_streak')
oja.estimate(X, V_true=pca.components_.T)
tools.create_plot(oja.longest_streak, xlabel='Iterations, n', ylabel='Longest streak', 
                    title='Digits dataset', filename='oja_digits_longest_streak')

