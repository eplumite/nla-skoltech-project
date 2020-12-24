import unittest
import numpy as np
import datasets as data
import tools

class SyntheticGenerator(unittest.TestCase):

    def test_linear_spectrum(self):
        M = data.generate_synthetic(spectrum='linear')
        self.assertEqual(M.shape, (50,50))
        self.assertTrue(isinstance(M, np.ndarray))

    def test_log_spectrum(self):
        M = data.generate_synthetic(spectrum='log')
        self.assertEqual(M.shape, (50,50))
        self.assertTrue(isinstance(M, np.ndarray))

class DatasetsLoader(unittest.TestCase):

    def test_iris_loader(self):
        X = data.load_iris_dataset()
        self.assertEqual(X.shape, (150,4))
        self.assertTrue(isinstance(X, np.ndarray))

    def test_digits_loader(self):
        X = data.load_digits_dataset()
        self.assertEqual(X.shape, (1797, 64))
        self.assertTrue(isinstance(X, np.ndarray))

class DatasetsPreprocessing(unittest.TestCase):

    def test_standardization(self):
        dim = 20
        X = np.random.random((dim,dim)) 
        X = tools.preprocess_dataset(X)
        self.assertAlmostEqual(X.mean(axis=0).sum(), 0.0, 12)
        self.assertAlmostEqual(X.std(axis=0).sum(), dim, 12)

if __name__ == '__main__':
    unittest.main()