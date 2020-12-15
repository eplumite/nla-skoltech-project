import unittest
import numpy as np
import datasets as data

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

if __name__ == '__main__':
    unittest.main()