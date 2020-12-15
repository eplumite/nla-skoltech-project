import unittest
import numpy as np
from datasets import generate_synthetic

class SyntheticGenerator(unittest.TestCase):

    def test_linear_spectrum(self):
        M = generate_synthetic(spectrum='linear')
        self.assertEqual(M.shape, (50,50))
        self.assertTrue(isinstance(M, np.ndarray))

    def test_log_spectrum(self):
        M = generate_synthetic(spectrum='log')
        self.assertEqual(M.shape, (50,50))
        self.assertTrue(isinstance(M, np.ndarray))

if __name__ == '__main__':
    unittest.main()