#!/usr/bin/env python3
"""LightGBM tests."""

import unittest
import subprocess
import sys

class TestLightGBMModel(unittest.TestCase):
    def test_lgbm_iris(self):
        """LGBM iris."""
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\niris\nlgbm\ny\n', 
                              text=True, timeout=60, cwd='..', 
                              capture_output=True)
        self.assertIn('time completed', result.stdout)
    
    def test_lgbm_heart(self):
        """LGBM heart."""
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\nheart\nlgbm\ny\n', 
                              text=True, timeout=60, cwd='..', 
                              capture_output=True)
        self.assertIn('time completed', result.stdout)
    
    def test_lgbm_breast(self):
        """LGBM breast."""
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\nbreast\nlgbm\ny\n', 
                              text=True, timeout=60, cwd='..', 
                              capture_output=True)
        self.assertIn('time completed', result.stdout)
    
    def test_lgbm_wine(self):
        """LGBM wine."""
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\nwine\nlgbm\ny\n', 
                              text=True, timeout=60, cwd='..', 
                              capture_output=True)
        self.assertIn('time completed', result.stdout)
    
    def test_lgbm_phishing(self):
        """LGBM phishing."""
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\nphishing\nlgbm\ny\n', 
                              text=True, timeout=300, cwd='..', 
                              capture_output=True)
        self.assertIn('time completed', result.stdout)
    
    def test_lgbm_mushroom(self):
        """LGBM mushroom."""
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\nmushroom\nlgbm\ny\n', 
                              text=True, timeout=120, cwd='..', 
                              capture_output=True)
        self.assertIn('time completed', result.stdout)

if __name__ == '__main__':
    unittest.main()

