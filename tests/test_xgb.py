#!/usr/bin/env python3
"""XGBoost tests."""

import unittest
import subprocess
import sys

class TestXGBoostModel(unittest.TestCase):
    def test_xgb_iris(self):
        """XGB iris."""
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\niris\nxgb\ny\n', 
                              text=True, timeout=60, cwd='..', 
                              capture_output=True)
        self.assertIn('time completed', result.stdout)
    
    def test_xgb_heart(self):
        """XGB heart."""
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\nheart\nxgb\ny\n', 
                              text=True, timeout=60, cwd='..', 
                              capture_output=True)
        self.assertIn('time completed', result.stdout)
    
    def test_xgb_breast(self):
        """XGB breast."""
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\nbreast\nxgb\ny\n', 
                              text=True, timeout=60, cwd='..', 
                              capture_output=True)
        self.assertIn('time completed', result.stdout)
    
    def test_xgb_wine(self):
        """XGB wine."""
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\nwine\nxgb\ny\n', 
                              text=True, timeout=60, cwd='..', 
                              capture_output=True)
        self.assertIn('time completed', result.stdout)
    
    def test_xgb_phishing(self):
        """XGB phishing."""
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\nphishing\nxgb\ny\n', 
                              text=True, timeout=300, cwd='..', 
                              capture_output=True)
        self.assertIn('time completed', result.stdout)
    
    def test_xgb_mushroom(self):
        """XGB mushroom."""
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\nmushroom\nxgb\ny\n', 
                              text=True, timeout=120, cwd='..', 
                              capture_output=True)
        self.assertIn('time completed', result.stdout)

if __name__ == '__main__':
    unittest.main()

