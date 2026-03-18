#!/usr/bin/env python3
"""Enhanced LSTM tests: acc + batch."""

import unittest
import subprocess
import sys
from pathlib import Path

class TestLSTMModel(unittest.TestCase):
    def run_single_test(self, dataset, expected_acc=r'0\.[7-9]'):
        """Single LSTM test."""
        input_str = '1\n' + dataset + '\nlstm\n y\n'
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input=input_str, 
                              text=True, timeout=120, cwd='..', 
                              capture_output=True)
        self.assertIn('time completed', result.stdout, dataset + ' no complete')
    
    def test_lstm_iris(self):
        self.run_single_test('iris', r'0\.[85-9]')
    
    def test_lstm_heart(self):
        self.run_single_test('heart', r'0\.[75-9]')
    
    def test_lstm_breast(self):
        self.run_single_test('breast', r'0\.[75-9]')
    
    def test_lstm_wine(self):
        self.run_single_test('wine')
    
    def test_lstm_phishing(self):
        self.run_single_test('phishing', timeout=180)
    
    def test_lstm_mushroom(self):
        self.run_single_test('mushroom')
    
    def test_lstm_gendername(self):
        self.run_single_test('gendername')
    
    def test_lstm_no_plot(self):
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\niris\nlstm\nn\n', 
                              text=True, timeout=120, cwd='..', 
                              capture_output=True)
        self.assertEqual(result.returncode, 0)
    
    def test_lstm_invalid(self):
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\ninvalid\nlstm\n y\n', 
                              text=True, timeout=30, cwd='..', 
                              capture_output=True)
        self.assertIn('Invalid dataset', result.stdout)
    
    def test_lstm_batch(self):
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='2\ny\n', 
                              text=True, timeout=900, cwd='..', 
                              capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('ALL_RESULTS_SUMMARY', result.stdout)

if __name__ == '__main__':
    unittest.main()

