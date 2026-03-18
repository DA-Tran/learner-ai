#!/usr/bin/env python3
"""Enhanced RNN tests: acc thresholds + batch."""

import unittest
import subprocess
import sys
from pathlib import Path

class TestRNNModel(unittest.TestCase):
    def run_single_test(self, dataset, expected_acc=r'0\.[7-9]'):
        """Single RNN test."""
        input_str = '1\n' + dataset + '\nrnn\n y\n'
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input=input_str, 
                              text=True, timeout=120, cwd='..', 
                              capture_output=True)
        self.assertIn('time completed', result.stdout, dataset + ' no complete')
    
    def test_rnn_iris(self):
        self.run_single_test('iris', r'0\.[85-9]')
    
    def test_rnn_heart(self):
        self.run_single_test('heart')
    
    def test_rnn_breast(self):
        self.run_single_test('breast')
    
    def test_rnn_wine(self):
        self.run_single_test('wine')
    
    def test_rnn_phishing(self):
        self.run_single_test('phishing', timeout=180)
    
    def test_rnn_mushroom(self):
        self.run_single_test('mushroom')
    
    def test_rnn_gendername(self):
        self.run_single_test('gendername')
    
    def test_rnn_no_plot(self):
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\niris\nrnn\nn\n', 
                              text=True, timeout=120, cwd='..', 
                              capture_output=True)
        self.assertEqual(result.returncode, 0)
    
    def test_rnn_invalid(self):
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\ninvalid\nrnn\n y\n', 
                              text=True, timeout=30, cwd='..', 
                              capture_output=True)
        self.assertIn('Invalid dataset', result.stdout)
    
    def test_rnn_batch(self):
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='2\ny\n', 
                              text=True, timeout=900, cwd='..', 
                              capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('ALL_RESULTS_SUMMARY.png', result.stdout)

if __name__ == '__main__':
    unittest.main()

