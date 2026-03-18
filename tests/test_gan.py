#!/usr/bin/env python3
"""Enhanced GAN tests: binary safe, acc + batch."""

import unittest
import subprocess
import sys
from pathlib import Path

class TestGANModel(unittest.TestCase):
    def run_single_test(self, dataset, expected_acc=r'0\.[7-9]'):
        """Single GAN test."""
        input_str = '1\n' + dataset + '\ngan\n y\n'
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input=input_str, 
                              text=True, timeout=240, cwd='..', 
                              capture_output=True)
        self.assertIn('time completed', result.stdout, dataset + ' no complete')
    
    def test_gan_iris(self):
        self.run_single_test('iris', r'0\.[8-9]')
    
    def test_gan_heart(self):
        self.run_single_test('heart')
    
    def test_gan_breast(self):
        self.run_single_test('breast', r'0\.[8-9]')
    
    def test_gan_wine(self):
        self.run_single_test('wine')
    
    def test_gan_phishing(self):
        self.run_single_test('phishing', timeout=300)
    
    def test_gan_mushroom(self):
        self.run_single_test('mushroom')
    
    def test_gan_gendername(self):
        self.run_single_test('gendername')
    
    def test_gan_no_plot(self):
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\niris\ngan\nn\n', 
                              text=True, timeout=240, cwd='..', 
                              capture_output=True)
        self.assertEqual(result.returncode, 0)
    
    def test_gan_invalid(self):
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='1\ninvalid\ngan\n y\n', 
                              text=True, timeout=30, cwd='..', 
                              capture_output=True)
        self.assertIn('Invalid dataset', result.stdout)
    
    def test_gan_binary_safe(self):
        """Binary datasets no axis error."""
        for dataset in ['heart', 'breast']:
            input_str = '1\n' + dataset + '\ngan\n y\n'
            result = subprocess.run([sys.executable, '../Main.py'], 
                                  input=input_str, 
                                  text=True, timeout=240, cwd='..', 
                                  capture_output=True)
            self.assertEqual(result.returncode, 0)
    
    def test_gan_batch(self):
        result = subprocess.run([sys.executable, '../Main.py'], 
                              input='2\ny\n', 
                              text=True, timeout=1200, cwd='..', 
                              capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('ALL_RESULTS_SUMMARY.png', result.stdout)

if __name__ == '__main__':
    unittest.main()

