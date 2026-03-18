#!/usr/bin/env python3
"""Master test runner - merged test_<model>.py."""

import subprocess
import sys
from pathlib import Path

def run_model_tests(test_file):
    """Run test file."""
    result = subprocess.run([sys.executable, test_file], 
                          timeout=1800, 
                          cwd=Path('.'),
                          capture_output=True, text=True)
    print(f"\n{'='*60}")
    print(f"Test file: {test_file}")
    print(f"PASSED: {result.returncode == 0}")
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout[:200]}...")
        print(f"STDERR: {result.stderr[:200]}...")
    return result.returncode == 0

def main():
    """Run all."""
    tests = [
        'tests/test_rnn.py',
        'tests/test_lstm.py',
        'tests/test_gan.py',
        'tests/test_lgbm.py',
        'tests/test_xgb.py',
        'tests/test_all_models.py'
    ]
    
    print("ML PIPELINE FULL TEST SUITE")
    print("=" * 60)
    
    passed = 0
    for test in tests:
        if run_model_tests(test):
            passed += 1
    
    print(f"\nFINAL: {passed}/6 TESTS PASSED")
    if passed == 6:
        print("ALL GOOD!")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()

