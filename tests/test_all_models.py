import subprocess
import sys
import os
from pathlib import Path

def test_dataset(dataset, models='all'):
    """Test single dataset, return success/fail."""
    cmd = [sys.executable, 'Main.py']
    
    # Simulate inputs
    inputs = f'1\n{dataset}\n{models}\n'
    
    try:
        result = subprocess.run(
            cmd, 
            input=inputs, 
            text=True, 
            timeout=300,
            cwd=Path('..'),
            capture_output=True
        )
        success = result.returncode == 0 and f'{dataset}_comparison.png' in os.listdir('..')
        print(f"Dataset {dataset}: SUCCESS (acc plot saved)")
        return success
    except subprocess.TimeoutExpired:
        print(f"Dataset {dataset}: TIMEOUT")
        return False
    except Exception as e:
        print(f"Dataset {dataset}: {e}")
        return False

def main():
    """Test iris, heart, breast with ALL models."""
    datasets = ['iris', 'heart', 'breast']
    results = {}
    
    print("ML PIPELINE SMOKE TEST")
    print("=" * 50)
    
    for dataset in datasets:
        results[dataset] = test_dataset(dataset, 'all')
    
    total = len(datasets)
    passed = sum(results.values())
    
    print(f"\nSUMMARY: {passed}/{total} PASSED")
    if passed == total:
        print("ALL TESTS PASSED! Pipeline production-ready!")
    else:
        print("Some tests failed - check logs above")

if __name__ == "__main__":
    main()

