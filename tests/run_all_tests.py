#!/usr/bin/env python3
"""ML test - dataset time limits."""

import subprocess
import sys
import json
from pathlib import Path
import os
import time
import numpy as np

ROOT_DIR = Path.cwd()
TEST_DATASETS = ['iris', 'heart', 'breast', 'wine']
TIME_LIMITS = {'iris': 60, 'heart': 120, 'breast': 180, 'wine': 300}  # seconds
MODELS = ['rnn', 'lstm', 'gan', 'lgbm', 'xgb']

def clean_txt():
    for p in ['*_comparison.txt', 'test_*_results.json', 'ALL_RESULTS_SUMMARY.txt']:
        for f in ROOT_DIR.glob(p):
            f.unlink(missing_ok=True)

def run_individual(model_name):
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} - Datasets: {', '.join(TEST_DATASETS)}")
    print(f"{'='*60}")
    
    times = []
    for ds in TEST_DATASETS:
        print(f"  Testing {ds}...")
        input_str = f"1\n{ds}\n{model_name}\nexit\n"
        
        env = os.environ.copy()
        env.update({'TF_CPP_MIN_LOG_LEVEL': '3', 'SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL': 'True', 'TEST_MODE': '1'})
        
        start_time = time.time()
        result = subprocess.run([sys.executable, 'Main.py'], input=input_str, cwd=ROOT_DIR, 
                                timeout=300, capture_output=True, text=True, env=env)
        duration = time.time() - start_time
        
        print(f"    rc={result.returncode} duration={duration:.1f}s")
        
        txt = ROOT_DIR / f'{ds}_comparison.txt'
        model_time = duration
        for _ in range(20):
            if txt.exists():
                try:
                    data = json.loads(txt.read_text())
                    model_time = data.get(f'{model_name}_time', duration)
                    txt.unlink()
                    break
                except:
                    pass
            time.sleep(0.5)
        times.append(model_time)
    
    total_time = sum(times)
    limit_total = sum(TIME_LIMITS.values())
    status = 'PASS' if total_time <= limit_total else 'FAIL'
    print(f"TOTAL TIME: {total_time:.1f}s vs limit {limit_total}s ({status})")
    
    result_data = {
        'model': model_name, 'total_time': float(total_time), 
        'status': status, 'per_dataset': dict(zip(TEST_DATASETS, times))
    }
    (ROOT_DIR / f'test_{model_name}_results.json').write_text(json.dumps(result_data, indent=2))
    return status == 'PASS'

def main():
    print(f"ML Suite: {len(TEST_DATASETS)} datasets x {len(MODELS)} models (time-based)")
    clean_txt()
    
    passes = [run_individual(m) for m in MODELS]
    
    print("\nSUMMARY:")
    for m, p in zip(MODELS, passes):
        print(f"  {m}: {p}")
    
    sys.exit(0 if all(passes) else 1)

if __name__ == '__main__':
    main()

