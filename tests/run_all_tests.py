#!/usr/bin/env python3
"""ML test - per dataset-model rc/acc/time."""

import subprocess
import sys
import json
from pathlib import Path
import os
import time
import numpy as np

ROOT_DIR = Path.cwd()
DATASETS = ['iris', 'heart', 'breast', 'wine']
MODELS = ['rnn', 'lstm', 'gan', 'lgbm', 'xgb']

def clean_txt():
    for p in ['*_comparison.txt', 'test_*_results.json', 'ALL_RESULTS_SUMMARY.txt']:
        for f in ROOT_DIR.glob(p):
            f.unlink(missing_ok=True)

def run_individual(model_name):
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} - Datasets: {', '.join(DATASETS)}")
    print(f"{'='*60}")
    
    test_scores = []
    times = []
    for ds in DATASETS:
        print(f"  Testing {ds}...")
        input_str = f"1\n{ds}\n{model_name}\nexit\n"
        
        env = os.environ.copy()
        env.update({'TF_CPP_MIN_LOG_LEVEL': '3', 'SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL': 'True', 'TEST_MODE': '1'})
        
        start_time = time.time()
        result = subprocess.run([sys.executable, 'Main.py'], input=input_str, cwd=ROOT_DIR, 
                                timeout=180, capture_output=True, text=True, env=env)
        duration = time.time() - start_time
        
        rc_status = 'PASS' if result.returncode == 0 else 'FAIL'
        print(f"    rc={result.returncode} ({rc_status}) duration={duration:.1f}s")
        
        txt = ROOT_DIR / f'{ds}_comparison.txt'
        test_acc = 0.0
        model_time = duration
        for _ in range(20):
            if txt.exists():
                try:
                    data = json.loads(txt.read_text())
                    test_acc = data.get(f'{model_name}_test', data.get(f'{model_name}_acc', 0.0))
                    model_time = data.get(f'{model_name}_time', duration)
                    txt.unlink()
                    print(f"    acc={test_acc:.3f} time={model_time:.1f}s")
                    break
                except:
                    pass
            time.sleep(0.5)
        test_scores.append(test_acc)
        times.append(model_time)
    
    avg_test = np.mean(test_scores)
    avg_time = np.mean(times)
    status = 'PASS' if avg_test >= 0.6 else 'FAIL'
    print(f"AVG test={avg_test:.3f} time={avg_time:.1f}s ({status})")
    
    result_data = {
        'model': model_name, 'avg_test': float(avg_test), 'avg_time': float(avg_time),
        'status': status, 'per_dataset': dict(zip(DATASETS, test_scores))
    }
    (ROOT_DIR / f'test_{model_name}_results.json').write_text(json.dumps(result_data, indent=2))
    return avg_test >= 0.6

def run_all_suite():
    print(f"\n{'='*60}")
    print("ALL DATASETS x ALL MODELS")
    print(f"{'='*60}")
    
    env = os.environ.copy()
    env.update({'TF_CPP_MIN_LOG_LEVEL': '3', 'SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL': 'True'})
    
    start_time = time.time()
    result = subprocess.run([sys.executable, 'Main.py'], input='2\n', cwd=ROOT_DIR, 
                            timeout=1800, capture_output=True, text=True, env=env)
    duration = time.time() - start_time
    
    print(f"ALL rc={result.returncode} duration={duration:.1f}s")
    
    # Parse all comparison.txt
    all_scores = []
    for ds in DATASETS:
        txt = ROOT_DIR / f'{ds}_comparison.txt'
        if txt.exists():
            try:
                data = json.loads(txt.read_text())
                # Average across models if multiple
                model_scores = [v for k, v in data.items() if 'acc' in k or 'cv' in k]
                avg_ds = np.mean(model_scores) if model_scores else 0
                all_scores.append(avg_ds)
                txt.unlink()
            except:
                pass
    suite_avg = np.mean(all_scores)
    suite_status = 'PASS' if suite_avg >= 0.6 else 'FAIL'
    print(f"SUITE avg={suite_avg:.3f} ({suite_status})")
    
    (ROOT_DIR / 'test_suite_results.json').write_text(json.dumps({'avg': float(suite_avg), 'status': suite_status}))
    return suite_avg >= 0.6

def main():
    print("ML Suite: 4 datasets x 5 models (rc/acc/time)")
    clean_txt()
    
    passes = [run_individual(m) for m in MODELS]
    suite_pass = run_all_suite()
    
    print("\nSUMMARY:")
    for m, p in zip(MODELS, passes):
        print(f"  {m}: {p}")
    print(f"  SUITE: {suite_pass}")
    
    sys.exit(0 if all(passes + [suite_pass]) else 1)

if __name__ == '__main__':
    main()

