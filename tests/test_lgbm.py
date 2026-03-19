#!/usr/bin/env python3
"""LGBM test - robust."""

import subprocess
import sys
import json
import numpy as np
import os
import time

def run_test_lgbm():
    datasets = ['iris', 'heart']
    acc_scores, times = [], []
    
    original_cwd = os.getcwd()
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(root_dir)
    
    for dataset in datasets:
        print(f"Running {dataset} lgbm...")
        input_str = f"1\n{dataset}\nlgbm\nexit\n"
        
        env = os.environ.copy()
        env['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        result = subprocess.run([sys.executable, 'Main.py'], 
                                input=input_str, 
                                text=True, 
                                timeout=120, 
                                cwd=root_dir,
                                env=env,
                                capture_output=True)
        
        print(f"lgbm returncode: {result.returncode}")
        print(f"lgbm STDOUT tail: {result.stdout[-300:]}")
        if result.stderr:
            print(f"lgbm STDERR: {result.stderr[-300:]}")
        
        time.sleep(1)
        
        txt_file = f'{dataset}_comparison.txt'
        for _ in range(5):
            if os.path.exists(txt_file):
                break
            time.sleep(0.3)
        
        if os.path.exists(txt_file):
            try:
                with open(txt_file, 'r') as f:
                    data = json.load(f)
                acc_scores.append(data.get('lgbm_acc', 0))
                times.append(data.get('lgbm_time', 0))
                print(f"Parsed {dataset}: acc={data.get('lgbm_acc',0):.3f} time={data.get('lgbm_time',0):.3f}")
                os.remove(txt_file)
            except Exception as e:
                print(f"Parse error {dataset}: {e}")
                acc_scores.append(0)
                times.append(0)
        else:
            print(f"No TXT for {dataset}")
            acc_scores.append(0)
            times.append(0)
    
    os.chdir(original_cwd)
    
    avg_acc = np.mean(acc_scores)
    avg_time = np.mean(times)
    
    log_data = {
        'datasets': datasets,
        'avg_lgbm_acc': float(avg_acc),
        'avg_lgbm_time': float(avg_time),
        'status': 'PASS' if avg_acc > 0.75 else 'FAIL'
    }
    
    with open('test_lgbm_results.txt', 'w') as f:
        json.dump(log_data, f, indent=2, default=float)
    
    print(f"LGBM test_lgbm_results.txt: status={log_data['status']}, avg_acc={avg_acc:.3f}, avg_time={avg_time:.1f}s")
    sys.exit(0 if log_data['status'] == 'PASS' else 1)

if __name__ == '__main__':
    run_test_lgbm()
