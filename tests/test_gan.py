#!/usr/bin/env python3
"""GAN test - robust."""

import subprocess
import sys
import json
import numpy as np
import os
import time

def run_test_gan():
    datasets = ['iris']
    test_scores, times = [], []
    
    original_cwd = os.getcwd()
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(root_dir)
    
    for dataset in datasets:
        print(f"Running {dataset} gan...")
        input_str = f"1\n{dataset}\ngan\n"
        
        env = os.environ.copy()
        env['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        result = subprocess.run([sys.executable, 'Main.py'], 
                                input=input_str, 
                                text=True, 
                                timeout=900,
                                cwd=root_dir,
                                env=env,
                                capture_output=True)
        
        print(f"gan returncode: {result.returncode}")
        print(f"gan STDOUT tail: {result.stdout[-400:]}")
        if result.stderr:
            print(f"gan STDERR: {result.stderr[-400:]}")
        
        time.sleep(5)
        
        txt_file = f'{dataset}_comparison.txt'
        for _ in range(20):
            if os.path.exists(txt_file):
                break
            time.sleep(0.5)
        
        if os.path.exists(txt_file):
            try:
                with open(txt_file, 'r') as f:
                    data = json.load(f)
                test_scores.append(data.get('gan_test', 0))
                times.append(data.get('gan_time', 0))
                print(f"Parsed {dataset}: test={data.get('gan_test',0):.3f} time={data.get('gan_time',0):.1f}")
                os.remove(txt_file)
            except Exception as e:
                print(f"Parse error {dataset}: {e}")
                test_scores.append(0)
                times.append(0)
        else:
            print(f"No TXT for {dataset}")
            test_scores.append(0)
            times.append(0)
    
    os.chdir(original_cwd)
    
    avg_test = np.mean(test_scores)
    avg_time = np.mean(times)
    
    log_data = {
        'datasets': datasets,
        'avg_gan_test': float(avg_test),
        'avg_gan_time': float(avg_time),
        'status': 'PASS' if avg_test > 0.3 else 'FAIL'
    }
    
    with open('test_gan_results.txt', 'w') as f:
        json.dump(log_data, f, indent=2, default=float)
    
    print(f"GAN test_gan_results.txt: status={log_data['status']}, avg_test={avg_test:.3f}, avg_time={avg_time:.1f}s")
    sys.exit(0 if log_data['status'] == 'PASS' else 1)

if __name__ == '__main__':
    run_test_gan()
