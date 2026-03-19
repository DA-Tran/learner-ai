#!/usr/bin/env python3
"""RNN test - robust."""

import subprocess
import sys
import json
import numpy as np
import os
import time

def run_test_rnn():
    datasets = ['iris', 'heart']
    cv_scores, test_scores, times = [], [], []
    
    original_cwd = os.getcwd()
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(root_dir)
    
    for dataset in datasets:
        print(f"Running {dataset} rnn...")
        input_str = f"1\n{dataset}\nrnn\nexit\n"
        
        env = os.environ.copy()
        env['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        result = subprocess.run([sys.executable, 'Main.py'], 
                                input=input_str, 
                                text=True, 
                                timeout=300, 
                                cwd=root_dir,
                                env=env,
                                capture_output=True)
        
        print(f"rnn returncode: {result.returncode}")
        print(f"rnn STDOUT tail: {result.stdout[-400:]}")
        if result.stderr:
            print(f"rnn STDERR: {result.stderr[-400:]}")
        
        time.sleep(3)
        
        txt_file = f'{dataset}_comparison.txt'
        for _ in range(10):
            if os.path.exists(txt_file):
                break
            time.sleep(0.5)
        
        if os.path.exists(txt_file):
            try:
                with open(txt_file, 'r') as f:
                    data = json.load(f)
                cv_scores.append(data.get('rnn_cv', 0))
                test_scores.append(data.get('rnn_test', 0))
                times.append(data.get('rnn_time', 0))
                print(f"Parsed {dataset}: cv={data.get('rnn_cv',0):.3f} test={data.get('rnn_test',0):.3f}")
                os.remove(txt_file)
            except Exception as e:
                print(f"Parse error {dataset}: {e}")
                cv_scores.append(0)
                test_scores.append(0)
                times.append(0)
        else:
            print(f"No TXT for {dataset}")
            cv_scores.append(0)
            test_scores.append(0)
            times.append(0)
    
    os.chdir(original_cwd)
    
    avg_cv = np.mean(cv_scores)
    avg_test = np.mean(test_scores)
    avg_time = np.mean(times)
    
    log_data = {
        'datasets': datasets,
        'avg_rnn_cv': float(avg_cv),
        'avg_rnn_test': float(avg_test),
        'avg_rnn_time': float(avg_time),
        'status': 'PASS' if avg_test > 0.4 else 'FAIL'
    }
    
    with open('test_rnn_results.txt', 'w') as f:
        json.dump(log_data, f, indent=2, default=float)
    
    print(f"RNN test_rnn_results.txt: status={log_data['status']}, avg_cv={avg_cv:.3f}, avg_test={avg_test:.3f}, avg_time={avg_time:.1f}s")
    sys.exit(0 if log_data['status'] == 'PASS' else 1)

if __name__ == '__main__':
    run_test_rnn()
