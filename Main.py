"""Main ML model trainer."""

import numpy as np
import pandas as pd
from data_utils import load_and_preprocess_dataset, reshape_for_rnn
from rnn_model import RNNModel
from lstm_model import LSTMModel
from gan_model import GAN, GANClassifier
from lightgbm_model import LightGBMModel
from xgboost_model import XGBoostModel
from plot_utils import plot_single_dataset_comparison
from cross_validation import get_cv_scores
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import time
import warnings
warnings.filterwarnings('ignore')

# TF setup - SILENT
os.environ['SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.random.set_seed(42)
tf.config.optimizer.set_jit(True)

datasets = ['iris', 'heart', 'breast', 'wine', 'phishing', 'mushroom', 'gendername']
models = ['rnn', 'lstm', 'gan', 'lgbm', 'xgb']

def train_single_model(dataset_name, model_name):
    print(f"  {model_name.upper()}:")
    
    X_train, X_test, y_train, y_test, scaler, encoder, is_binary = load_and_preprocess_dataset(dataset_name)
    X_rnn_train, X_rnn_test = reshape_for_rnn(X_train, X_test)
    
    all_labels = np.concatenate([y_train.ravel() if len(y_train.shape) > 1 else y_train,
                                 y_test.ravel() if len(y_test.shape) > 1 else y_test])
    encoder.fit(all_labels)
    
    input_shape = (1, X_train.shape[1])
    num_classes = 2 if is_binary else y_train.shape[1]
    
    result = {'dataset': dataset_name, 'is_binary': is_binary}
    model_start = time.time()
    
    if model_name == 'lgbm':
        lgbm = LightGBMModel()
        lgbm.build(input_dim=X_train.shape[1], num_classes=num_classes, is_binary=is_binary)
        lgbm.train(X_train, y_train)
        result['lgbm_acc'] = lgbm.evaluate(X_test, y_test)['accuracy']
        result['lgbm_time'] = time.time() - model_start
        print(f"time completed: {result['lgbm_time']:.2f}s")
        
    elif model_name == 'xgb':
        xgb = XGBoostModel()
        xgb.build(input_dim=X_train.shape[1], num_classes=num_classes, is_binary=is_binary)
        xgb.train(X_train, y_train)
        result['xgb_acc'] = xgb.evaluate(X_test, y_test)['accuracy']
        result['xgb_time'] = time.time() - model_start
        print(f"time completed: {result['xgb_time']:.2f}s")
        
    elif model_name == 'rnn':
        rnn_cv = get_cv_scores(RNNModel, X_rnn_train, y_train, input_shape, is_binary=is_binary)
        result['rnn_cv'] = rnn_cv['mean_accuracy']
        rnn = RNNModel(hidden_units=50, dropout=0.3)
        rnn.build(input_shape, num_classes, is_binary)
        rnn.train(X_rnn_train, y_train, epochs=30, verbose=0)
        result['rnn_test'] = rnn.evaluate(X_rnn_test, y_test)['accuracy']
        result['rnn_time'] = time.time() - model_start
        print(f"time completed: {result['rnn_time']:.2f}s")
        
    elif model_name == 'lstm':
        lstm_cv = get_cv_scores(LSTMModel, X_rnn_train, y_train, input_shape, is_binary=is_binary)
        result['lstm_cv'] = lstm_cv['mean_accuracy']
        lstm = LSTMModel(hidden_units=50, dropout=0.1)
        lstm.build(input_shape, num_classes, is_binary)
        lstm.train(X_rnn_train, y_train, epochs=30, verbose=0)
        result['lstm_test'] = lstm.evaluate(X_rnn_test, y_test)['accuracy']
        result['lstm_time'] = time.time() - model_start
        print(f"time completed: {result['lstm_time']:.2f}s")
        
    elif model_name == 'gan':
        gen_units = min(64, X_train.shape[1]*8)
        gan = GAN(input_dim=X_train.shape[1], generator_units=gen_units, discriminator_units=gen_units)
        gan.build_generator()
        gan.build_discriminator()
        gan.build_gan()
        gan.train(X_train, epochs=50, batch_size=min(32, X_train.shape[0]//10))
        
        generated = gan.generate_samples(X_test.shape[0])
        gan_X = np.concatenate((X_train, generated), axis=0)
        
        if len(y_train.shape) > 1:
            y_train_sparse = np.argmax(y_train, axis=1)
            y_test_sparse = np.argmax(y_test, axis=1)
        else:
            y_train_sparse = y_train
            y_test_sparse = y_test
        gan_y = np.tile(y_train_sparse, 2)
        gan_num_classes = len(np.unique(gan_y))
        gan_classifier = GANClassifier(hidden_units=min(32, X_train.shape[1]*2))
        gan_classifier.build(input_dim=X_train.shape[1], num_classes=gan_num_classes)
        gan_classifier.train(gan_X, gan_y, epochs=30, verbose=0)
        result['gan_test'] = gan_classifier.evaluate(X_test, y_test_sparse)['accuracy']
        result['gan_time'] = time.time() - model_start
        print(f"time completed: {result['gan_time']:.2f}s")
    
    print(f"      {model_name.upper()} done")
    return result

def main():
    print("Mode:")
    print("1. User mode")
    print("2. Testing mode")
    
    mode_choice = input("Enter (1/2): ").strip()
    
    if mode_choice == '2':
        print("\nTesting:")
        print("1. Specific model test")
        print("2. All testcases")
        
        test_choice = input("Enter (1/2): ").strip()
        
        os.environ['TEST_MODE'] = '1'
        
        if test_choice == '1':
            model = input("Model (rnn/lstm/gan/lgbm/xgb): ").strip()
            dataset = input("Dataset: ").strip()
            result = train_single_model(dataset, model)
            os.makedirs('tests', exist_ok=True)
            with open(f'tests/test_{dataset}_{model}.json', 'w') as f:
                json.dump(result, f, indent=4)
            print(f"Test complete: tests/test_{dataset}_{model}.json")
        
        elif test_choice == '2':
            datasets_test = ['iris', 'heart', 'breast', 'wine']
            results = []
            for dataset in datasets_test:
                print(f"\nTesting ALL MODELS on {dataset}")
                full_result = {'dataset': dataset}
                for model in models:
                    result = train_single_model(dataset, model)
                    full_result[model] = result
                results.append(full_result)
            
            os.makedirs('tests', exist_ok=True)
            with open('tests/test_all_suite.json', 'w') as f:
                json.dump(results, f, indent=4)
            print("Full test suite: tests/test_all_suite.json")
    
    # User mode
    print("1. Single dataset")
    print("2. ALL datasets -> ALL_RESULTS_SUMMARY.txt")
    
    overall_start = time.time()
    
    try:
        choice = input("Enter (1/2): ").strip()
    except:
        choice = '1'
    
    if choice == '2':
        results_data = []
        for dataset in datasets:
            print(f"\n{'='*80}")
            print(f"DATASET: {dataset.upper()} - ALL MODELS {models}")
            print(f"{'='*80}")
            
            full_result = {'dataset': dataset}
            for model in models:
                result = train_single_model(dataset, model)
                full_result.update(result)
            
            os.makedirs('results', exist_ok=True)
            with open(f'results/{dataset}_comparison.txt', 'w') as f:
                json.dump(full_result, f, indent=4)
            results_data.append(full_result)
            print(f"{dataset} ALL MODELS COMPLETE")
        
        os.makedirs('results', exist_ok=True)
        with open('results/ALL_RESULTS_SUMMARY.txt', 'w') as f:
            json.dump(results_data, f, indent=4)
        print("ALL_RESULTS_SUMMARY.txt saved")
    
    else:
        datasets_str = 'iris/heart/breast/wine/phishing/mushroom/gendername'
        print("\nAvailable: lgbm, xgb, rnn, lstm, gan (comma-separate)")
        while True:
            dataset_input = input("Dataset (iris/heart/breast/wine/phishing/mushroom/gendername) or 'exit': ").strip()
            if dataset_input == 'exit':
                print("Goodbye!")
                break
            if dataset_input not in datasets:
                print("Invalid dataset!")
                continue
            
            models_input = input("Models ('all' or comma-separate e.g. 'rnn,lstm'): ").strip().lower()
            if 'all' in models_input:
                sel_models = models
            else:
                sel_models = [m.strip() for m in models_input.split(',') if m.strip()]
            
            full_result = {'dataset': dataset_input}
            for model_name in sel_models:
                result = train_single_model(dataset_input, model_name)
                full_result.update({k: v for k, v in result.items() if k not in full_result})
            
            os.makedirs('results', exist_ok=True)
            with open(f'results/{dataset_input}_comparison.txt', 'w') as f:
                json.dump(full_result, f, indent=4)
            # Test mode - only txt/JSON, no PNG
            if os.getenv('TEST_MODE') != '1':
                plot_single_dataset_comparison(full_result, dataset_input)
                print(f"results/{dataset_input}_comparison.png saved (models: {sel_models})")
            else:
                print(f"results/{dataset_input}_comparison.txt saved (test mode)")
    
    print(f"time completed: {time.time() - overall_start:.2f}s")

if __name__ == '__main__':
    main()

