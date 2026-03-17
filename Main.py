"""Main ML model trainer."""

import numpy as np
import pandas as pd
from data_utils import load_and_preprocess_dataset, reshape_for_rnn
from rnn_model import RNNModel
from lstm_model import LSTMModel
from gan_model import GAN, GANClassifier
from lightgbm_model import LightGBMModel
from xgboost_model import XGBoostModel
from pipeline import train_all_models
from plot_utils import plot_single_dataset_comparison
from cross_validation import get_cv_scores
import os
import tensorflow as tf
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# TF setup - SILENT execution
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Fatal only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.random.set_seed(42)
tf.config.optimizer.set_jit(True)

def train_single_model(dataset_name, model_name):
    """Train single model. Returns dict with its metrics only."""
    import time
    print(f"  {model_name.upper()}:")
    
    X_train, X_test, y_train, y_test, scaler, encoder, is_binary = load_and_preprocess_dataset(dataset_name)
    X_rnn_train, X_rnn_test = reshape_for_rnn(X_train, X_test)
    
    all_labels = np.concatenate([y_train.ravel() if len(y_train.shape) > 1 else y_train,
                                 y_test.ravel() if len(y_test.shape) > 1 else y_test])
    encoder.fit(all_labels)
    
    input_shape = (1, X_train.shape[1])
    num_classes = 2 if is_binary else y_train.shape[1]
    
    result = {'dataset': dataset_name, 'is_binary': is_binary}
    
    if model_name == 'lgbm':
        lgbm = LightGBMModel()
        lgbm.build(input_dim=X_train.shape[1], num_classes=num_classes, is_binary=is_binary)
        start = time.time()
        lgbm.train(X_train, y_train)
        result['lgbm_acc'] = lgbm.evaluate(X_test, y_test)['accuracy']
        result['lgbm_time'] = time.time() - start
        
    elif model_name == 'xgb':
        xgb = XGBoostModel()
        xgb.build(input_dim=X_train.shape[1], num_classes=num_classes, is_binary=is_binary)
        start = time.time()
        xgb.train(X_train, y_train)
        result['xgb_acc'] = xgb.evaluate(X_test, y_test)['accuracy']
        result['xgb_time'] = time.time() - start
        
    elif model_name == 'rnn':
        rnn_cv = get_cv_scores(RNNModel, X_rnn_train, y_train, input_shape, is_binary=is_binary)
        result['rnn_cv'] = rnn_cv['mean_accuracy']
        rnn = RNNModel(hidden_units=50, dropout=0.3)
        rnn.build(input_shape, num_classes, is_binary)
        start = time.time()
        rnn.train(X_rnn_train, y_train, epochs=30, verbose=0)
        result['rnn_test'] = rnn.evaluate(X_rnn_test, y_test)['accuracy']
        result['rnn_time'] = time.time() - start
        
    elif model_name == 'lstm':
        lstm_cv = get_cv_scores(LSTMModel, X_rnn_train, y_train, input_shape, is_binary=is_binary)
        result['lstm_cv'] = lstm_cv['mean_accuracy']
        lstm = LSTMModel(hidden_units=50, dropout=0.1)
        lstm.build(input_shape, num_classes, is_binary)
        start = time.time()
        lstm.train(X_rnn_train, y_train, epochs=30, verbose=0)
        result['lstm_test'] = lstm.evaluate(X_rnn_test, y_test)['accuracy']
        result['lstm_time'] = time.time() - start
        
    elif model_name == 'gan':
        gen_units = min(64, X_train.shape[1]*8)
        gan = GAN(input_dim=X_train.shape[1], generator_units=gen_units, discriminator_units=gen_units)
        gan.build_generator()
        gan.build_discriminator()
        gan.build_gan()
        gan.train(X_train, epochs=50, batch_size=min(32, X_train.shape[0]//10))
        
        generated = gan.generate_samples(X_test.shape[0])
        gan_X = np.concatenate((X_test, generated), axis=0)
        
        if len(y_test.shape) == 1:
            y_test_labels = y_test
            gan_y = np.tile(y_test_labels, 2)
            gan_y_onehot = encoder.transform(gan_y)
        else:
            y_test_labels = np.argmax(y_test, axis=1)
            gan_y = np.tile(y_test_labels, 2)
            gan_y_onehot = encoder.transform(gan_y)
        
        # Fix num_classes for iris (3 classes)
        gan_num_classes = len(np.unique(all_labels))
        gan_classifier = GANClassifier(hidden_units=min(32, X_train.shape[1]*2))
        gan_classifier.build(input_dim=X_train.shape[1], num_classes=gan_num_classes)
        start = time.time()
        gan_classifier.train(gan_X, gan_y_onehot, epochs=30, verbose=0)
        result['gan_test'] = gan_classifier.evaluate(X_test, y_test)['accuracy']
        result['gan_time'] = time.time() - start
    
    print(f"    ✓ {model_name.upper()} {result[list(result.keys())[-1]]:.3f}")
    return result

def train_single_dataset(dataset_name):
    """Train all models on single dataset. Return metrics only."""
    import time
    print("="*80)
    print(f"PROCESSING {dataset_name.upper()}")
    print("="*80)
    
    # Load data
    X_train, X_test, y_train, y_test, scaler, encoder, is_binary = load_and_preprocess_dataset(dataset_name)

    X_rnn_train, X_rnn_test = reshape_for_rnn(X_train, X_test)
    
    # Fix encoder for GAN - fit on all labels
    all_labels = np.concatenate([
        y_train.ravel() if len(y_train.shape) > 1 else y_train,
        y_test.ravel() if len(y_test.shape) > 1 else y_test
    ])
    encoder.fit(all_labels)
    
    input_shape = (1, X_train.shape[1])
    num_classes = 2 if is_binary else y_train.shape[1]
    
    # Cross-validation

    print("\nCROSS-VALIDATION")
    rnn_cv = get_cv_scores(RNNModel, X_rnn_train, y_train, input_shape=input_shape, is_binary=is_binary)
    lstm_cv = get_cv_scores(LSTMModel, X_rnn_train, y_train, input_shape=input_shape, is_binary=is_binary)
    # Tree models use flat data, no CV here (sklearn CV later)
    
    # Train TREE MODELS (flat data)
    print("\nTREE MODELS")
    lgbm = LightGBMModel()
    lgbm.build(input_dim=X_train.shape[1], num_classes=num_classes, is_binary=is_binary)
    lgbm_start = time.time()
    lgbm.train(X_train, y_train)
    lgbm_time = time.time() - lgbm_start
    lgbm_metrics = lgbm.evaluate(X_test, y_test)
    print(f"  LightGBM: {lgbm_time:.2f}s Acc: {lgbm_metrics['accuracy']:.3f}")
    
    xgb = XGBoostModel()
    xgb.build(input_dim=X_train.shape[1], num_classes=num_classes, is_binary=is_binary)
    xgb_start = time.time()
    xgb.train(X_train, y_train)
    xgb_time = time.time() - xgb_start
    xgb_metrics = xgb.evaluate(X_test, y_test)
    print(f"  XGBoost: {xgb_time:.2f}s Acc: {xgb_metrics['accuracy']:.3f}")
    
    # Train NN MODELS
    print("\nNN MODELS")
    rnn = RNNModel(hidden_units=50, dropout=0.3)
    lstm = LSTMModel(hidden_units=50, dropout=0.1)
    num_classes = 2 if is_binary else y_train.shape[1]
    
    rnn.build(input_shape=input_shape, num_classes=num_classes, is_binary=is_binary)
    lstm.build(input_shape=input_shape, num_classes=num_classes, is_binary=is_binary)
    
    print("  RNN:")
    rnn_start = time.time()
    rnn_history = rnn.train(X_rnn_train, y_train, epochs=30, validation_split=0.2, verbose=0)
    rnn_train_time = time.time() - rnn_start
    print(f"    Train: {rnn_train_time:.2f}s")
    rnn_test_start = time.time()
    rnn_metrics = rnn.evaluate(X_rnn_test, y_test)
    print(f"    Test: {time.time() - rnn_test_start:.2f}s")
    print("  LSTM:")
    lstm_start = time.time()
    lstm_history = lstm.train(X_rnn_train, y_train, epochs=30, validation_split=0.2, verbose=0)
    print(f"    Train: {time.time() - lstm_start:.2f}s")
    lstm_test_start = time.time()
    lstm_metrics = lstm.evaluate(X_rnn_test, y_test)
    print(f"    Test: {time.time() - lstm_test_start:.2f}s")
    
    rnn_metrics = rnn.evaluate(X_rnn_test, y_test)
    lstm_metrics = lstm.evaluate(X_rnn_test, y_test)
    
    # GAN
    print("\nGAN TRAINING")
    gen_units = min(64, X_train.shape[1]*8)
    gan = GAN(input_dim=X_train.shape[1], generator_units=gen_units, discriminator_units=gen_units)
    gan.build_generator()
    gan.build_discriminator()
    gan.build_gan()
    gan.train(X_train, epochs=50, batch_size=min(32, X_train.shape[0]//10))
    
    generated = gan.generate_samples(X_test.shape[0])
    gan_X = np.concatenate((X_test, generated), axis=0)
    
    # Encoder transform only
    if len(y_test.shape) == 1:
        y_test_labels = y_test
        gan_y = np.tile(y_test_labels, 2)
        gan_y_onehot = encoder.transform(gan_y)
    else:
        y_test_labels = np.argmax(y_test, axis=1)
        gan_y = np.tile(y_test_labels, 2)
        gan_y_onehot = encoder.transform(gan_y)
    
    gan_classifier = GANClassifier(hidden_units=min(32, X_train.shape[1]*2))
    gan_classifier.build(input_dim=X_train.shape[1], num_classes=num_classes)
    num_classes = 1 if is_binary else len(np.unique(all_labels))
    gan_classifier.build(input_dim=X_train.shape[1], num_classes=num_classes)
    print("  GAN:")
    gan_start = time.time()
    gan_history = gan_classifier.train(gan_X, gan_y_onehot, epochs=30, validation_split=0.2)
    print(f"    Train: {time.time() - gan_start:.2f}s")
    gan_test_start = time.time()
    gan_metrics = gan_classifier.evaluate(X_test, y_test)
    print(f"    Test: {time.time() - gan_test_start:.2f}s")
    
    print(f"\n{dataset_name.upper()} RESULTS:")
    print(f"RNN: {rnn_metrics['accuracy']:.3f} | LSTM: {lstm_metrics['accuracy']:.3f} | GAN: {gan_metrics['accuracy']:.3f}")
    
    return {
        'dataset': dataset_name,
        'is_binary': is_binary,
        'rnn_cv': rnn_cv['mean_accuracy'],
        'lstm_cv': lstm_cv['mean_accuracy'],
        'rnn_test': rnn_metrics['accuracy'],
        'lstm_test': lstm_metrics['accuracy'],
        'gan_test': gan_metrics['accuracy'],
        'lgbm_acc': lgbm_metrics['accuracy'],
        'xgb_acc': xgb_metrics['accuracy'],
        'lgbm_time': lgbm_time,
        'xgb_time': xgb_time,
        'rnn_time': (time.time() - rnn_start) if 'rnn_start' in locals() else 0,
        'lstm_time': (time.time() - lstm_start) if 'lstm_start' in locals() else 0,
        'gan_time': (time.time() - gan_start) if 'gan_start' in locals() else 0
    }

def main():
    datasets = ['iris', 'heart', 'breast', 'wine', 'phishing', 'mushroom', 'gendername']
    
    print("ML Pipeline v3.0")
    print("1. Single dataset")
    print("2. ALL datasets → 1 PNG summary")
    
    import time
    overall_start_time = time.time()
    
    choice = input("Enter (1/2): ").strip()
    
    results_data = []
    
    if choice == '2':
        print("\nRUNNING ALL...")
        for dataset in datasets:
            try:
                result = train_single_dataset(dataset)
                results_data.append(result)
                print(f"{dataset.upper()} COMPLETE\n")
            except Exception as e:
                print(f"{dataset}: {e}")
        
        # Create 1 PNG with ALL results - handle partial failures
        successful_datasets = [r['dataset'] for r in results_data]
        if results_data:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Accuracy test bar
            n_datasets = len(results_data)
            x = np.arange(n_datasets)
            width = 0.25
            rnn_acc = [r['rnn_test'] for r in results_data]
            lstm_acc = [r['lstm_test'] for r in results_data]
            gan_acc = [r['gan_test'] for r in results_data]
            ax1.bar(x - width, rnn_acc, width, label='RNN', alpha=0.8)
            ax1.bar(x, lstm_acc, width, label='LSTM', alpha=0.8)
            ax1.bar(x + width, gan_acc, width, label='GAN', alpha=0.8)
            ax1.set_title('Test Accuracy by Model')
            ax1.set_xticks(x)
            ax1.set_xticklabels(successful_datasets)
            ax1.legend()
            ax1.set_ylim(0, 1)
            
            # 2. CV Accuracy

            n_datasets = len(results_data)
            rnn_cv = [r['rnn_cv'] for r in results_data]
            lstm_cv = [r['lstm_cv'] for r in results_data]
            x2 = np.arange(n_datasets)
            width_cv = 0.4
            ax2.bar(x2 - width_cv/2, rnn_cv, width_cv, label='RNN CV', alpha=0.8)
            ax2.bar(x2 + width_cv/2, lstm_cv, width_cv, label='LSTM CV', alpha=0.8)
            ax2.set_title('Cross-Validation Accuracy')
            ax2.set_xticks(x2)
            ax2.set_xticklabels(successful_datasets)

            ax2.legend()
            ax2.set_ylim(0, 1)
            
            # 3. Average per model
            avg_rnn = np.mean(rnn_acc)
            avg_lstm = np.mean(lstm_acc)
            avg_gan = np.mean(gan_acc)
            ax3.bar(['RNN', 'LSTM', 'GAN'], [avg_rnn, avg_lstm, avg_gan], alpha=0.8, color=['red', 'blue', 'green'])
            ax3.set_title('Average Test Accuracy')
            ax3.set_ylim(0, 1)
            for i, v in enumerate([avg_rnn, avg_lstm, avg_gan]):
                ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
            
            # 4. Dataset performance overview

            dataset_acc = [ (r['rnn_test'] + r['lstm_test'] + r['gan_test'])/3 for r in results_data ]
            ax4.bar(successful_datasets, dataset_acc, alpha=0.8, color='orange')

            ax4.set_title('Average Accuracy per Dataset')
            ax4.set_ylim(0, 1)
            for i, v in enumerate(dataset_acc):
                ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('ALL_RESULTS_SUMMARY.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("\nSINGLE SUMMARY PNG: ALL_RESULTS_SUMMARY.png")
        
        print("\nDONE!")
        
    elif choice == '1':
        while True:
            print("\nAvailable: lgbm, xgb, rnn, lstm, gan (comma-separate)")
            dataset_input = input("Dataset (iris/heart/breast/wine/phishing/mushroom/gendername) or 'exit': ").strip()
            if dataset_input == 'exit':
                print("Goodbye!")
                break
            if dataset_input not in datasets:
                print("Invalid dataset!")
                continue
            
            models_input = input("Models ('all' or comma-separate e.g. 'lgbm,xgb,rnn'): ").strip().lower()
            if 'all' in models_input:
                models = ['lgbm', 'xgb', 'rnn', 'lstm', 'gan']
            else:
                models = [m.strip() for m in models_input.split(',') if m.strip()]
            
            # Train selected
            full_result = {'dataset': dataset_input, 'is_binary': False}
            for model_name in models:
                result = train_single_model(dataset_input, model_name)
                # Merge metrics
                for k, v in result.items():
                    full_result[k] = v
            
            plot_single_dataset_comparison(full_result, dataset_input)
            print(f"{dataset_input}_comparison.png (models: {models})")
    else:
        main()

if __name__ == "__main__":
    main()

