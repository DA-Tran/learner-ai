"""Main ML model trainer."""

import numpy as np
import pandas as pd
from data_utils import load_and_preprocess_dataset, reshape_for_rnn
from rnn_model import RNNModel
from lstm_model import LSTMModel
from gan_model import GAN, GANClassifier
from evaluation import print_metrics, plot_confusion_matrix, plot_learning_curves, plot_metrics
from cross_validation import get_cv_scores
import os
import tensorflow as tf

# TF setup
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(42)
tf.config.optimizer.set_jit(False)

def main():
    # Dataset selection
    dataset_choice = input("Choose dataset (iris/heart/breast/youtube): ").strip().lower()
    if dataset_choice not in ['iris', 'heart', 'breast', 'youtube']:
        dataset_choice = 'iris'
    
    print("="*60)
    print(f"Loading {dataset_choice.upper()}...")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test, scaler, encoder = load_and_preprocess_dataset(dataset_choice)
    X_rnn_train, X_rnn_test = reshape_for_rnn(X_train, X_test)
    print(f"Dataset shape: {X_train.shape}")
    
    # Cross-validation
    print("\nCROSS-VALIDATION")
    print("="*60)
    input_shape = (1, X_train.shape[1])
    rnn_cv = get_cv_scores(RNNModel, X_rnn_train, y_train, input_shape=input_shape)
    print(f"RNN CV: {rnn_cv['mean_accuracy']:.4f} (+/- {rnn_cv['std_accuracy']:.4f})")
    lstm_cv = get_cv_scores(LSTMModel, X_rnn_train, y_train, input_shape=input_shape)
    print(f"LSTM CV: {lstm_cv['mean_accuracy']:.4f} (+/- {lstm_cv['std_accuracy']:.4f})")
    
    # Train models
    print("\nTRAINING MODELS")
    print("="*60)
    
    # RNN
    rnn = RNNModel(hidden_units=50, dropout=0.1)
    rnn.build(input_shape=input_shape, num_classes=y_train.shape[1])
    rnn_history = rnn.train(X_rnn_train, y_train, epochs=50, batch_size=min(32, X_rnn_train.shape[0]//4), validation_split=0.2)
    rnn_metrics = rnn.evaluate(X_rnn_test, y_test)
    y_pred_rnn = np.argmax(rnn.predict(X_rnn_test), axis=1)
    
    # LSTM
    lstm = LSTMModel(hidden_units=50, dropout=0.1)
    lstm.build(input_shape=input_shape, num_classes=y_train.shape[1])
    lstm_history = lstm.train(X_rnn_train, y_train, epochs=50, batch_size=min(32, X_rnn_train.shape[0]//4), validation_split=0.2)
    lstm_metrics = lstm.evaluate(X_rnn_test, y_test)
    y_pred_lstm = np.argmax(lstm.predict(X_rnn_test), axis=1)
    
    # GAN
    print("\nGAN TRAINING")
    print("="*60)
    gen_units = min(128, X_train.shape[1]*16)
    disc_units = min(128, X_train.shape[1]*16)
    gan_batch = min(32, X_train.shape[0]//8)
    
    gan = GAN(input_dim=X_train.shape[1], generator_units=gen_units, discriminator_units=disc_units)
    gan.build_generator()
    gan.build_discriminator()
    gan.build_gan()
    gan.train(X_train, epochs=100, batch_size=gan_batch)
    
    generated = gan.generate_samples(X_test.shape[0])
    gan_X = np.concatenate((X_test, generated))
    gan_y = np.tile(np.argmax(y_test, axis=1), 2)
    gan_y_onehot = encoder.transform(gan_y)
    
    gan_classifier = GANClassifier(hidden_units=min(64, X_train.shape[1]*4))
    gan_classifier.build(input_dim=X_train.shape[1], num_classes=y_train.shape[1])
    gan_history = gan_classifier.train(gan_X, gan_y_onehot, epochs=50, batch_size=gan_batch)
    gan_metrics = gan_classifier.evaluate(X_test, y_test)
    y_pred_gan = np.argmax(gan_classifier.predict(X_test), axis=1)
    
    # Prepare for plotting
    y_test_labels = np.argmax(y_test, axis=1)
    results = {
        'rnn': {'history': rnn_history, 'metrics': rnn_metrics, 'preds': y_pred_rnn},
        'lstm': {'history': lstm_history, 'metrics': lstm_metrics, 'preds': y_pred_lstm},
        'gan': {'history': gan_history, 'metrics': gan_metrics, 'preds': y_pred_gan}
    }
    
    # Display all graphs
    print("\nGRAPHS")
    print("="*60)
    for model_name, data in results.items():
        print_metrics(model_name.upper(), data['metrics'])
        plot_learning_curves(data['history'], f"{dataset_choice.upper()} {model_name.upper()}")
        plot_confusion_matrix(y_test_labels, data['preds'], f"{dataset_choice.upper()} {model_name.upper()}")
    
    plot_metrics(['RNN', 'LSTM', 'GAN'], [rnn_metrics, lstm_metrics, gan_metrics], f'{dataset_choice}_results.png')
    print("\nGraphs complete!")
    
    # Save models
    os.makedirs(f'models/{dataset_choice}', exist_ok=True)
    dataset_dir = f'models/{dataset_choice}'
    rnn.save(f'{dataset_dir}/rnn_model.keras')
    lstm.save(f'{dataset_dir}/lstm_model.keras')
    gan.generator.save(f'{dataset_dir}/gan_generator.keras')
    gan.discriminator.save(f'{dataset_dir}/gan_discriminator.keras')
    gan_classifier.model.save(f'{dataset_dir}/gan_classifier.keras')

if __name__ == "__main__":
    main()

