"""Main orchestrator for training and evaluating multiple neural network models."""

import numpy as np
import pandas as pd
from data_utils import load_and_preprocess_iris, reshape_for_rnn
from rnn_model import RNNModel
from lstm_model import LSTMModel
from gan_model import GAN, GANClassifier
from evaluation import print_metrics, plot_confusion_matrix, plot_learning_curves, plot_metrics
from cross_validation import get_cv_scores


def main():
    """Main training pipeline."""
    
    print("="*60)
    print("Loading and preprocessing data...")
    print("="*60)
    
    # Load data
    X, X_test, y, y_test, scaler, encoder = load_and_preprocess_iris()
    X_rnn, X_test_rnn = reshape_for_rnn(X, X_test)
    
    print(f"Full dataset shape: {X.shape}")
    
    # ==================== CROSS-VALIDATION ====================
    print("\n" + "="*60)
    print("CROSS-VALIDATION (5-fold)")
    print("="*60)
    
    rnn_cv = get_cv_scores(RNNModel, X_rnn, y, input_shape=(1, 4), cv_folds=5)
    print(f"RNN CV: {rnn_cv['mean_accuracy']:.4f} (+/- {rnn_cv['std_accuracy']:.4f})")
    
    lstm_cv = get_cv_scores(LSTMModel, X_rnn, y, input_shape=(1, 4), cv_folds=5)
    print(f"LSTM CV: {lstm_cv['mean_accuracy']:.4f} (+/- {lstm_cv['std_accuracy']:.4f})")
    
    # ==================== TRAIN AND EVALUATE ====================
    print("\n" + "="*60)
    print("Training individual models...")
    print("="*60)
    
    # RNN
    rnn = RNNModel(hidden_units=50, dropout=0.1)
    rnn.build(input_shape=(1, 4), num_classes=3)
    rnn_history = rnn.train(X_rnn, y, epochs=50, batch_size=10, validation_split=0.2)
    rnn_metrics = rnn.evaluate(X_test_rnn, y_test)
    print_metrics("RNN", rnn_metrics)
    plot_learning_curves(rnn_history, "RNN")
    
    y_pred_rnn = np.argmax(rnn.predict(X_test_rnn), axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    plot_confusion_matrix(y_test_labels, y_pred_rnn, "RNN")
    
    # LSTM
    lstm = LSTMModel(hidden_units=50, dropout=0.1)
    lstm.build(input_shape=(1, 4), num_classes=3)
    lstm_history = lstm.train(X_rnn, y, epochs=50, batch_size=10, validation_split=0.2)
    lstm_metrics = lstm.evaluate(X_test_rnn, y_test)
    print_metrics("LSTM", lstm_metrics)
    plot_learning_curves(lstm_history, "LSTM")
    
    y_pred_lstm = np.argmax(lstm.predict(X_test_rnn), axis=1)
    plot_confusion_matrix(y_test_labels, y_pred_lstm, "LSTM")
    
    # GAN (no CV for GAN - complex)
    print("\n" + "="*60)
    print("Training GAN...")
    print("="*60)
    
    gan = GAN(input_dim=4, generator_units=8, discriminator_units=8)
    gan.build_generator()
    gan.build_discriminator()
    gan.build_gan()
    gan.train(X, epochs=100, batch_size=32)
    
    generated = gan.generate_samples(X_test.shape[0])
    gan_X = np.concatenate((X_test, generated))
    gan_y = np.concatenate((y_test.argmax(axis=1), y_test.argmax(axis=1)))
    gan_y_onehot = np.eye(3)[gan_y]
    
    gan_classifier = GANClassifier(hidden_units=10)
    gan_classifier.build(input_dim=4, num_classes=3)
    gan_history = gan_classifier.train(gan_X, gan_y_onehot, epochs=50, batch_size=10)
    gan_metrics = gan_classifier.evaluate(X_test, y_test)
    print_metrics("GAN Classifier", gan_metrics)
    plot_learning_curves(gan_history, "GAN Classifier")
    
    y_pred_gan = np.argmax(gan_classifier.predict(X_test), axis=1)
    plot_confusion_matrix(y_test_labels, y_pred_gan, "GAN Classifier")
    
    # Summary plot
    all_metrics = [rnn_metrics, lstm_metrics, gan_metrics]
    plot_metrics(['RNN', 'LSTM', 'GAN'], all_metrics)
    
    print("\nCV and test results saved as PNG files. Task complete!")
    
    # Save models
    rnn.save('models/rnn_model.keras')
    lstm.save('models/lstm_model.keras')
    gan.generator.save('models/gan_generator.keras')
    gan.discriminator.save('models/gan_discriminator.keras')
    gan_classifier.model.save('models/gan_classifier.keras')


if __name__ == "__main__":
    main()

