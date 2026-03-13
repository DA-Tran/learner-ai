"""Main orchestrator for training and evaluating multiple neural network models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import load_and_preprocess_iris, reshape_for_rnn
from rnn_model import RNNModel
from lstm_model import LSTMModel
from gan_model import GAN, GANClassifier


def print_metrics(model_name, metrics):
    """Print model evaluation metrics."""
    print(f"\n{model_name} Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")


def plot_metrics(model_names, metrics_list, filename='results.png'):
    """Plot bar chart for model metrics."""
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1']
    
    for i, key in enumerate(metrics_keys):
        values = [m[key] for m in metrics_list]
        ax.bar(x + i*width, values, width, label=key.capitalize())
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Graph saved as {filename}")


def main():
    """Main training pipeline."""
    
    print("="*60)
    print("Loading and preprocessing data...")
    print("="*60)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, encoder = load_and_preprocess_iris()
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Reshape for RNN/LSTM
    X_train_rnn, X_test_rnn = reshape_for_rnn(X_train, X_test)
    print(f"RNN-reshaped training shape: {X_train_rnn.shape}")
    
    # ==================== RNN MODEL ====================
    print("\n" + "="*60)
    print("Training RNN Model...")
    print("="*60)
    
    rnn = RNNModel(hidden_units=50, dropout=0.1, learning_rate=0.001)
    rnn.build(input_shape=(1, 4), num_classes=3)
    rnn.train(X_train_rnn, y_train, epochs=50, batch_size=10, validation_split=0.2, verbose=0)
    rnn_metrics = rnn.evaluate(X_test_rnn, y_test)
    print_metrics("RNN", rnn_metrics)
    
    # ==================== LSTM MODEL ====================
    print("\n" + "="*60)
    print("Training LSTM Model...")
    print("="*60)
    
    lstm = LSTMModel(hidden_units=50, dropout=0.1, learning_rate=0.001)
    lstm.build(input_shape=(1, 4), num_classes=3)
    lstm.train(X_train_rnn, y_train, epochs=50, batch_size=10, validation_split=0.2, verbose=0)
    lstm_metrics = lstm.evaluate(X_test_rnn, y_test)
    print_metrics("LSTM", lstm_metrics)
    
    # ==================== GAN MODEL ====================
    print("\n" + "="*60)
    print("Training GAN Model...")
    print("="*60)
    
    gan = GAN(input_dim=4, generator_units=8, discriminator_units=8, learning_rate=0.0002)
    gan.build_generator()
    gan.build_discriminator()
    gan.build_gan()
    gan.train(X_train, epochs=100, batch_size=32, verbose_interval=50)
    
    print("\nGenerating augmented data with GAN...")
    generated_samples = gan.generate_samples(X_test.shape[0])
    gan_X = np.concatenate((X_test, generated_samples), axis=0)
    gan_y = np.concatenate((y_test.argmax(axis=1), y_test.argmax(axis=1)), axis=0)
    gan_y_onehot = np.eye(3)[gan_y]
    
    print("Training classifier on GAN-augmented data...")
    gan_classifier = GANClassifier(hidden_units=10, learning_rate=0.001)
    gan_classifier.build(input_dim=4, num_classes=3)
    gan_classifier.train(gan_X, gan_y_onehot, epochs=50, batch_size=10, verbose=0)
    
    gan_metrics = gan_classifier.evaluate(X_test, y_test)
    print_metrics("GAN-based Classifier", gan_metrics)
    
    # ==================== SUMMARY ====================
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    
    summary_data = {
        'Model': ['RNN', 'LSTM', 'GAN Classifier'],
        'Accuracy': [rnn_metrics['accuracy'], lstm_metrics['accuracy'], gan_metrics['accuracy']],
        'Precision': [rnn_metrics['precision'], lstm_metrics['precision'], gan_metrics['precision']],
        'Recall': [rnn_metrics['recall'], lstm_metrics['recall'], gan_metrics['recall']],
        'F1-Score': [rnn_metrics['f1'], lstm_metrics['f1'], gan_metrics['f1']]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.to_string(index=False))
    
    best_model = summary_df.loc[summary_df['Accuracy'].idxmax(), 'Model']
    best_accuracy = summary_df['Accuracy'].max()
    print(f"\nBest Model: {best_model} with Accuracy: {best_accuracy:.4f}")
    
    # ==================== PLOT RESULTS ====================
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    
    all_metrics = [rnn_metrics, lstm_metrics, gan_metrics]
    plot_metrics(['RNN', 'LSTM', 'GAN Classifier'], all_metrics)
    
    # ==================== SAVE MODELS ====================
    print("\n" + "="*60)
    print("Saving models...")
    print("="*60)
    
    rnn.save('models/rnn_model.h5')
    lstm.save('models/lstm_model.h5')
    gan.generator.save('models/gan_generator.h5')
    gan.discriminator.save('models/gan_discriminator.h5')
    gan_classifier.model.save('models/gan_classifier.h5')
    
    print("Models saved to 'models/' directory")


if __name__ == "__main__":
    main()

