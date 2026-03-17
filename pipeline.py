"""ML pipeline functions."""
import time
import matplotlib.pyplot as plt
import numpy as np

def train_all_models(X_train, X_test, y_train, y_test, X_rnn_train, X_rnn_test, input_shape, is_binary, num_classes):
    """Train all models and return metrics dict."""
    metrics = {}
    
    # TREE MODELS
    print("\nTREE MODELS")
    from lightgbm_model import LightGBMModel
    from xgboost_model import XGBoostModel
    
    lgbm = LightGBMModel()
    lgbm.build(input_dim=X_train.shape[1], num_classes=num_classes, is_binary=is_binary)
    lgbm_start = time.time()
    lgbm.train(X_train, y_train)
    lgbm_time = time.time() - lgbm_start
    metrics['lgbm'] = lgbm.evaluate(X_test, y_test)
    metrics['lgbm_time'] = lgbm_time
    print(f"  LightGBM: {lgbm_time:.2f}s Acc: {metrics['lgbm']['accuracy']:.3f}")
    
    xgb = XGBoostModel()
    xgb.build(input_dim=X_train.shape[1], num_classes=num_classes, is_binary=is_binary)
    xgb_start = time.time()
    xgb.train(X_train, y_train)
    xgb_time = time.time() - xgb_start
    metrics['xgb'] = xgb.evaluate(X_test, y_test)
    metrics['xgb_time'] = xgb_time
    print(f"  XGBoost: {xgb_time:.2f}s Acc: {metrics['xgb']['accuracy']:.3f}")
    
    # NN MODELS
    print("\nNN MODELS")
    from rnn_model import RNNModel
    from lstm_model import LSTMModel
    
    rnn = RNNModel(hidden_units=50, dropout=0.3)
    rnn.build(input_shape=input_shape, num_classes=num_classes, is_binary=is_binary)
    rnn_start = time.time()
    rnn.train(X_rnn_train, y_train, epochs=30, validation_split=0.2, verbose=0)
    rnn_time = time.time() - rnn_start
    metrics['rnn'] = rnn.evaluate(X_rnn_test, y_test)
    metrics['rnn_time'] = rnn_time
    print(f"  RNN: {rnn_time:.2f}s Acc: {metrics['rnn']['accuracy']:.3f}")
    
    lstm = LSTMModel(hidden_units=50, dropout=0.1)
    lstm.build(input_shape=input_shape, num_classes=num_classes, is_binary=is_binary)
    lstm_start = time.time()
    lstm.train(X_rnn_train, y_train, epochs=30, validation_split=0.2, verbose=0)
    lstm_time = time.time() - lstm_start
    metrics['lstm'] = lstm.evaluate(X_rnn_test, y_test)
    metrics['lstm_time'] = lstm_time
    print(f"  LSTM: {lstm_time:.2f}s Acc: {metrics['lstm']['accuracy']:.3f}")
    
    # GAN
    print("\nGAN")
    from gan_model import GAN, GANClassifier
    gen_units = min(64, X_train.shape[1]*8)
    gan = GAN(input_dim=X_train.shape[1], generator_units=gen_units, discriminator_units=gen_units)
    gan.build_generator()
    gan.build_discriminator()
    gan.build_gan()
    gan_start_gan = time.time()
    gan.train(X_train, epochs=50, batch_size=min(32, X_train.shape[0]//10))
    generated = gan.generate_samples(X_test.shape[0])
    gan_X = np.concatenate((X_test, generated), axis=0)
    
    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer()
    all_labels = np.concatenate([y_train.ravel(), y_test.ravel()])
    encoder.fit(all_labels)
    if len(y_test.shape) == 1:
        gan_y = np.tile(y_test, 2)
    else:
        gan_y = np.tile(np.argmax(y_test, axis=1), 2)
    gan_y_onehot = encoder.transform(gan_y)
    
    gan_classifier = GANClassifier(hidden_units=min(32, X_train.shape[1]*2))
    gan_classifier.build(input_dim=X_train.shape[1], num_classes=num_classes)
    gan_classifier_start = time.time()
    gan_history = gan_classifier.train(gan_X, gan_y_onehot, epochs=30, validation_split=0.2, verbose=0)
    gan_classifier_time = time.time() - gan_classifier_start
    gan_time = (time.time() - gan_start_gan) + gan_classifier_time
    metrics['gan'] = gan_classifier.evaluate(X_test, y_test)
    metrics['gan_time'] = gan_time
    print(f"  GAN: {gan_time:.2f}s Acc: {metrics['gan']['accuracy']:.3f}")
    
    return metrics

def plot_single_results(metrics):
    """Plot for single dataset."""
    models = ['LGBM', 'XGB', 'RNN', 'LSTM', 'GAN']
    accs = [metrics['lgbm']['accuracy'], metrics['xgb']['accuracy'], metrics['rnn']['accuracy'], 
            metrics['lstm']['accuracy'], metrics['gan']['accuracy']]
    times = [metrics['lgbm_time'], metrics['xgb_time'], metrics['rnn_time'], 
             metrics['lstm_time'], metrics['gan_time']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    bars1 = ax1.bar(models, accs, alpha=0.8, color=['green', 'blue', 'red', 'orange', 'purple'])
    ax1.set_title('Test Accuracy')
    ax1.set_ylim(0, 1)
    for bar, acc in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center')
    
    bars2 = ax2.bar(models, times, alpha=0.8, color=['green', 'blue', 'red', 'orange', 'purple'])
    ax2.set_title('Training Time (s)')
    ax2.set_yscale('log')
    for bar, t in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width()/2, t * 1.1, f'{t:.2f}s', ha='center')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("model_comparison.png saved")
