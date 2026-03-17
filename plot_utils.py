"""Plot utilities for ML pipeline."""
import matplotlib.pyplot as plt
import numpy as np

def plot_single_dataset_comparison(metrics, dataset_name):
    """Plot accuracy & time for single dataset (RNN, LSTM, GAN, LGBM, XGB)."""
    models = ['RNN', 'LSTM', 'GAN', 'LGBM', 'XGB']
    # Handle partial results (single model)
    model_keys = {
        'LGBM': ('lgbm_acc', 'lgbm_time'),
        'XGB': ('xgb_acc', 'xgb_time'),
        'RNN': ('rnn_test', 'rnn_time'),
        'LSTM': ('lstm_test', 'lstm_time'),
        'GAN': ('gan_test', 'gan_time')
    }
    accs = [metrics.get(acc_key, None) for acc_key, _ in model_keys.values()]
    times = [metrics.get(time_key, None) for _, time_key in model_keys.values()]
    models = list(model_keys.keys())
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # CV - show available NN CV scores (rnn only OK)
    cv_labels = []
    cv_values = []
    cv_colors = []
    
    if 'rnn_cv' in metrics:
        cv_labels.append('RNN-CV')
        cv_values.append(metrics['rnn_cv'])
        cv_colors.append('red')
    if 'lstm_cv' in metrics:
        cv_labels.append('LSTM-CV')
        cv_values.append(metrics['lstm_cv'])
        cv_colors.append('orange')
    
    if cv_labels:
        ax3.bar(cv_labels, cv_values, alpha=0.8, color=cv_colors)
        ax3.set_title(f'{dataset_name.upper()} Cross-Validation')
        ax3.set_ylim(0, 1)
        for i, v in enumerate(cv_values):
            ax3.text(i, v+0.01, f'{v:.3f}', ha='center', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No NN CV\n(Tree models)', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Cross-Validation')
    
    colors = ['green', 'blue', 'red', 'orange', 'purple']
    
    # Accuracy
    # Filter None for bar heights
    valid_accs = [a for a in accs if a is not None]
    valid_models = [m for m, a in zip(models, accs) if a is not None]
    valid_colors = [c for c, a in zip(colors, accs) if a is not None]
    
    if valid_accs:
        bars1 = ax1.bar(valid_models, valid_accs, alpha=0.8, color=valid_colors)
        ax1.set_title(f'{dataset_name.upper()} Test Accuracy ({len(valid_accs)} models)')
        ax1.set_ylim(0, 1)
        for bar, acc in zip(bars1, valid_accs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{acc:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No accuracy data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'{dataset_name.upper()} Test Accuracy')
    
    # Training time (log scale) - filter None
    valid_times = [t for t in times if t is not None]
    valid_models_t = [m for m, t in zip(models, times) if t is not None]
    valid_colors_t = [c for c, t in zip(colors, times) if t is not None]
    
    if valid_times:
        bars2 = ax2.bar(valid_models_t, valid_times, alpha=0.8, color=valid_colors_t)
        ax2.set_title(f'Training Time ({len(valid_times)} models)')
        ax2.set_yscale('log')
        for bar, t in zip(bars2, valid_times):
            ax2.text(bar.get_x() + bar.get_width()/2., t * 1.1, f'{t:.2f}s', 
                    ha='center', va='bottom', rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No timing data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Training Time')
    
    plt.suptitle(f'{dataset_name.upper()} - 5 Models Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"{dataset_name}_comparison.png saved")

def plot_all_datasets_summary(results_data):
    """Plot summary across all datasets (LGBM, XGB, RNN, LSTM, GAN)."""
    if not results_data:
        print("No results data")
        return
    
    successful_datasets = [r['dataset'] for r in results_data]
    n_datasets = len(results_data)
    
    # All 5 models acc
models = ['RNN', 'LSTM', 'GAN', 'LGBM', 'XGB']
    rnn_acc = [r['rnn'] for r in results_data]
    lstm_acc = [r['lstm'] for r in results_data]
    gan_acc = [r['gan'] for r in results_data]
    lgbm_acc = [r['lgbm'] for r in results_data]
    xgb_acc = [r['xgb'] for r in results_data]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    x = np.arange(n_datasets)
    width = 0.16
    
    # Test acc all models
    axes[0,0].bar(x - 2*width, lgbm_acc, width, label='LGBM', alpha=0.8, color='green')
    axes[0,0].bar(x - width, xgb_acc, width, label='XGB', alpha=0.8, color='blue')
    axes[0,0].bar(x, rnn_acc, width, label='RNN', alpha=0.8, color='red')
    axes[0,0].bar(x + width, lstm_acc, width, label='LSTM', alpha=0.8, color='orange')
    axes[0,0].bar(x + 2*width, gan_acc, width, label='GAN', alpha=0.8, color='purple')
    axes[0,0].set_title('Test Accuracy (5 Models)')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(successful_datasets, rotation=45)
    axes[0,0].legend()
    axes[0,0].set_ylim(0, 1)
    
    # Average per model
    avg_acc = [np.mean(lgbm_acc), np.mean(xgb_acc), np.mean(rnn_acc), np.mean(lstm_acc), np.mean(gan_acc)]
    axes[0,1].bar(models, avg_acc, alpha=0.8, color=['green', 'blue', 'red', 'orange', 'purple'])
    axes[0,1].set_title('Average Test Accuracy')
    axes[0,1].set_ylim(0, 1)
    for i, v in enumerate(avg_acc):
        axes[0,1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # RNN/LSTM CV
    rnn_cv = [r['rnn_cv'] for r in results_data]
    lstm_cv = [r['lstm_cv'] for r in results_data]
    x_cv = np.arange(n_datasets)
    width_cv = 0.4
    axes[1,0].bar(x_cv - width_cv/2, rnn_cv, width_cv, label='RNN CV', alpha=0.8, color='red')
    axes[1,0].bar(x_cv + width_cv/2, lstm_cv, width_cv, label='LSTM CV', alpha=0.8, color='orange')
    axes[1,0].set_title('NN Cross-Validation')
    axes[1,0].set_xticks(x_cv)
    axes[1,0].set_xticklabels(successful_datasets, rotation=45)
    axes[1,0].legend()
    axes[1,0].set_ylim(0, 1)
    
    # Best model per dataset
    best_accs = [max(r['lgbm'], r['xgb'], r['rnn'], r['lstm'], r['gan']) for r in results_data]
    axes[1,1].bar(successful_datasets, best_accs, alpha=0.8, color='gold')
    axes[1,1].set_title('Best Model per Dataset')
    axes[1,1].set_ylim(0, 1)
    for i, v in enumerate(best_accs):
        axes[1,1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', rotation=45)
    
    plt.tight_layout()
    plt.savefig('ALL_RESULTS_5MODELS.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("ALL_RESULTS_5MODELS.png saved (LGBM/XGB/RNN/LSTM/GAN)")
