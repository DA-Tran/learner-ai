# Learner AI - ML Models Learning Project

A **personal learning project** to understand and compare 5 AI/ML models on UCI classification datasets.

##  Purpose
- **Learn core concepts**: How neural networks, GANs, and tree ensembles work under the hood
- **Compare performance**: CV/Test accuracy + train time across datasets
- **Production-ready patterns**: Clean code, testing suite, data preprocessing
- **Interactive exploration**: Run specific model-dataset combinations

##  Quick Start

```bash
git clone <repo>
cd learner-ai
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install deps (includes sklearn env var)
pip install tensorflow scikit-learn lightgbm xgboost

# Interactive run (recommended)
./run_clean.sh

# Full suite test
./run_ml_suite.sh
```

##  Structure

```
├── Main.py              # Core: model training + interactive CLI
├── data_utils.py        # UCI datasets (iris/heart/breast/wine + large)
├── rnn_model.py         # RNN implementation
├── lstm_model.py        # LSTM with CV
├── gan_model.py         # GAN generator + classifier
├── lightgbm_model.py    # LGBM baseline
├── xgboost_model.py     # XGB baseline
├── tests/run_all_tests.py # Time-based tests (iris/heart/breast/wine <660s)
├── run_clean.sh         # Clean interactive launcher
├── run_ml_suite.sh      # Full ML pipeline
├── plot_utils.py        # Dynamic comparison PNGs
├── TODO.md              # Progress tracking
└── results/             # *_comparison.png + JSON
```

##  Models Implemented

| Model        | Type              | Strengths           | Datasets |
|--------------|-------------------|---------------------|----------|
| **RNN**      | Recurrent NN      | Sequential data     | All      |
| **LSTM**     | Advanced RNN      | Long dependencies   | All      |
| **GAN**      | Generative        | Data augmentation   | All      |
| **LightGBM** | Gradient Boosting | Speed + accuracy    | Baseline |
| **XGBoost**  | Gradient Boosting | Production standard | Baseline |

##  Key Datasets

**Small (quick)**: iris (4 feats), heart/breast/wine (binary/multi-class)  
**Medium**: phishing, mushroom  
**Large**: gendername (1M+ rows, RobustScaler)

##  Features

- **Per-model timing**: `time completed: 24.08s` after each model
- **Time-based tests**: PASS if total <660s (iris-heart-breast-wine dataset)
- **Interactive CLI**: Pick dataset + models → instant PNG
- **JSON artifacts**: Accurate times/accuracies for tests
- **Robust scaling**: RobustScaler for large/inf/NaN datasets
- **Clean outputs**: Silent TF warnings

## Test Suite

```bash
./run_ml_suite.sh
```

 **Expected** (time PASS):
```
TOTAL TIME: 20.9s vs limit 660s (PASS) x 5 models
SUMMARY: rnn: True, lstm: True...
```

##  Example Output

```
RNN done
time completed: 24.08s
iris_comparison.png saved
```

**PNG**: Green bars = test/CV accuracy per model

##  Tech Stack

```
Python 3.8+ • TensorFlow/Keras • scikit-learn • LightGBM • XGBoost
matplotlib • pandas • numpy • subprocess testing
```

##  Learning Goals Achieved

1.  **Neural Nets**: Backprop, layers, optimizers
2.  **Recurrent**: RNN vs LSTM gates/memory
3.  **GANs**: Generator vs discriminator training
4.  **Trees**: Gradient boosting, feature importance
5.  **Scaling**: RobustScaler for outliers/inf
6.  **Testing**: Time-based suite (no flaky acc parsing)
7.  **CLI**: Interactive model selection + timing

## Next Steps (TODO.md)

- Add Transformer model
- Hyperparameter tuning
- Docker + CI tests

