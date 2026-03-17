# AI Learning Project

A personal project created to understand how neural networks work. This explores five models: deep learning (RNN, LSTM, GAN) + tree ensembles (LGBM, XGB) applied to UCI datasets (iris, heart, breast, wine, phishing websites, mushroom, gendername).

## What is This?

Hands-on implementations of fundamental models:

**RNN** - Processes sequences remembering past inputs  
**LSTM** - Improved RNN for long-term dependencies  
**GAN** - Generator vs discriminator (creates + classifies fake data)  
**LGBM/XGB** - Fast tree ensemble baselines

Trained on classification tasks comparing all model performance.

## Project Files

```
Main.py          - Runs models + creates summary PNGs (selective training)
data_utils.py    - Loads UCI datasets (iris/heart/breast/wine/phishing/mushroom/gendername)
rnn_model.py     - RNN class
lstm_model.py    - LSTM class
gan_model.py     - GAN + classifier
lightgbm_model.py| Tree models (LGBM/XGB fast baselines)
xgboost_model.py|
plot_utils.py    - Dynamic plots (selective/CV safe)
run_clean.sh     - Clean interactive run script
```

## How to Run

```bash
chmod +x run_clean.sh
./run_clean.sh 2  # ALL datasets → ALL_RESULTS_SUMMARY.png
```

**Interactive**:
```
1. iris → lgbm → single green bar PNG
1. iris → rnn,gan → RNN-CV + GAN bars
```

**Output**: CV/Test accuracies + performance PNGs (warnings normal).

## Technologies Used

Python • TensorFlow/Keras • scikit-learn • LightGBM • XGBoost • matplotlib

## Key Takeaways

- Neural networks adjust weights via backpropagation
- Tree models (LGBM/XGB) fastest baselines
- Different architectures suit different tasks  
- Data preprocessing essential
- GAN data augmentation boosts performance
- Multi-metric evaluation > accuracy alone

## Why These Models?

**RNN**: Sequential learning  
**LSTM**: Long-term dependencies  
**GAN**: Generative concepts  
**LGBM/XGB**: Production tree baselines
