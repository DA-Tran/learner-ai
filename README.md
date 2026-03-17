# AI Learning Project

A personal project created to understand how neural networks work. This explores three different deep learning models (RNN, LSTM, and GAN) applied to UCI datasets (iris, heart, breast, wine, phishing websites, mushroom, gendername).

## What is This?

Hands-on implementations of fundamental neural networks:

**RNN** - Processes sequences remembering past inputs  
**LSTM** - Improved RNN for long-term dependencies  
**GAN** - Generator vs discriminator (creates + classifies fake data)

Trained on classification tasks comparing RNN/LSTM/GAN performance.

## Project Files

```
Main.py          - Runs models + creates summary PNGs (selective training ✓)
data_utils.py    - Loads UCI datasets (iris/heart/breast/wine/phishing/mushroom/gendername)
rnn_model.py     - RNN class
lstm_model.py    - LSTM class
gan_model.py     - GAN + classifier
lightgbm_model.py| Tree baselines (LGBM/XGB) ✓
xgboost_model.py |
plot_utils.py    - Dynamic plots ✓ selective ✓
run_clean.sh     - Clean run script ✓
```

## How to Run

```bash
chmod +x run_clean.sh
./run_clean.sh 2  # ALL datasets → ALL_RESULTS_SUMMARY.png
```

**Interactive (NEW)**:
```
1. iris → lgbm → single bar PNG ✓
1. iris → rnn,gan → RNN-CV + GAN ✓
```

**Output**: CV/Test accuracies + performance PNGs. Warnings normal (TF Windows).

## Technologies Used

Python • TensorFlow/Keras • scikit-learn • LightGBM • XGBoost • matplotlib

## Key Takeaways

- Neural networks adjust weights via backpropagation
- Different architectures suit different tasks
- Data preprocessing essential for results
- GAN data augmentation boosts classifier performance  
- Multi-metric evaluation > accuracy alone
- **Selective training** speeds experimentation ✓

## Why These Models?

**RNN**: Sequential data learning  
**LSTM**: Long-term memory  
**GAN**: Generative adversarial concepts  
**LGBM/XGB**: Fast tree baselines
