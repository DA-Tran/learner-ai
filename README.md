# AI Learning Project

A personal project created to understand how neural networks work. This explores three different deep learning models (RNN, LSTM, and GAN) applied to UCI datasets (iris, heart, breast, wine).

## What is This?

Hands-on implementations of fundamental neural networks:

**RNN** - Processes sequences remembering past inputs
**LSTM** - Improved RNN for long-term dependencies  
**GAN** - Generator vs discriminator (creates + classifies fake data)

Trained on classification tasks comparing RNN/LSTM/GAN performance.

## Project Files

```
Main.py          - Runs models + creates summary PNG
data_utils.py    - Loads UCI datasets (iris/heart/breast/wine)
rnn_model.py     - RNN class
lstm_model.py    - LSTM class
gan_model.py     - GAN + classifier
run_clean.sh     - Clean run script
```

## How to Run

```bash
chmod +x run_clean.sh
./run_clean.sh 2  # All 4 datasets → ALL_RESULTS_SUMMARY.png
```

**Output**: CV/Test accuracies + 4-panel performance PNG.

## Technologies Used

Python • TensorFlow/Keras • scikit-learn • matplotlib/seaborn

## Key Takeaways

Neural networks adjust weights via backpropagation
Different architectures suit different tasks
Data preprocessing essential for results
GAN data augmentation boosts classifier performance
Multi-metric evaluation > accuracy alone

## Why These Models?

**RNN**: Sequential data learning
**LSTM**: Long-term memory
**GAN**: Generative adversarial concepts

---

*Learning deep learning fundamentals through comparison*
