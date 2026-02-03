# AI Learning Project

A personal project created to understand how neural networks work. This explores three different deep learning models (RNN, LSTM, and GAN) applied to the Iris dataset.

## What is This?

This project is a hands-on learning exercise building working implementations of three fundamental neural network architectures:

- **RNN (Recurrent Neural Network)** - Processes sequences of data by remembering past inputs
- **LSTM (Long Short-Term Memory)** - An improved RNN that better remembers long-term dependencies
- **GAN (Generative Adversarial Network)** - Two networks competing: one generates fake data, one detects fakes

All three models are trained to classify iris flower species (3 categories) based on 4 flower measurements.

## Project Files

```
Main.py              - Runs all three models and compares results
data_utils.py        - Loads and prepares the Iris dataset
rnn_model.py         - RNN implementation
lstm_model.py        - LSTM implementation
gan_model.py         - GAN implementation with data augmentation
```

## How to Run

```bash
python Main.py
```

The script will:
1. Load and prepare the Iris dataset
2. Train each model (RNN, LSTM, GAN)
3. Evaluate and compare performance
4. Show which model performed best

## Technologies Used

- Python 3.13
- TensorFlow/Keras (neural network framework)
- scikit-learn (data preprocessing and metrics)
- NumPy & Pandas (data handling)

## Key Takeaways

- Neural networks learn by adjusting weights through backpropagation
- Different architectures excel at different tasks
- Proper data preprocessing is crucial for good results
- Evaluation metrics beyond accuracy give better insight into model performance
- GANs demonstrate interesting adversarial learning concepts

## Why These Models?

These three architectures represent important concepts:
- **Sequential learning** with RNN
- **Long-term memory** with LSTM  
- **Generative modeling** with GAN

---

*A learning project to understand the fundamentals of deep learning.*
