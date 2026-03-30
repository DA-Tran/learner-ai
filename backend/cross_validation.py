from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

def get_cv_scores(model_class, X, y, input_shape, is_binary=False, cv_folds=5, epochs=30, hidden_units=50):
    """
    Custom CV for Keras models with binary/multi-class support.
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    # Use passed is_binary param (dataset-global level - avoids fold mis-detection)
    is_binary_fold = is_binary
    num_classes = 1 if is_binary else len(np.unique(np.argmax(y, axis=1) if y.ndim > 1 else y))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"CV Fold {fold+1}/{cv_folds}")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Build model
        model = model_class(hidden_units=hidden_units)
        model.build(input_shape=input_shape, num_classes=num_classes, is_binary=is_binary_fold)
        
        # Train
        history = model.train(X_train_fold, y_train_fold, epochs=epochs, batch_size=10, validation_split=0.0, verbose=0)
        
        # Evaluate
        y_pred_fold = model.predict(X_val_fold)
        if is_binary_fold:
            y_pred_labels = (y_pred_fold.flatten() > 0.5).astype(int)
            y_val_labels = (y_val_fold.flatten() > 0.5).astype(int) if y_val_fold.ndim > 1 else y_val_fold
        else:
            y_pred_labels = np.argmax(y_pred_fold, axis=1)
            y_val_labels = np.argmax(y_val_fold, axis=1)
        
        score = accuracy_score(y_val_labels, y_pred_labels)
        cv_scores.append(score)
    
    return {
        'mean_accuracy': np.mean(cv_scores),
        'std_accuracy': np.std(cv_scores),
        'scores': cv_scores
    }

