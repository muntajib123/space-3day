# scripts/train_lstm.py
"""
High-accuracy LSTM for 3-day forecast
- Uses deeper LSTM with more units
- Learning rate scheduler
- Early stopping
- Target: MSE < 1
"""

import numpy as np
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    LearningRateScheduler,
)
import tensorflow as tf

DATA_NPZ = Path("data/dataset.npz")
MODEL_PATH = Path("models/seq2seq_lstm.h5")

def build_model(in_steps, n_features, n_units=256, dropout=0.3):
    inp = Input(shape=(in_steps, n_features))
    x = LSTM(n_units, return_sequences=True)(inp)
    x = Dropout(dropout)(x)
    x = LSTM(n_units // 2, return_sequences=True)(x)
    x = Dropout(dropout)(x)
    x = LSTM(n_units // 4, return_sequences=True)(x)
    out = TimeDistributed(Dense(n_features))(x)

    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])
    return model

def lr_schedule(epoch, lr):
    # reduce LR after 20, 40 epochs
    if epoch > 40:
        return lr * 0.2
    elif epoch > 20:
        return lr * 0.5
    return lr

def main(batch_size=32, epochs=80):
    data = np.load(DATA_NPZ)
    X, Y = data["X"], data["Y"]
    in_steps = int(data["in_steps"])
    out_steps = int(data["out_steps"])
    n_features = X.shape[2]

    print("Train data:", X.shape, Y.shape)

    # Train/val split
    split = int(0.85 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    model = build_model(in_steps, n_features)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(str(MODEL_PATH), save_best_only=True,
                        monitor="val_loss", verbose=1),
        EarlyStopping(patience=12, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(patience=6, factor=0.5, monitor="val_loss", verbose=1),
        LearningRateScheduler(lr_schedule)
    ]

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )

    model.save(MODEL_PATH)
    print(f"✅ Model saved -> {MODEL_PATH}")
    final_loss = history.history["val_loss"][-1]
    print(f"Final Validation MSE: {final_loss:.4f}")

if __name__ == "__main__":
    main()
