# scripts/retrain.py
import os, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "noaa_all_parsed_enriched.csv"
MODEL_PATH = ROOT / "models" / "seq2seq_lstm.h5"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"

# Hyperparams - tune as needed
IN_STEPS = 24   # same as used in prep_dataset
OUT_STEPS = 24
FEATURES = ["kp","solar_radiation","radio_blackout"]
BATCH = 64
EPOCHS = 80
LATENT = 64
VAL_SPLIT = 0.1

def load_df():
    df = pd.read_csv(DATA_CSV, parse_dates=["datetime"])
    # ensure sorted
    df = df.sort_values("datetime").reset_index(drop=True)
    # only keep needed columns and drop rows with nan kp
    df = df[["datetime"] + FEATURES].dropna(subset=["kp"])
    return df

def make_sequences(values, in_steps=IN_STEPS, out_steps=OUT_STEPS):
    X, Y = [], []
    L = len(values)
    for i in range(0, L - in_steps - out_steps + 1):
        X.append(values[i:i+in_steps])
        Y.append(values[i+in_steps:i+in_steps+out_steps])
    return np.array(X), np.array(Y)

def build_model(n_features, in_steps=IN_STEPS, out_steps=OUT_STEPS, latent=LATENT):
    model = Sequential()
    model.add(LSTM(latent, input_shape=(in_steps, n_features), return_sequences=False))
    model.add(RepeatVector(out_steps))
    model.add(LSTM(latent, return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model

def main():
    print("Loading data...")
    df = load_df()
    values = df[FEATURES].values.astype("float32")
    # scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    # sequences
    X, Y = make_sequences(scaled)
    print("Shapes:", X.shape, Y.shape)
    # shuffle & split
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, Y = X[idx], Y[idx]
    split = int(len(X)*(1-VAL_SPLIT))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    # build model
    model = build_model(n_features=len(FEATURES))
    # callbacks
    cb = [
        EarlyStopping(patience=8, restore_best_weights=True),
        ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6),
        ModelCheckpoint(str(MODEL_PATH), save_best_only=True, monitor="val_loss")
    ]
    # train
    history = model.fit(X_train, Y_train,
                        validation_data=(X_val, Y_val),
                        epochs=EPOCHS, batch_size=BATCH, callbacks=cb, verbose=2)
    # save scaler
    joblib.dump(scaler, SCALER_PATH)
    # final metrics (best val loss)
    val_losses = history.history.get("val_loss", [])
    final_val_mse = float(val_losses[-1]) if val_losses else None
    metrics = {"final_val_mse": final_val_mse, "history": {k: v for k,v in history.history.items()}}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Training finished. Final val MSE:", final_val_mse)
    print("Model saved to:", MODEL_PATH)
    print("Scaler saved to:", SCALER_PATH)
    print("Metrics saved to:", METRICS_PATH)

if __name__ == "__main__":
    main()
