"""
evaluate_model.py
- Loads trained model (trained_model.pkl)
- Loads processed test data (processed_data.csv) and recomputes test split
- Computes R2, MSE, MAE on held-out test set and writes metrics to ../results/metrics.txt
- Generates a plot comparing predicted vs actual SOH saved to ../results/soh_predictions.png
Usage:
    python src/evaluate_model.py --input "../results/processed_data.csv" --model "../trained_model.pkl" --test-size 0.2 --random-seed 42 --threshold 0.6
"""
import argparse, os, json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

def main(args):
    base = os.path.dirname(__file__)
    inp = os.path.join(base, args.input)
    model_path = os.path.join(base, args.model)
    if not os.path.exists(inp):
        raise FileNotFoundError(f"Processed data not found: {inp}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    df = pd.read_csv(inp)
    y = df['SOH'].values
    X = df.drop(columns=['SOH']).values

    # reload model and scaler
    saved = joblib.load(model_path)
    model = saved['model']
    scaler = saved['scaler']
    threshold = float(saved.get('threshold', args.threshold))

    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=args.test_size, random_state=args.random_seed)

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    # classification using threshold
    status_true = np.where(y_test < threshold, "Problem", "Healthy")
    status_pred = np.where(preds < threshold, "Problem", "Healthy")
    # classification accuracy
    class_acc = (status_true == status_pred).mean()

    # save metrics
    os.makedirs(os.path.join(base, "..", "results"), exist_ok=True)
    metrics_path = os.path.join(base, "..", "results", "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"R2: {r2:.6f}\\n")
        f.write(f"MSE: {mse:.6f}\\n")
        f.write(f"MAE: {mae:.6f}\\n")
        f.write(f"Threshold: {threshold}\\n")
        f.write(f"Classification accuracy (threshold rule): {class_acc:.6f}\\n")
    print(f"Saved metrics to {metrics_path}")

    # plot predicted vs actual
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, preds, alpha=0.6)
    minv = min(y_test.min(), preds.min())
    maxv = max(y_test.max(), preds.max())
    plt.plot([minv, maxv], [minv, maxv], linestyle='--')
    plt.xlabel("Actual SOH")
    plt.ylabel("Predicted SOH")
    plt.title("Predicted vs Actual SOH")
    outplot = os.path.join(base, "..", "results", "soh_predictions.png")
    plt.savefig(outplot, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {outplot}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="../results/processed_data.csv")
    p.add_argument("--model", type=str, default="../trained_model.pkl")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.6)
    args = p.parse_args()
    main(args)
