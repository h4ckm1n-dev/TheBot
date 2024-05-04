import os
from typing import Any, Dict, Tuple

import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_combine_data(directory: str) -> pd.DataFrame:
    """Load all CSV files from the specified directory and combine them into one DataFrame."""
    frames = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            frames.append(df)
    combined_df = pd.concat(frames, ignore_index=True)
    return combined_df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing data."""
    df["return"] = df["Close"].pct_change()
    df["moving_average"] = df["Close"].rolling(window=10).mean()
    df["volatility"] = df["Close"].rolling(window=10).std()
    df.dropna(inplace=True)
    return df


def label_data(df: pd.DataFrame) -> pd.DataFrame:
    """Define labels for training the model."""
    df["future_return"] = df["Close"].shift(-1) / df["Close"] - 1
    df["label"] = 0
    df.loc[df["future_return"] > 0.01, "label"] = 1
    df.loc[df["future_return"] < -0.01, "label"] = -1
    df.dropna(inplace=True)
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[Any, Any, Any, Any]:
    """Prepare data for model training."""
    features = df[["return", "moving_average", "volatility"]]
    labels = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(X_train: Any, y_train: Any) -> RandomForestClassifier:
    """Train a machine learning model on the training data."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier, X_test: Any, y_test: Any
) -> Dict[str, float]:
    """Evaluate the model on the test set."""
    predictions = model.predict(X_test)
    results = {
        "classification_report": classification_report(y_test, predictions),
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions, average="weighted"),
    }
    return results


def perform_cross_validation(
    model: RandomForestClassifier, X_train: Any, X_test: Any, y_train: Any, y_test: Any
) -> float:
    """Perform cross-validation and return the average accuracy."""
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    scores = cross_val_score(model, X, y, cv=5)
    return scores.mean()


def save_data_h5(X_train: Any, X_test: Any, y_train: Any, y_test: Any) -> None:
    """Save datasets to HDF5."""
    with h5py.File("./data.h5", "w") as hf:
        hf.create_dataset("X_train", data=X_train)
        hf.create_dataset("X_test", data=X_test)
        hf.create_dataset("y_train", data=y_train)
        hf.create_dataset("y_test", data=y_test)


def save_model(model: RandomForestClassifier, filename: str) -> None:
    """Save the trained model to a file."""
    try:
        joblib.dump(model, filename)
        print(f"Model successfully saved to {filename}")
    except Exception as e:
        print(f"Failed to save the model. Error: {e}")


if __name__ == "__main__":
    directory_path = "./data/"
    data = load_and_combine_data(directory_path)
    data = feature_engineering(data)
    data = label_data(data)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    print("Evaluating model...")
    results = evaluate_model(model, X_test, y_test)
    print("Classification Report:", results["classification_report"])
    print("Accuracy:", results["accuracy"])
    print("F1 Score:", results["f1_score"])
    print("Performing cross-validation...")
    cv_score = perform_cross_validation(model, X_train, X_test, y_train, y_test)
    print(f"Cross-validation score: {cv_score}")
    save_data_h5(X_train, X_test, y_train, y_test)
    save_model(model, "random_forest_model.joblib")
