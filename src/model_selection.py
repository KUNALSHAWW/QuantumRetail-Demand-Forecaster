import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import joblib
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "models")


def time_series_split(df: pd.DataFrame, train_window: int):
    """
    Split df into train and validation based on given windows.
    Assumes df is already filtered for a single store/product combination.
    """
    df = create_features(df)
    df = df.reset_index(drop=True)
    train = df.iloc[:train_window]
    val = df.iloc[train_window:]
    return train, val


def create_features(df: pd.DataFrame):
    """Create lag and rolling features for modeling."""
    # df = df.copy()
    df["lag_1"] = df["sale_amount"].shift(1)
    df["lag_7"] = df["sale_amount"].shift(7)
    df["rolling_mean_7"] = df["sale_amount"].shift(1).rolling(7).mean()
    df = df.dropna()
    return df


def evaluate_model(y_true, y_pred):
    """Return evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmse": rmse, "mae": mae}


def train_and_select_model(train: pd.DataFrame, val: pd.DataFrame):
    """
    Train multiple models and select the best based on RMSE.
    """
    results = {}

    # Features/Target


    X_train, y_train = train.drop(columns=["sale_amount", "dt"]), train["sale_amount"]
    X_val, y_val = val.drop(columns=["sale_amount", "dt"]), val["sale_amount"]

    # Candidate models
    models = {
        "LightGBM": lgb.LGBMRegressor(verbosity=-1),
        "XGBoost": xgb.XGBRegressor(),
    }

    best_model = None
    best_score = float("inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        metrics = evaluate_model(y_val, preds)
        results[name] = metrics

        if metrics["rmse"] < best_score:
            best_score = metrics["rmse"]
            best_model = model

    return best_model,results


def get_model_path(store_id: str, product_id: str, train_days: int) -> str:
    """Return full model path for a given store/product/train_days combo."""
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    filename = f"{store_id}_{product_id}_{train_days}.pkl"
    return os.path.join(MODEL_SAVE_PATH, filename)


def save_model(model, store_id: str, product_id: str, train_days: int):
    """Save the trained model if not already present."""
    path = get_model_path(store_id, product_id, train_days)
    joblib.dump(model, path)
    return path


def load_model(store_id: str, product_id: str, train_days: int):
    """Load model if it exists, else return None."""
    path = get_model_path(store_id, product_id, train_days)
    if os.path.exists(path):
        return joblib.load(path)
    return None



