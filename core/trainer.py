import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_squared_error, mean_absolute_error
)
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def preprocess_for_training(df, target_col, task_type="classification", max_rows=10000):
    """
    Cleans and prepares X, y for modeling with auto-downsampling safeguards.
    """
    df_clean = df.dropna(subset=[target_col]).copy()
    is_downsampled = False

    # Guard against huge datasets
    if len(df_clean) > max_rows:
        if task_type == "classification" and df_clean[target_col].nunique() < 20:
            df_clean = df_clean.groupby(target_col, group_keys=False).apply(
                lambda x: x.sample(min(len(x), int(max_rows * len(x) / len(df_clean))), random_state=42)
            )
        else:
            df_clean = df_clean.sample(n=max_rows, random_state=42)
        is_downsampled = True

    y_raw = df_clean[target_col].copy()
    X = df_clean.drop(columns=[target_col]).copy()

    # Encode target if classification
    target_encoder = None
    if task_type == "classification":
        if y_raw.dtype == "object" or str(y_raw.dtype).startswith("cat") or y_raw.dtype == "bool":
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y_raw.astype(str))
        else:
            unique_vals = np.sort(np.unique(y_raw))
            if not np.array_equal(unique_vals, np.arange(len(unique_vals))):
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y_raw)
            else:
                y = y_raw.values.astype(int)
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0).values.astype(float)

    # Preprocess features: encode categoricals & handle dates
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for col in cat_cols:
        col_lower = str(col).lower()
        if any(k in col_lower for k in ["date", "time", "timestamp"]):
            try:
                parsed = pd.to_datetime(X[col], errors="coerce")
                if parsed.notnull().mean() > 0.7:
                    X[f"{col}_year"] = parsed.dt.year.fillna(2000).astype(int)
                    X[f"{col}_month"] = parsed.dt.month.fillna(1).astype(int)
                    X[f"{col}_day"] = parsed.dt.day.fillna(1).astype(int)
                    X.drop(columns=[col], inplace=True)
                    continue
            except Exception:
                pass

        X[col] = LabelEncoder().fit_transform(X[col].astype(str).fillna("missing"))

    # Impute and clean remaining features
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X, y, target_encoder, is_downsampled


def evaluate_model(y_true, y_pred, task_type="classification"):
    """
    Computes standard evaluation metrics.
    """
    if task_type == "classification":
        return {
            "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "F1 (Weighted)": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "Precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "Recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        }
    else:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return {
            "R² Score": round(float(r2_score(y_true, y_pred)), 4),
            "RMSE": round(rmse, 4),
            "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        }


def run_automl_pipeline(
    df,
    target_col,
    task_type="classification",
    is_time_series=False,
    max_trials=15,
    timeout_seconds=40,
    progress_callback=None
):
    """
    Full AutoML loop: Baseline fit -> Optuna-tuned Ensemble -> Evaluation.
    """
    start_time = time.time()

    # 1. Preprocess & safe sample
    X, y, target_encoder, is_downsampled = preprocess_for_training(
        df, target_col, task_type=task_type
    )

    # 2. Train-test split
    if is_time_series:
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    else:
        stratify_param = y if (task_type == "classification" and len(np.unique(y)) > 1) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )

    # 3. Train Baseline Model (Logistic or Linear Regression)
    if task_type == "classification":
        baseline_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=300, random_state=42))
        ])
    else:
        baseline_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])

    baseline_pipeline.fit(X_train, y_train)
    baseline_preds = baseline_pipeline.predict(X_test)
    baseline_metrics = evaluate_model(y_test, baseline_preds, task_type)

    if progress_callback:
        progress_callback(0.25, "Baseline model evaluated. Running Bayesian optimization...")

    # 4. Optuna Bayesian Optimization on Random Forest Ensemble
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": 42,
            "n_jobs": -1
        }

        if task_type == "classification":
            model = RandomForestClassifier(**params)
        else:
            model = RandomForestRegressor(**params)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if task_type == "classification":
            return accuracy_score(y_test, preds)
        else:
            return r2_score(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=max_trials,
        timeout=timeout_seconds,
        n_jobs=1
    )

    best_params = study.best_params
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1

    if task_type == "classification":
        best_model = RandomForestClassifier(**best_params)
    else:
        best_model = RandomForestRegressor(**best_params)

    best_model.fit(X_train, y_train)
    tuned_preds = best_model.predict(X_test)
    tuned_metrics = evaluate_model(y_test, tuned_preds, task_type)

    elapsed_time = round(time.time() - start_time, 2)

    return {
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": tuned_metrics,
        "best_params": study.best_params,
        "best_model": best_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "target_encoder": target_encoder,
        "is_downsampled": is_downsampled,
        "trials_completed": len(study.trials),
        "elapsed_time": elapsed_time,
        "task_type": task_type
    }
