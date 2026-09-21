import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap


def compute_shap_explanations(model, X_test, max_display=10, sample_size=150):
    """
    Computes TreeSHAP feature importances and returns summary figure and importance table.
    """
    # Sample test set to maintain fast response times
    if len(X_test) > sample_size:
        X_sample = X_test.sample(sample_size, random_state=42)
    else:
        X_sample = X_test.copy()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Handle multiclass vs binary/regression
        if isinstance(shap_values, list):
            # Multiclass: take class 0 or mean across classes
            val_to_plot = shap_values[0]
        elif len(getattr(shap_values, "shape", [])) == 3:
            val_to_plot = shap_values[:, :, 0]
        else:
            val_to_plot = shap_values

        # Generate SHAP Summary Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(
            val_to_plot,
            X_sample,
            max_display=max_display,
            show=False,
            plot_type="dot"
        )
        plt.title("TreeSHAP Global Feature Impact (Beeswarm)", fontsize=13, pad=12)
        plt.tight_layout()

        # Compute mean absolute SHAP for table
        mean_abs_shap = np.abs(val_to_plot).mean(axis=0)
        importance_df = pd.DataFrame({
            "Feature": X_sample.columns,
            "Mean |SHAP Value|": mean_abs_shap
        }).sort_values(by="Mean |SHAP Value|", ascending=False).reset_index(drop=True)

        return fig, importance_df.head(max_display)

    except Exception as e:
        # Fallback to LightGBM native feature importance if SHAP fails
        fig, ax = plt.subplots(figsize=(8, 5))
        importances = model.feature_importances_
        features = X_test.columns
        indices = np.argsort(importances)[::-1][:max_display]

        ax.barh([features[i] for i in reversed(indices)], [importances[i] for i in reversed(indices)], color="#4CAF50")
        ax.set_title("Feature Importance (Split-Gain Fallback)")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()

        importance_df = pd.DataFrame({
            "Feature": [features[i] for i in indices],
            "Importance Score": [importances[i] for i in indices]
        })
        return fig, importance_df
