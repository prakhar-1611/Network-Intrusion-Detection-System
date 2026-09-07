import argparse
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def load_bundle(model_path: str):
    return joblib.load(model_path)


def select_class_shap(shap_values, class_idx: int, row_idx: int = 0):
    if isinstance(shap_values, list):
        return shap_values[class_idx][row_idx]
    if shap_values.ndim == 3:
        return shap_values[row_idx, :, class_idx]
    return shap_values[row_idx]


def global_feature_importance(model_path: str, test_path: str, out_path: str,
                               sample_size: int = 2000):
    bundle = load_bundle(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    test_data = joblib.load(test_path)
    X_test = test_data["X_test"]

    if len(X_test) > sample_size:
        idx = np.random.RandomState(42).choice(len(X_test), sample_size, replace=False)
        X_sample = X_test[idx]
    else:
        X_sample = X_test

    X_df = pd.DataFrame(X_sample, columns=feature_cols)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    plt.figure()
    if isinstance(shap_values, list):
        stacked = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        stacked = np.abs(shap_values).mean(axis=2)
    else:
        stacked = shap_values
    shap.summary_plot(stacked, X_df, show=False, plot_type="bar")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved SHAP global feature importance plot to {out_path}")
    return explainer, feature_cols


def explain_single_flow(model_path: str, explainer, flow_row: pd.Series,
                         feature_cols: list, top_n: int = 8):
    bundle = load_bundle(model_path)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]

    X_row = flow_row[feature_cols].values.reshape(1, -1)
    pred_class_idx = model.predict(X_row)[0]
    pred_label = label_encoder.inverse_transform([pred_class_idx])[0]

    shap_values = explainer.shap_values(pd.DataFrame(X_row, columns=feature_cols))
    class_shap = select_class_shap(shap_values, pred_class_idx)

    contributions = list(zip(feature_cols, X_row[0], class_shap))
    contributions.sort(key=lambda t: abs(t[2]), reverse=True)

    return pred_label, contributions[:top_n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP explainability for the NIDS model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_split", required=True)
    parser.add_argument("--out", default="reports/shap_summary.png")
    args = parser.parse_args()
    global_feature_importance(args.model, args.test_split, args.out)