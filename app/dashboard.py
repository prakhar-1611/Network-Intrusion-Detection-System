import sys
import os
# adds the src folder to python's search path so we can import from it even
# though this file lives inside the app folder, one level away from src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# joblib is used to load the saved model and test split back from disk
import joblib
# numpy isn't used directly but several libraries here rely on it internally
import numpy as np
# pandas is used to wrap rows back into labeled tables for shap
import pandas as pd
# shap is used to compute the per row feature explanations shown on the page
import shap
# streamlit is the library that actually builds and runs this web page
import streamlit as st
# reuses the same shape-normalizing helper defined in explain.py instead of
# duplicating that logic here, works because src/ was added to the path above
from explain import select_class_shap

# a manually written lookup table mapping each attack label to a rough severity
# word, this isn't learned by the model, it's just our own triage decision
SEVERITY_MAP = {
    "BENIGN": "Info",
    "DDoS": "Critical",
    "Bot": "Critical",
    "Heartbleed": "Critical",
    "Infiltration": "Critical",
    "PortScan": "Low",
    "DoS Hulk": "High",
    "DoS GoldenEye": "High",
    "DoS slowloris": "High",
    "DoS Slowhttptest": "High",
    "FTP-Patator": "Medium",
    "SSH-Patator": "Medium",
    "Web Attack - Brute Force": "Medium",
    "Web Attack - XSS": "High",
    "Web Attack - Sql Injection": "High",
}

# severity -> (accent color, text color) used throughout the page. kept in one
# place so the alert card, the metric labels, and the chart bars all agree.
SEVERITY_COLORS = {
    "Critical": "#E5484D",
    "High": "#F2994A",
    "Medium": "#F2C94C",
    "Low": "#5B9DD9",
    "Info": "#3DD68C",
    "Unknown": "#8B95A1",
}


@st.cache_resource
def load_artifacts(model_path: str, test_path: str):
    # this decorator tells streamlit to only actually run this function once and
    # remember the result, instead of reloading everything on every single click
    # loads the saved model dictionary back from disk
    bundle = joblib.load(model_path)
    # loads the saved held out test data back from disk
    test_data = joblib.load(test_path)
    # builds one shap explainer here so it can be reused instead of rebuilding
    # it every time the page refreshes
    explainer = shap.TreeExplainer(bundle["model"])
    # hands all three back to whoever called this function
    return bundle, test_data, explainer


def inject_css():
    # one block of custom css that turns the default streamlit theme into a
    # dark, monospace-leaning "SOC console" look. everything here is scoped to
    # streamlit's existing class names so no javascript is needed.
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: #0A0E12;
        }

        /* hides streamlit's own dev toolbar (Deploy button, hamburger menu,
           footer) so the page reads as a finished product, not a work-in-
           progress app still open in dev mode */
        #MainMenu, header[data-testid="stHeader"], footer {
            visibility: hidden;
            height: 0;
        }

        .block-container {
            padding-top: 2.5rem;
            max-width: 1200px;
        }

        section[data-testid="stSidebar"] {
            background: #0D1218;
            border-right: 1px solid #1C242D;
        }

        .sidebar-heading {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            font-weight: 700;
            color: #5B9DD9;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
        }

        section[data-testid="stSidebar"] .stTextInput label,
        section[data-testid="stSidebar"] .stNumberInput label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: #8B95A1;
            letter-spacing: 0.02em;
        }

        section[data-testid="stSidebar"] input {
            font-family: 'JetBrains Mono', monospace;
            background: #12181F !important;
            color: #E6EAEF !important;
            border: 1px solid #1C242D !important;
            border-radius: 5px !important;
            box-shadow: none !important;
        }

        /* streamlit's default focus/active state paints inputs red, which
           reads as an error state next to our own severity red. swap it to
           the console's blue accent so focus looks intentional, not broken */
        section[data-testid="stSidebar"] input:focus {
            border: 1px solid #5B9DD9 !important;
            box-shadow: 0 0 0 1px #5B9DD9 !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="input"] {
            border-color: #1C242D !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within {
            border-color: #5B9DD9 !important;
            box-shadow: 0 0 0 1px #5B9DD9 !important;
        }

        /* number input +/- stepper buttons: default to a flat panel style
           that matches the rest of the console instead of unstyled boxes */
        section[data-testid="stSidebar"] button {
            background: #12181F !important;
            border: 1px solid #1C242D !important;
            color: #8B95A1 !important;
        }

        section[data-testid="stSidebar"] button:hover {
            border-color: #5B9DD9 !important;
            color: #5B9DD9 !important;
        }

        /* the "Model & data source" expander, styled as a quiet panel rather
           than default streamlit chrome */
        section[data-testid="stSidebar"] details {
            background: #0D1218;
            border: 1px solid #1C242D;
            border-radius: 6px;
        }

        section[data-testid="stSidebar"] summary {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: #8B95A1;
        }

        section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
            font-family: 'JetBrains Mono', monospace;
            color: #5B6572 !important;
        }

        h1 {
            font-weight: 700;
            color: #E6EAEF;
            letter-spacing: -0.01em;
            font-size: 2.1rem;
        }

        .console-caption {
            font-family: 'JetBrains Mono', monospace;
            color: #5B6572;
            font-size: 0.85rem;
            margin-top: -0.6rem;
            margin-bottom: 2rem;
        }

        /* the alert summary card: a dark panel with a colored left edge that
           carries the severity color, styled like a real SOC alert row */
        .alert-card {
            background: #12181F;
            border: 1px solid #1C242D;
            border-left: 4px solid var(--sev-color);
            border-radius: 6px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1.5rem;
        }

        .alert-stat-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: #8B95A1;
            text-transform: none;
            margin-bottom: 0.2rem;
        }

        .alert-stat-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.6rem;
            font-weight: 700;
            color: #E6EAEF;
        }

        .severity-pill {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--sev-color);
        }

        .ground-truth-note {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: #8B95A1;
            background: #12181F;
            border: 1px solid #1C242D;
            border-radius: 6px;
            padding: 0.7rem 1rem;
            margin-bottom: 1.5rem;
        }

        h3 {
            color: #E6EAEF;
            font-weight: 600;
            font-size: 1.05rem;
            margin-top: 0.5rem;
            margin-bottom: 1.25rem;
        }

        .stDataFrame {
            font-family: 'JetBrains Mono', monospace;
        }

        /* adds breathing room around the matplotlib chart so it doesn't sit
           flush against the heading and table above/below it */
        div[data-testid="stImage"], div[data-testid="stPyplotGlobalUseWarning"] {
            margin-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_alert_card(pred_label, proba, severity):
    # renders the predicted/confidence/severity summary as one bordered card
    # instead of three separate st.metric boxes, so it reads as a single alert
    color = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["Unknown"])
    conf_text = f"{proba:.1%}" if proba is not None else "n/a"
    st.markdown(
        f"""
        <div class="alert-card" style="--sev-color: {color};">
            <div>
                <div class="alert-stat-label">predicted</div>
                <div class="alert-stat-value">{pred_label}</div>
            </div>
            <div>
                <div class="alert-stat-label">confidence</div>
                <div class="alert-stat-value">{conf_text}</div>
            </div>
            <div>
                <div class="alert-stat-label">severity</div>
                <div class="severity-pill" style="color: {color};">{severity}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    # sets the browser tab title and makes the page use the full screen width.
    # menu_items={} plus the CSS in inject_css() below hide streamlit's own
    # dev toolbar ("Deploy" button, hamburger menu) so the page looks like a
    # standalone product instead of an app still being developed
    st.set_page_config(page_title="NIDS Alert Console", layout="wide", menu_items={})
    # applies the custom dark/monospace theme defined above
    inject_css()

    # draws the big title text at the top of the page
    st.title("Network Intrusion Detection — Alert Console")
    # draws a small caption line under the title
    st.markdown(
        '<div class="console-caption">CIC-IDS-2017 flow data · Random Forest classifier</div>',
        unsafe_allow_html=True,
    )

    # sidebar is framed as a control panel: a short section label, then the
    # two file paths tucked under an expander so they read as "advanced /
    # config" rather than the first thing a viewer's eye lands on
    st.sidebar.markdown('<div class="sidebar-heading">Flow inspector</div>', unsafe_allow_html=True)

    with st.sidebar.expander("Model & data source", expanded=False):
        # draws a text box for the model path, pre filled with a default
        model_path = st.text_input("Model path", "models/rf_multiclass.joblib")
        # draws a text box for the test split path, also pre filled
        test_path = st.text_input(
            "Test split path", "models/rf_multiclass_test_split.joblib"
        )

    # checks that both files actually exist at the paths currently typed in
    if not (os.path.exists(model_path) and os.path.exists(test_path)):
        # shows a warning message on the page telling the user what to do
        st.warning(
            "Model or test split not found. Run `python src/train.py` first, "
            "then point the sidebar paths at the generated .joblib files."
        )
        # stops the function here so nothing below tries to load a missing file
        return

    # calls the cached loader function to get the model, test data, and explainer
    bundle, test_data, explainer = load_artifacts(model_path, test_path)
    # pulls the trained model object out of the loaded dictionary
    model = bundle["model"]
    # pulls the label encoder out so we can convert numbers back to text
    label_encoder = bundle["label_encoder"]
    # pulls the exact feature column order the model was trained with
    feature_cols = bundle["feature_cols"]
    # pulls the test features and true answers out of the loaded test data
    X_test, y_test = test_data["X_test"], test_data["y_test"]

    # draws a number input box in the sidebar letting the user pick which test
    # row to inspect, defaulting to row zero and capped at the last available row
    idx = st.sidebar.number_input(
        "Flow index",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        step=1,
        help=f"0 to {len(X_test) - 1} — each number is one flow from the held-out test set",
    )
    st.sidebar.caption(f"{len(X_test):,} flows available in this test split")

    # pulls out just that one chosen row and reshapes it into the two dimensional
    # shape the model expects for a single prediction
    row = X_test[idx].reshape(1, -1)
    # converts the true answer number for this row back into its readable text label
    true_label = label_encoder.inverse_transform([y_test[idx]])[0]
    # predicts the class number for this row
    pred_idx = model.predict(row)[0]
    # converts that predicted number back into its readable text label
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    # gets the model's confidence percentage specifically for the class it predicted,
    # only if the model actually supports probability output
    proba = model.predict_proba(row)[0][pred_idx] if hasattr(model, "predict_proba") else None
    # looks up the severity word for whatever the model predicted
    severity = SEVERITY_MAP.get(pred_label, "Unknown")

    # renders the predicted / confidence / severity summary as one alert card
    render_alert_card(pred_label, proba, severity)

    # if the model's prediction doesn't match the real answer, reveal what the
    # correct answer actually was, useful for spotting where the model struggles
    if pred_label != true_label:
        st.markdown(
            f'<div class="ground-truth-note">ground truth for this demo row: {true_label}</div>',
            unsafe_allow_html=True,
        )

    # draws a subheading before the explainability section
    st.subheader("Why the model flagged this flow")
    # wraps this single row back into a labeled table so shap can use the column names
    row_df = pd.DataFrame(row, columns=feature_cols)
    # computes shap values for just this one row
    shap_values = explainer.shap_values(row_df)
    # pulls out just the contribution scores for the class that got predicted,
    # regardless of which shape this particular shap version returned
    class_shap = select_class_shap(shap_values, pred_idx)

    # builds a small table pairing each feature name with its value in this row
    # and how much it contributed to the prediction
    contrib = pd.DataFrame({
        "feature": feature_cols,
        "value": row[0],
        "shap_contribution": class_shap,
    }).reindex(columns=["feature", "value", "shap_contribution"])
    # adds a column with the absolute size of each contribution so strongly negative
    # and strongly positive pushes both count as important
    contrib["abs_impact"] = contrib["shap_contribution"].abs()
    # sorts by that absolute impact and keeps only the top ten most influential features
    contrib = contrib.sort_values("abs_impact", ascending=False).head(10)
    # re-sorts ascending for display so the horizontal bar chart reads top-to-bottom
    # with the strongest contributor at the top, matplotlib draws bottom-up otherwise
    contrib_display = contrib.sort_values("shap_contribution", ascending=True)

    # draws the shap contributions as a horizontal bar chart, colored red for
    # pushes toward the predicted class and blue for pushes away from it, with
    # full (untruncated) feature names on the axis
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#0A0E12")
    ax.set_facecolor("#0A0E12")

    bar_colors = ["#E5484D" if v >= 0 else "#5B9DD9" for v in contrib_display["shap_contribution"]]
    ax.barh(contrib_display["feature"], contrib_display["shap_contribution"], color=bar_colors)

    ax.axvline(0, color="#3A4552", linewidth=0.8)
    ax.tick_params(colors="#8B95A1", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1C242D")
    ax.set_xlabel("SHAP contribution", color="#8B95A1", fontsize=9)
    ax.set_ylabel("")

    plt.tight_layout()
    st.pyplot(fig)

    # also shows the same data as a readable table below the chart, dropping the
    # helper column that was only needed for sorting, not for display
    st.dataframe(contrib.drop(columns="abs_impact"), width="stretch")


if __name__ == "__main__":
    # runs the main function, streamlit actually re-executes this whole file
    # on every interaction, this line mostly matters if the file is ever run
    # as plain python instead of through the streamlit command
    main()