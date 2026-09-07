import argparse
# os lets us build folder/file paths and create directories if missing
import os
# joblib is used to load the saved model and test split back from disk
import joblib
# numpy isn't used directly here but matplotlib/sklearn rely on it under the hood
import numpy as np
# matplotlib is used to draw and save the confusion matrix image
import matplotlib.pyplot as plt
# seaborn makes the confusion matrix heatmap look nicer than plain matplotlib
import seaborn as sns
# these three give us the actual scoring functions for the model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def evaluate(model_path: str, test_path: str, report_dir: str):
    # loads back the dictionary that train.py saved
    bundle = joblib.load(model_path)
    # pulls the trained model object out of that dictionary
    model = bundle["model"]
    # pulls the label encoder out so we can convert numbers back to text labels
    label_encoder = bundle["label_encoder"]

    # loads back the held out test data that train.py saved separately
    test_data = joblib.load(test_path)
    # pulls the test features and true answers out of that dictionary
    X_test, y_test = test_data["X_test"], test_data["y_test"]

    # feeds every test row into the trained model and gets back predicted class numbers
    y_pred = model.predict(X_test)
    # grabs the original text label names in the same order the encoder assigned numbers
    class_names = label_encoder.classes_

    # compares true answers against predictions and computes precision, recall, and
    # f1 score for every single class, zero_division stops it from crashing when a
    # class has zero predicted or zero true examples
    report = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )
    # prints the full report text to the terminal
    print(report)

    # makes sure the reports output folder exists before saving into it
    os.makedirs(report_dir, exist_ok=True)
    # builds the path where the text report will be saved
    report_path = os.path.join(report_dir, "classification_report.txt")
    # opens a new text file and writes the report into it, closes automatically after
    with open(report_path, "w") as f:
        f.write(report)
    # prints confirmation of where the report got saved
    print(f"Saved report to {report_path}")

    # builds the confusion matrix grid, counting how many rows of each true class
    # got predicted as each possible class
    cm = confusion_matrix(y_test, y_pred)
    # opens a new blank drawing canvas, sized bigger automatically if there are
    # more classes so the labels don't overlap each other
    plt.figure(figsize=(max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.7)))
    # draws the confusion matrix as a colored grid, annot writes the actual numbers
    # inside each cell, fmt keeps them as whole integers, cmap sets the blue color scale
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    # labels the horizontal axis
    plt.xlabel("Predicted")
    # labels the vertical axis
    plt.ylabel("Actual")
    # adds a title above the chart
    plt.title("Confusion Matrix")
    # rotates the class names on the x axis so long ones don't overlap each other
    plt.xticks(rotation=45, ha="right")
    # automatically adjusts spacing so nothing gets cut off at the edges
    plt.tight_layout()
    # builds the path where the confusion matrix image will be saved
    cm_path = os.path.join(report_dir, "confusion_matrix.png")
    # saves the chart as a png image at a decent resolution
    plt.savefig(cm_path, dpi=150)
    # prints confirmation of where the image got saved
    print(f"Saved confusion matrix to {cm_path}")

    # only bothers computing roc-auc if the model actually supports probability output
    # and there are more than two classes since binary auc works differently
    if hasattr(model, "predict_proba") and len(class_names) > 2:
        try:
            # gets a probability score for every class, for every test row
            y_proba = model.predict_proba(X_test)
            # computes one score measuring how well the model separates each class
            # from all the others, then averages those scores equally across classes
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            # prints the final averaged score
            print(f"Macro-average ROC-AUC (one-vs-rest): {auc:.4f}")
        except ValueError as e:
            # catches the case where the math fails, usually because a class has
            # too few examples in the test set, and just prints why instead of crashing
            print(f"Could not compute multiclass ROC-AUC: {e}")

    # returns the report text and the raw confusion matrix numbers in case this
    # function is called from other code instead of the command line
    return report, cm


if __name__ == "__main__":
    # sets up the command line argument parser with a short description
    parser = argparse.ArgumentParser(description="Evaluate a trained NIDS model")
    # requires the path to the saved model bundle, no default since it depends on
    # what was actually trained
    parser.add_argument("--model", required=True, help="Path to .joblib model bundle")
    # requires the path to the saved test split
    parser.add_argument("--test_split", required=True, help="Path to .joblib test split")
    # defines the --report_dir flag, defaulting to the reports folder
    parser.add_argument("--report_dir", default="reports")
    # actually reads whatever flags were typed on the command line
    args = parser.parse_args()
    # kicks off the evaluation using those paths
    evaluate(args.model, args.test_split, args.report_dir)
