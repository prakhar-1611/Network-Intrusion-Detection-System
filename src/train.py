import argparse
# os lets us build folder/file paths and create directories if missing
import os
# joblib is used to save and load python objects like a trained model to/from disk
import joblib
# numpy is used for building the sample weight array for xgboost
import numpy as np
# pandas is used to load the cleaned parquet file back into a table
import pandas as pd
# train_test_split is the function that splits our data into training and testing chunks
from sklearn.model_selection import train_test_split
# labelencoder converts the text labels like "benign" or "ddos" into plain numbers
from sklearn.preprocessing import LabelEncoder
# randomforestclassifier is one of the two model types we can train
from sklearn.ensemble import RandomForestClassifier
# xgbclassifier is the other model type we can train
from xgboost import XGBClassifier

# reuses the exact same "label" string constant defined in preprocess.py instead
# of retyping it here, keeps both files in sync if the column name ever changes
from preprocess import LABEL_COLUMN


# these two column names are never actual input features, they're the answers,
# so this set is used to filter them out when building the feature list
FEATURE_BLOCKLIST = {LABEL_COLUMN, "Label_Binary"}


def get_feature_columns(df: pd.DataFrame) -> list:
    # returns every column name except the ones in the blocklist above, this
    # gives us the actual 78 numeric columns the model is allowed to look at
    return [c for c in df.columns if c not in FEATURE_BLOCKLIST]


def train_model(df: pd.DataFrame, target_col: str, model_type: str = "rf"):
    # grabs the list of feature column names to use as input
    feature_cols = get_feature_columns(df)
    # pulls just those feature columns out as a plain numpy array, this is what
    # the model actually reads as input
    X = df[feature_cols].values
    # pulls the chosen target column out as an array of text labels
    y_raw = df[target_col].values

    # creates a label encoder object, not yet trained on anything
    le = LabelEncoder()
    # learns the mapping from text label to number, and immediately converts
    # y_raw from text into that array of numbers in one step
    y = le.fit_transform(y_raw)

    # splits both X and y together into a training chunk and a testing chunk
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        # twenty five percent of the rows go into the test set, the rest train the model
        test_size=0.25,
        # fixes the randomness so the exact same split happens every time this runs
        random_state=42,
        # keeps the same proportion of each class in both the train and test sets so
        # rare attack types don't accidentally end up missing from one side
        stratify=y,
    )

    # only runs this branch if the random forest option was chosen
    if model_type == "rf":
        # builds a random forest model with these settings, nothing is trained yet
        model = RandomForestClassifier(
            # builds three hundred individual decision trees
            n_estimators=300,
            # lets each tree grow as deep as it needs to, no artificial limit
            max_depth=None,
            # uses every available cpu core in parallel to speed up training
            n_jobs=-1,
            # automatically gives more weight to rare classes so they aren't ignored
            class_weight="balanced",
            # fixes randomness inside the forest itself for reproducibility
            random_state=42,
        )
    # only runs this branch if the xgboost option was chosen
    elif model_type == "xgb":
        # finds every distinct class number in the training labels and how many
        # times each one appears
        classes, counts = np.unique(y_train, return_counts=True)
        # computes a weight per class using total rows divided by number of classes
        # times how many rows that class has, so rarer classes get bigger weights
        weight_map = {c: len(y_train) / (len(classes) * cnt)
                      for c, cnt in zip(classes, counts)}
        # builds an array the same length as y_train where each row gets the
        # weight that matches its own class
        sample_weight = np.array([weight_map[label] for label in y_train])
        # builds the xgboost model with these settings, nothing trained yet
        model = XGBClassifier(
            # runs four hundred sequential boosting rounds
            n_estimators=400,
            # lets each individual tree go up to eight levels deep
            max_depth=8,
            # controls how much each new tree corrects the mistakes of earlier ones
            learning_rate=0.1,
            # each tree only sees a random eighty percent of the rows, reduces overfitting
            subsample=0.8,
            # each tree only considers a random eighty percent of the features
            colsample_bytree=0.8,
            # uses the faster histogram based method for finding tree splits
            tree_method="hist",
            # sets the loss function used internally for multiclass problems
            eval_metric="mlogloss",
            # fixes randomness for reproducibility
            random_state=42,
        )
        # actually trains the xgboost model, using the per row weights calculated above
        model.fit(X_train, y_train, sample_weight=sample_weight)
        # returns straight away here so the random forest only line below gets skipped
        return model, le, feature_cols, (X_train, X_test, y_train, y_test)
    # if neither "rf" nor "xgb" was passed in, stop with a clear error
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # this line only runs for the random forest branch, this is the actual training
    # step where all three hundred trees get built from the training data
    model.fit(X_train, y_train)
    # returns the trained model, the label encoder, the feature names, and all four splits
    return model, le, feature_cols, (X_train, X_test, y_train, y_test)


def run(processed_path: str, target: str, model_type: str, out_dir: str):
    # loads the cleaned parquet file back into a pandas table
    df = pd.read_parquet(processed_path)
    # picks which column is the actual answer depending on whether we want
    # the simple binary target or the full multiclass target
    target_col = LABEL_COLUMN if target == "multiclass" else "Label_Binary"

    # prints a status line showing what's about to be trained and how many classes exist
    print(f"Training {model_type} model for target='{target}' "
          f"({df[target_col].nunique()} classes)...")
    # actually calls the training function defined above
    model, label_encoder, feature_cols, splits = train_model(df, target_col, model_type)

    # makes sure the models output folder exists before saving into it
    os.makedirs(out_dir, exist_ok=True)
    # builds a filename like rf_multiclass.joblib based on the chosen settings
    model_path = os.path.join(out_dir, f"{model_type}_{target}.joblib")
    # saves everything needed to use this model later into one file: the model
    # itself, the label encoder, the exact feature order, and which target it used
    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
            "feature_cols": feature_cols,
            "target_col": target_col,
        },
        model_path,
    )
    # prints confirmation of where the model got saved
    print(f"Saved model to {model_path}")

    # unpacks the four way split tuple back into individual variables
    X_train, X_test, y_train, y_test = splits
    # builds the filename for the saved test split
    test_path = os.path.join(out_dir, f"{model_type}_{target}_test_split.joblib")
    # saves just the held out test portion so later scripts always evaluate on
    # the exact same data the model never trained on
    joblib.dump({"X_test": X_test, "y_test": y_test}, test_path)
    # prints confirmation of where the test split got saved
    print(f"Saved held-out test split to {test_path}")

    # returns the model path in case this function is called from other code
    return model_path


if __name__ == "__main__":
    # sets up the command line argument parser with a short description
    parser = argparse.ArgumentParser(description="Train a NIDS model on CIC-IDS-2017")
    # defines the --data flag, defaulting to the standard cleaned parquet path
    parser.add_argument("--data", default="data/processed/cicids2017_clean.parquet")
    # defines the --target flag, only allows binary or multiclass as valid choices
    parser.add_argument("--target", choices=["binary", "multiclass"], default="multiclass")
    # defines the --model flag, only allows rf or xgb as valid choices
    parser.add_argument("--model", choices=["rf", "xgb"], default="rf")
    # defines the --out_dir flag, defaulting to the models folder
    parser.add_argument("--out_dir", default="models")
    # actually reads whatever flags were typed on the command line
    args = parser.parse_args()
    # kicks off the whole training pipeline with those settings
    run(args.data, args.target, args.model, args.out_dir)
