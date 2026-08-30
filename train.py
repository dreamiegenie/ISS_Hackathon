"""Train and evaluate the Track 1 diagnosis classifier.

Example:
    python train.py --data /path/to/track1_participant_dataset.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

TARGET = "target_diagnosis"
ID_COLUMN = "patient_id"


def build_model(categorical: list[str], numeric: list[str]) -> Pipeline:
    """Create a fast, robust multiclass baseline without using patient IDs."""
    preprocess = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=250,
                    max_leaf_nodes=31,
                    l2_regularization=0.2,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the participant CSV")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)

    required = {ID_COLUMN, TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    X = df.drop(columns=[TARGET, ID_COLUMN])
    y = df[TARGET]
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    model = build_model(categorical, numeric)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "rows": int(len(df)),
        "features_used": X.columns.tolist(),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }
    joblib.dump(model, output_dir / "diagnosis_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(f"Saved model to {output_dir / 'diagnosis_model.joblib'}")


if __name__ == "__main__":
    main()
