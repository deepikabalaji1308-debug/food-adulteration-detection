
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "cleaned_food_adulteration_data.csv"

FEATURES = ["product_name", "brand", "adulterant"]
TARGET = "severity"


def main():
    # Load dataset and normalize column names
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[required].dropna().copy()

    for column in required:
        df[column] = df[column].astype(str).str.strip()

    df = df[(df[required] != "").all(axis=1)]

    X = df[FEATURES]
    y = df[TARGET]

    if y.nunique() < 2:
        raise ValueError("The severity column must contain at least two classes.")

    # Same held-out test set for every model
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Majority Baseline": DummyClassifier(strategy="most_frequent"),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        )
    }

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, classifier in models.items():
        preprocessing = ColumnTransformer([
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                FEATURES
            )
        ])

        model = Pipeline([
            ("preprocessing", preprocessing),
            ("classifier", classifier)
        ])

        # Cross-validation on training data only
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "precision": "precision_weighted",
                "recall": "recall_weighted",
                "f1": "f1_weighted"
            },
            n_jobs=-1
        )

        # Fit once on training data and evaluate on held-out test data
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Test Accuracy": accuracy_score(y_test, predictions),
            "Test Precision": precision_score(
                y_test, predictions,
                average="weighted", zero_division=0
            ),
            "Test Recall": recall_score(
                y_test, predictions,
                average="weighted", zero_division=0
            ),
            "Test F1 Score": f1_score(
                y_test, predictions,
                average="weighted", zero_division=0
            ),
            "CV Accuracy Mean": cv_scores["test_accuracy"].mean(),
            "CV Accuracy Std": cv_scores["test_accuracy"].std(),
            "CV F1 Mean": cv_scores["test_f1"].mean()
        })

        print(f"\n{'=' * 45}")
        print(name)
        print("=" * 45)
        print(classification_report(
            y_test, predictions, zero_division=0
        ))
        print("Confusion matrix:")
        print(confusion_matrix(
            y_test, predictions, labels=sorted(y.unique())
        ))

    results_df = pd.DataFrame(results).sort_values(
        by="Test F1 Score", ascending=False
    )

    output_path = BASE_DIR / "model_comparison_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nMODEL COMPARISON")
    print(results_df.round(3).to_string(index=False))
    print(f"\nResults saved to: {output_path}")

    print(
        "\nUse these results as an evaluation of the supplied labels. "
        "High scores alone do not establish real-world food safety."
    )


if __name__ == "__main__":
    main()
