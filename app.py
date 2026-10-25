
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Food Adulteration Detection",
    page_icon="🥛",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "cleaned_food_adulteration_data.csv"

FEATURES = ["product_name","brand","adulterant"]
TARGET = "severity"

st.title("🥛 Food Adulteration Detection")
st.subheader("Random Forest vs Gradient Boosting")
st.write(
    "A machine-learning demonstration that predicts the recorded "
    "severity category from the dataset."
)

st.warning(
    "Academic demonstration only. The supplied dataset contains "
    "synthetic/randomized records. This application does not chemically "
    "test food or establish whether a real sample is safe."
)

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(
            f"Required columns are missing from the CSV: {missing}"
        )

    data = data[required].dropna().copy()

    for col in required:
        data[col] = data[col].astype(str).str.strip()

    data = data[(data[required] != "").all(axis=1)]

    if data.empty:
        raise ValueError("No usable rows were found in the dataset.")

    if data[TARGET].nunique() < 2:
        raise ValueError("The severity target must have at least two classes.")

    return data


try:
    df = load_data(DATA_PATH)
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.info(
        "Place cleaned_food_adulteration_data.csv in the same folder "
        "as app.py."
    )
    st.stop()

# ---------------- DATA OVERVIEW ----------------

st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total usable records", f"{len(df):,}")
col2.metric("Input features", len(FEATURES))
col3.metric("Severity categories", df[TARGET].nunique())

with st.expander("View dataset sample"):
    st.dataframe(df.head(10), use_container_width=True)

st.caption(
    "Input features: product_name, brand and adulterant. "
    "Prediction target: severity."
)

# ---------------- SPLIT DATA ----------------

X = df[FEATURES]
y = df[TARGET]

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
except ValueError as error:
    st.error(f"Unable to split the dataset: {error}")
    st.stop()

# ---------------- DEFINE MODELS ----------------

def build_model(classifier):
    preprocessing = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            FEATURES
        )
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier)
    ])


model_definitions = {
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

# ---------------- TRAIN AND EVALUATE ----------------

@st.cache_resource
def train_models(X_train, y_train):
    fitted_models = {}

    for name, classifier in model_definitions.items():
        model = build_model(classifier)
        model.fit(X_train, y_train)
        fitted_models[name] = model

    return fitted_models


try:
    with st.spinner("Training and evaluating both models..."):
        trained_models = train_models(X_train, y_train)
except Exception as error:
    st.error(f"Model training failed: {error}")
    st.stop()

results = []
predictions_by_model = {}

for name, model in trained_models.items():
    predictions = model.predict(X_test)
    predictions_by_model[name] = predictions

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions,
            average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions,
            average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions,
            average="weighted", zero_division=0
        )
    })

results_df = pd.DataFrame(results)

# ---------------- RESULTS TABLE ----------------

st.header("2. Model Performance")

st.write(
    "Both models use the same 80% training / 20% testing split. "
    "Scores are calculated on the held-out test set."
)

st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1 Score": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)

# ---------------- COMPARISON CHART ----------------

st.header("3. Performance Comparison")

fig, ax = plt.subplots(figsize=(9, 5))

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(kind="bar", ax=ax, ylim=(0, 1))

ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_title("Random Forest vs Gradient Boosting")
ax.legend(loc="lower right")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

best_model_name = results_df.sort_values(
    "F1 Score", ascending=False
).iloc[0]["Model"]

st.success(
    f"Best weighted F1 score on this test split: {best_model_name}"
)


# ---------------- CONFUSION MATRICES ----------------

st.header("4. Confusion Matrices")

labels = sorted(y.unique())

col1, col2 = st.columns(2)

for index, (name, predictions) in enumerate(
    predictions_by_model.items()
):
    with (col1 if index == 0 else col2):
        st.subheader(name)

        fig, ax = plt.subplots(figsize=(4, 3))

        display = ConfusionMatrixDisplay(
            confusion_matrix(
                y_test,
                predictions,
                labels=labels
            ),
            display_labels=labels
        )

        display.plot(
            ax=ax,
            colorbar=False,
            values_format="d",
            xticks_rotation=30
        )

        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted label", fontsize=9)
        ax.set_ylabel("True label", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
# ---------------- CLASSIFICATION REPORT ----------------

st.header("5. Classification Report")

selected_report_model = st.selectbox(
    "Choose a model to inspect",
    list(trained_models.keys())
)

st.text(
    classification_report(
        y_test,
        predictions_by_model[selected_report_model],
        zero_division=0
    )
)

# ---------------- PREDICTION FORM ----------------

st.header("6. Predict Recorded Severity")

st.info(
    "Choose values from the existing dataset. The output predicts the "
    "recorded severity label; it is not a laboratory diagnosis."
)

col1, col2, col3 = st.columns(3)

with col1:
    selected_product = st.selectbox(
        "Product",
        sorted(df["product_name"].unique())
    )

with col2:
    selected_brand = st.selectbox(
        "Brand",
        sorted(df["brand"].unique())
    )

with col3:
    selected_adulterant = st.selectbox(
        "Adulterant",
        sorted(df["adulterant"].unique())
    )

if st.button("Compare Predictions", type="primary"):
    input_data = pd.DataFrame([{
        "product_name": selected_product,
        "brand": selected_brand,
        "adulterant": selected_adulterant
    }])

    st.subheader("Prediction Results")

    result_columns = st.columns(len(trained_models))
    for column, (name, model) in zip(result_columns, trained_models.items()):
        prediction = model.predict(input_data)[0]
        with column:
            st.markdown(f"### {name}")
            st.metric("Predicted recorded severity", str(prediction))

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                probability_df = pd.DataFrame({
                    "Severity category": model.classes_,
                    "Model score (%)": probabilities * 100
                }).sort_values("Model score (%)", ascending=False)

                st.write("Scores by severity category:")
                st.dataframe(
                    probability_df.style.format({"Model score (%)": "{:.2f}%"}),
                    use_container_width=True,
                    hide_index=True
                )
                st.bar_chart(
                    probability_df.set_index("Severity category")["Model score (%)"]
                )

    st.caption(
        "Scores are model output probabilities for the listed classes, "
        "not verified probabilities of real-world food safety or adulteration. "
        "Different models can legitimately produce the same prediction or score."
    )

st.divider()

st.caption(
    "Mini project: Food Adulteration Detection using Machine Learning | "
    "Models: Random Forest and Gradient Boosting"
)
