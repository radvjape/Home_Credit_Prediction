import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
from sklearn.compose import ColumnTransformer
from functions import convert_days_to_years, convert_days_to_months
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder

warnings.simplefilter(action="ignore", category=FutureWarning)


def merge_by_sk_id_curr(application_df, other_path="other_datasets.pkl"):

    other_df = pd.read_pickle(other_path)

    app = application_df.copy()
    other = other_df.copy()

    app.columns = app.columns.str.strip().str.lower().str.replace(" ", "_")
    other.columns = other.columns.str.strip().str.lower().str.replace(" ", "_")

    if "sk_id_curr" not in app.columns:
        raise KeyError("No 'sk_id_curr' found in the application DataFrame.")
    if "sk_id_curr" not in other.columns:
        raise KeyError("No 'sk_id_curr' found in the other dataset.")

    other_renamed = other.rename(
        columns={col: f"{col}_other" for col in other.columns if col != "sk_id_curr"}
    )

    merged = app.merge(other_renamed, on="sk_id_curr", how="left")

    return merged


def get_generic_preprocessor(df, impute=True):
    class DropAllNaNColumns(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self.cols_to_drop_ = X.columns[X.isna().all()].tolist()
            return self

        def transform(self, X):
            return X.drop(columns=self.cols_to_drop_, errors="ignore")

    class PandasNAToNaN(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            if isinstance(X, (pd.DataFrame, pd.Series)):
                return X.fillna(np.nan)
            return X

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    if impute:
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            [
                ("na_to_nan_converter", PandasNAToNaN()),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    else:
        numeric_transformer = "passthrough"
        categorical_transformer = Pipeline(
            [
                ("na_to_nan_converter", PandasNAToNaN()),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [("drop_all_nan_columns", DropAllNaNColumns()), ("preprocessing", preprocessor)]
    )

    return pipeline


def feature_engineering_custom(df):

    df = df.copy()

    df = convert_days_to_years(df, ["days_birth"])

    df["days_birth_years"] = (-df["days_birth"]) / 365

    df["days_birth_years_grouped"] = pd.cut(
        df["days_birth_years"],
        bins=[0, 18, 30, 50, 65, float("inf")],
        labels=["<18", "18–30", "31–50", "51–65", "65+"],
    )

    df["cnt_children_grouped"] = pd.cut(
        df["cnt_children"], bins=[-1, 0, 3, float("inf")], labels=["0", "1-3", "4+"]
    )

    df["cnt_fam_members_grouped"] = pd.cut(
        df["cnt_fam_members"],
        bins=[0, 1, 3, 5, float("inf")],
        labels=["Single", "2-3", "4-5", "6+"],
    )

    df = convert_days_to_months(
        df,
        [
            "days_employed",
            "days_id_publish",
            "days_registration",
            "days_last_phone_change",
        ],
    )

    placeholder = -11998.80
    df["days_employed_months"] = df["days_employed"] / 30.44

    df["days_employed_months_grouped"] = pd.cut(
        df["days_employed_months"].replace(placeholder, np.nan),
        bins=[-np.inf, 1, 12, 60, np.inf],
        labels=["< 1 month", "1 month - 1 year", "1-5 years", "5+ years"],
    )
    df["days_employed_months_grouped"] = df[
        "days_employed_months_grouped"
    ].cat.add_categories("Unknown")
    df.loc[
        df["days_employed_months"] == placeholder, "days_employed_months_grouped"
    ] = "Unknown"

    time_columns = {
        "days_id_publish_months": "days_id_publish_months_grouped",
        "days_registration_months": "days_registration_months_grouped",
        "days_last_phone_change_months": "days_last_phone_change_months_grouped",
    }

    for col, new_col in time_columns.items():
        df[new_col] = pd.cut(
            df[col],
            bins=[0, 1, 12, 60, float("inf")],
            labels=["< 1 month", "1 month - 1 year", "1-5 years", "5+ years"],
        )

    for col in ["amt_goods_price", "amt_credit"]:
        df[f"{col}_grouped"] = pd.cut(
            df[col],
            bins=[0, 250000, 500000, 1000000, 1500000, float("inf")],
            labels=[
                "< 250_000",
                "250_000 - 500_000",
                "500_000 - 1_000_000",
                "1_000_000 - 1_500_000",
                "1_500_000 +",
            ],
        )

    df["amt_annuity_grouped"] = pd.cut(
        df["amt_annuity"],
        bins=[0, 15000, 30000, 60000, float("inf")],
        labels=["< 15_000", "15_000 - 30_000", "30_000 - 60_000", "60_000 +"],
    )

    df["amt_income_total_grouped"] = pd.cut(
        df["amt_income_total"],
        bins=[0, 50000, 100000, 200000, 300000, float("inf")],
        labels=[
            "< 50_000",
            "50_000 - 100_000",
            "100_000 - 200_000",
            "200_000 - 300_000",
            "300_000 +",
        ],
    )

    df["application_start_time_of_day"] = pd.cut(
        df["hour_appr_process_start"],
        bins=[0, 5, 12, 18, 24],
        labels=[
            "Night (0–5)",
            "Morning (5–12)",
            "Afternoon (12–18)",
            "Evening (18–24)",
        ],
        right=False,
    )

    ext_source_cols = ["ext_source_1", "ext_source_2", "ext_source_3"]
    ext_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ext_labels = ["0 - 0.2", "0.2 - 0.4", "0.4 - 0.6", "0.6 - 0.8", "0.8 - 1.0"]

    for col in ext_source_cols:
        df[f"{col}_grouped"] = pd.cut(
            df[col], bins=ext_bins, labels=ext_labels, include_lowest=True
        )

    return df


def extract_feature_names(preprocessor, input_features):
    output_features = []
    if isinstance(preprocessor, ColumnTransformer):
        for name, transformer, columns in preprocessor.transformers_:
            if transformer == "drop":
                continue
            elif transformer == "passthrough":
                if isinstance(columns, (list, tuple)):
                    output_features.extend(columns)
                else:
                    output_features.append(columns)
            else:
                if isinstance(transformer, Pipeline):
                    last_step = transformer.steps[-1][1]
                else:
                    last_step = transformer
                if hasattr(last_step, "get_feature_names_out"):
                    try:
                        names = last_step.get_feature_names_out(columns)
                        output_features.extend(names)
                    except:
                        output_features.extend(columns)
                else:
                    output_features.extend(columns)
    elif isinstance(preprocessor, Pipeline):
        return extract_feature_names(preprocessor.steps[-1][1], input_features)
    else:
        return input_features
    return output_features


def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc_roc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC AUC: {auc_roc:.4f}")

    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    ax = plt.subplot()
    ax = sns.heatmap(cm, annot=True, fmt="", cmap="Blues")
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(["Other Cases", "Payment Difficulties"])
    ax.yaxis.set_ticklabels(["Other Cases", "Payment Difficulties"])

    plt.show()
