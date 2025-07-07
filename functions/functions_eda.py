import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import scipy.stats as stats
from scipy.stats import mode
import math
import re
import gc
from matplotlib.ticker import ScalarFormatter
from statsmodels.stats.proportion import proportion_confint
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from pandas.api.types import CategoricalDtype
import phik
from scipy.stats import norm
import warnings


def show_head_and_info(df, n=5):
    print("Data")
    display(df.head(n))
    print("\nInfo")
    df.info()


def count_clients_with_multiple_records(df, client_id_col="sk_id_curr"):
    client_counts = df[client_id_col].value_counts()
    clients_with_multiple = client_counts[client_counts > 1]
    return clients_with_multiple.shape[0]


def missing_value_summary(df, sort_by="Missing Count", ascending=False):
    summary = pd.DataFrame(
        {
            "Missing Count": df.isnull().sum(),
            "Missing %": (df.isnull().sum() / len(df) * 100).round(2),
        }
    )

    return (
        summary[summary["Missing Count"] > 0]
        .sort_values(by=sort_by, ascending=ascending)
        .head(10)
    )


def fill_missing_values(df, group_col):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns.drop(
        group_col, errors="ignore"
    )
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            medians = df.groupby(group_col)[col].transform("median")
            df[col] = df[col].fillna(medians)

    for col in cat_cols:
        if df[col].isna().sum() > 0:

            def get_mode(series):
                values, counts = np.unique(series.dropna(), return_counts=True)
                return values[np.argmax(counts)] if len(values) > 0 else np.nan

            modes = df.groupby(group_col)[col].transform(get_mode)
            df[col] = df[col].fillna(modes)

    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna("Unknown")

    return df


def print_unique_counts(df, features):
    for feature in features:
        unique_count = df[feature].nunique(dropna=False)
        print(f"{feature}: {unique_count} unique values")


def value_counts_rounded(df, column, decimals=2):
    return df[column].value_counts().round(decimals)


def plot_numeric_distribution(
    df, columns, title=None, figsize=(7, 4), color="steelblue", bins=50
):
    if isinstance(columns, str):
        columns = [columns]

    cols_per_row = 3
    rows = math.ceil(len(columns) / cols_per_row)
    fig, axes = plt.subplots(
        rows, cols_per_row, figsize=(figsize[0] * cols_per_row, figsize[1] * rows)
    )
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        sns.histplot(df[col].dropna(), bins=bins, color=color, ax=ax)
        ax.set_title(title or f"Count of {col}", fontsize=14)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.grid(True)
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style="plain", axis="x")
        ax.ticklabel_format(style="plain", axis="y")

    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_categorical_distribution(
    df, columns, title=None, figsize=(7, 4), color="skyblue"
):
    if isinstance(columns, str):
        columns = [columns]

    cols_per_row = 3
    rows = math.ceil(len(columns) / cols_per_row)
    fig, axes = plt.subplots(
        rows, cols_per_row, figsize=(figsize[0] * cols_per_row, figsize[1] * rows)
    )
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        counts = df[col].value_counts(dropna=False).sort_index()
        bars = sns.barplot(
            x=counts.index.astype(str), y=counts.values, color=color, ax=ax
        )

        ax.set_title(title or f"Distribution of {col}", fontsize=14)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(True, axis="y")
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style="plain", axis="y")

        for bar in bars.patches:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def detect_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"Outliers detected in '{col}': {outliers.shape[0]}")


def iqr_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"Outliers detected in '{col}': {outliers.shape[0]}")


def remove_outliers_by_std(df, columns, num_std=3):
    df_filtered = df.copy()
    for col in columns:
        mean = df_filtered[col].mean()
        std = df_filtered[col].std()
        lower_bound = mean - num_std * std
        upper_bound = mean + num_std * std
        df_filtered = df_filtered[
            (df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)
        ]
    return df_filtered


def convert_days_to_months(df, cols):
    for col in cols:
        df[f"{col}_months"] = (-df[col] / 30.44).round(1)
    return df


def mode_agg(series):
    modes = series.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    else:
        return pd.NA


def aggregate_bureau_with_balance(bureau_df, bureau_balance_df):
    merged = bureau_df.merge(bureau_balance_df, on="sk_id_bureau", how="left")

    numeric_cols = (
        merged.select_dtypes(include="number")
        .columns.difference(["sk_id_curr", "sk_id_bureau"])
        .tolist()
    )
    cat_cols = (
        merged.select_dtypes(exclude="number")
        .columns.difference(["sk_id_curr", "sk_id_bureau"])
        .tolist()
    )

    agg_dict = {col: "mean" for col in numeric_cols}
    agg_dict.update({col: mode_agg for col in cat_cols})

    aggregated = merged.groupby("sk_id_curr").agg(agg_dict).reset_index()
    return aggregated


def aggregate_datasets(application_df, prev_df):
    orig_dtypes = application_df.dtypes.to_dict()

    merged = application_df.merge(prev_df, on="sk_id_curr", how="left")
    exclude_cols = ["sk_id_curr", "sk_id_prev"]

    numeric_cols = (
        merged.select_dtypes(include="number").columns.difference(exclude_cols).tolist()
    )
    cat_cols = (
        merged.select_dtypes(exclude="number").columns.difference(exclude_cols).tolist()
    )

    agg_dict = {col: "mean" for col in numeric_cols}
    agg_dict.update({col: mode_agg for col in cat_cols})

    aggregated = merged.groupby("sk_id_curr").agg(agg_dict).reset_index()

    for col in aggregated.columns:
        if col in orig_dtypes:
            orig_dtype = orig_dtypes[col]

            if pd.api.types.is_integer_dtype(
                orig_dtype
            ) and pd.api.types.is_float_dtype(aggregated[col]):
                if (aggregated[col] % 1 == 0).all(skipna=True):
                    aggregated[col] = aggregated[col].astype("Int64")

    return aggregated


def phik_correlations_with_pval(df, features, target_col="target"):
    correlation_results = {}

    df[target_col] = df[target_col].astype(int)

    for col in features:
        if col not in df.columns:
            print(f"Skipping '{col}' — not found in DataFrame.")
            continue
        try:
            sub_df = df[[col, target_col]].dropna().copy()

            interval_cols = [col] if pd.api.types.is_numeric_dtype(df[col]) else []

            phik_corr_matrix = sub_df.phik_matrix(interval_cols=interval_cols)
            corr = phik_corr_matrix.loc[col, target_col]

            z_matrix = sub_df.significance_matrix(interval_cols=interval_cols)
            z_val = z_matrix.loc[col, target_col]

            p_val = 2 * (1 - norm.cdf(abs(z_val)))

            correlation_results[col] = (corr, p_val)

        except Exception as e:
            print(f"Error computing Phi-k for '{col}': {e}")

    correlation_df = pd.DataFrame(
        correlation_results, index=["Correlation", "P-Value"]
    ).T
    correlation_df = correlation_df.sort_values(
        by="Correlation", ascending=False
    ).round(4)

    return correlation_df
