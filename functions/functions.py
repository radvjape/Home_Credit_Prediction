import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from textwrap import wrap
import pandas as pd
import numpy as np
from phik import resources
import phik
from scipy.stats import norm


def show_head_and_info(df, n=5):
    print("Data")
    display(df.head(n))
    print("\nInfo")
    df.info()


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


def fill_missing_values(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

            if pd.api.types.is_integer_dtype(df[col].dropna()):
                df[col] = df[col].round().astype(int)

    return df


def iqr_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"Outliers detected in '{col}': {outliers.shape[0]}")


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


def convert_days_to_years(df, cols):
    for col in cols:
        df[f"{col}_years"] = (-df[col] / 365.25).round(1)
    return df


def print_value_counts_by_target_pct(df, features, target_col="target"):
    for feature in features:
        print(f"\n{feature} value counts by target percentage (sorted by target=1):")
        pct = (
            df.groupby(feature)[target_col]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
            * 100
        )
        pct = pct.sort_values(by=1, ascending=False)
        print(pct.round(2))


def plot_numeric_features(
    df, features, target_col="target", cols=4, fontsize=13, title_fontsize=15, bins=50
):
    rows = math.ceil(len(features) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten()

    for i, col in enumerate(features):
        ax = axes[i]

        sns.histplot(
            data=df[df[target_col] == 0],
            x=col,
            stat="percent",
            bins=bins,
            label="Other Cases",
            color="salmon",
            alpha=0.5,
            ax=ax,
            element="step",
        )

        sns.histplot(
            data=df[df[target_col] == 1],
            x=col,
            stat="percent",
            bins=bins,
            label="Client with Payment Difficulties",
            color="mediumseagreen",
            alpha=0.5,
            ax=ax,
            element="step",
        )

        ax.set_ylim(0, 100)
        ax.set_title(f"Percentage Distribution of {col}", fontsize=title_fontsize)
        ax.set_xlabel(col, fontsize=fontsize)
        ax.set_ylabel("Percentage", fontsize=fontsize)

        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        ax.tick_params(axis="x", rotation=45)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax.ticklabel_format(style="plain", axis="x")
        ax.ticklabel_format(style="plain", axis="y")

    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.show()


def plot_numeric_few_values_features(
    df, binary_features, target_col="target", cols=3, fontsize=10, title_fontsize=15
):

    rows = math.ceil(len(binary_features) / cols)
    plt.figure(figsize=(6 * cols, 4 * rows))

    for i, col in enumerate(binary_features, 1):
        ax = plt.subplot(rows, cols, i)
        pct = (
            df.groupby(col)[target_col].value_counts(normalize=True).unstack().fillna(0)
            * 100
        )
        pct.index = pct.index.astype(str)

        sns.barplot(x=pct.index, y=pct[0], color="salmon", ax=ax)
        sns.barplot(x=pct.index, y=pct[1], bottom=pct[0], color="mediumseagreen", ax=ax)

        for idx in pct.index:
            ax.text(
                idx,
                pct.loc[idx, 0] / 2,
                f"{pct.loc[idx, 0]:.1f}%",
                ha="center",
                va="center",
                fontsize=fontsize,
            )
            ax.text(
                idx,
                pct.loc[idx, 0] + pct.loc[idx, 1] / 2,
                f"{pct.loc[idx, 1]:.1f}%",
                ha="center",
                va="center",
                fontsize=fontsize,
            )

        ax.set_title(f"{col} vs {target_col}", fontsize=title_fontsize)
        ax.set_xlabel(col, fontsize=fontsize)
        ax.set_ylabel("Percentage", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()


def plot_categorical_features(
    df,
    cat_features,
    target_col="target",
    cols=3,
    wrap_width=9,
    fontsize=13,
    title_fontsize=15,
):

    rows = math.ceil(len(cat_features) / cols)
    plt.figure(figsize=(6 * cols, 4 * rows))

    for i, col in enumerate(cat_features, 1):
        ax = plt.subplot(rows, cols, i)
        pct = (
            df.groupby(col)[target_col].value_counts(normalize=True).unstack().fillna(0)
            * 100
        )

        sns.barplot(x=pct.index, y=pct[0], color="salmon")
        sns.barplot(x=pct.index, y=pct[1], bottom=pct[0], color="mediumseagreen")

        for idx, val in enumerate(pct.index):
            y0, y1 = pct[0][val], pct[1][val]
            plt.text(
                idx, y0 / 2, f"{y0:.1f}%", ha="center", va="center", fontsize=fontsize
            )
            plt.text(
                idx,
                y0 + y1 / 2,
                f"{y1:.1f}%",
                ha="center",
                va="center",
                fontsize=fontsize,
            )

        wrapped_labels = ["\n".join(wrap(label, wrap_width)) for label in pct.index]
        ax.set_xticklabels(wrapped_labels, fontsize=fontsize, rotation=45, ha="right")
        ax.set_xticklabels(wrapped_labels, fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        ax.set_xlabel(col, fontsize=fontsize)
        ax.set_ylabel("Percentage", fontsize=fontsize)
        ax.set_title(f"{col} vs {target_col.capitalize()}", fontsize=title_fontsize)
        ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()
