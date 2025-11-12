import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

# Use a non-interactive backend to allow saving plots without a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re

try:
    from scipy.stats import chi2  # optional, used for p-value
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def _sanitize_model_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-._")
    return sanitized or "unknown_model"


def extract_model_name_from_filename(csv_file_path: str) -> str:
    """Extract model name from CSV filename, if present.

    Handles common patterns, e.g.:
      - wlasl_predictions_<model>_<YYYYMMDD_HHMMSS>.csv
      - <model>_predictions_<YYYYMMDD_HHMMSS>.csv
    Falls back to 'unknown_model' if not detected.
    """
    base = os.path.basename(csv_file_path)

    patterns = [
        r"^wlasl_predictions_(?P<model>.+?)_(?P<ts>\d{8}_\d{6})\.csv$",
        r"^(?P<model>.+?)_predictions_(?P<ts>\d{8}_\d{6})\.csv$",
        r"^(?P<model>.+?)_prediction_(?P<ts>\d{8}_\d{6})\.csv$",
    ]

    for pat in patterns:
        m = re.match(pat, base)
        if m:
            return _sanitize_model_name(m.group("model"))

    # Fallback: try to strip trailing timestamp if present and remove known prefixes
    ts_m = re.search(r"(\d{8}_\d{6})", base)
    candidate = base
    if ts_m:
        candidate = base[: ts_m.start()]  # up to the timestamp
        candidate = candidate.rstrip("_-")
    # remove common prefixes/suffixes
    candidate = re.sub(r"^(wlasl_)?predictions?_?", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.rstrip("_-")
    if candidate:
        return _sanitize_model_name(candidate)
    return "unknown_model"


def analyze_prediction_distribution(
    csv_file_path: str,
    prediction_column: str = "predicted_gloss",
    output_dir: str = "output",
    save_plot: bool = True,
    top_k: int = 10,
):
    """Analyze distribution of model predictions from a CSV file.

    The function:
    - loads the predictions CSV (expects a column with predicted labels),
    - filters out rows with prediction errors (values starting with "ERROR:"),
    - normalizes predictions to lowercase/trimmed for counting,
    - computes concentration metrics (entropy, normalized entropy, Gini/HHI),
    - performs a chi-square goodness-of-fit test vs uniform distribution,
    - saves distribution CSV and plots to the output directory,
    - prints a short diagnosis of uniformity vs concentration.
    """

    if not os.path.exists(csv_file_path):
        print(f"Error: File '{csv_file_path}' not found!")
        return None

    print(f"Loading predictions from: {csv_file_path}")
    df = pd.read_csv(csv_file_path)

    if prediction_column not in df.columns:
        print(f"Error: Column '{prediction_column}' not found in CSV! Columns available: {list(df.columns)}")
        return None

    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Filter out error rows
    pred_as_str = df[prediction_column].astype(str)
    error_mask = pred_as_str.str.startswith("ERROR:")
    error_count = int(error_mask.sum())
    if error_count > 0:
        print(f"Found {error_count} error entries, excluding them from analysis")
    df_clean = df.loc[~error_mask].copy()

    if len(df_clean) == 0:
        print("No valid predictions found after filtering errors!")
        return None

    # Normalize predictions for consistent counting
    df_clean["pred_normalized"] = df_clean[prediction_column].astype(str).str.lower().str.strip()

    # Count distribution
    counts = df_clean["pred_normalized"].value_counts()
    total_predictions = int(counts.sum())
    unique_predicted = int(counts.shape[0])
    proportions = counts / total_predictions

    # Metrics
    # Shannon entropy (base-2) and normalized entropy
    # add small epsilon to avoid log(0), though proportions>0 by construction
    entropy = float(-(proportions * np.log2(proportions)).sum()) if unique_predicted > 0 else 0.0
    max_entropy = float(np.log2(unique_predicted)) if unique_predicted > 0 else 0.0
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # Concentration metrics: HHI = sum(p^2), Gini-like (1 - HHI)
    hhi = float((proportions ** 2).sum())
    gini_like = float(1.0 - hhi)

    # Top-k coverage
    top1_share = float(proportions.iloc[0]) if unique_predicted >= 1 else 0.0
    topk_share = float(proportions.iloc[: min(top_k, unique_predicted)].sum()) if unique_predicted > 0 else 0.0

    # Chi-square goodness-of-fit vs uniform distribution over observed labels
    chi2_stat = None
    chi2_df = None
    chi2_pvalue = None
    if unique_predicted > 1:
        expected = np.full(shape=unique_predicted, fill_value=total_predictions / unique_predicted)
        observed = counts.values.astype(float)
        chi2_stat = float(((observed - expected) ** 2 / expected).sum())
        chi2_df = int(unique_predicted - 1)
        if SCIPY_AVAILABLE:
            chi2_pvalue = float(chi2.sf(chi2_stat, chi2_df))

    # Simple diagnosis
    diagnosis = ""
    if unique_predicted <= 1 or topk_share >= 0.9:
        diagnosis = "Extremely concentrated: predictions collapse to very few glosses"
    elif normalized_entropy >= 0.9 and topk_share <= 0.5:
        diagnosis = "Near-uniform distribution across predicted glosses"
    elif normalized_entropy >= 0.7 and topk_share <= 0.7:
        diagnosis = "Moderately mixed distribution"
    else:
        diagnosis = "Concentrated on a subset of glosses (possible a priori bias)"

    # Prepare output directory and a timestamped subfolder for this run (with model name)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = extract_model_name_from_filename(csv_file_path)
    run_dir = os.path.join(output_dir, f"{model_name}_prediction_analysis_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save distribution CSV
    dist_df = (
        counts.rename("count").reset_index().rename(columns={"index": "predicted_gloss"})
    )
    dist_df["percentage"] = dist_df["count"] / total_predictions
    dist_csv_path = os.path.join(run_dir, f"{model_name}_prediction_distribution_{timestamp}.csv")
    dist_df.to_csv(dist_csv_path, index=False, encoding="utf-8")

    # Save summary TXT
    summary_lines = [
        "=== PREDICTION DISTRIBUTION ANALYSIS ===",
        f"Source CSV: {csv_file_path}",
        f"Model: {model_name}",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total valid predictions: {total_predictions}",
        f"Unique predicted glosses: {unique_predicted}",
        "",
        f"Entropy (bits): {entropy:.4f}",
        f"Normalized entropy [0-1]: {normalized_entropy:.4f}",
        f"HHI (sum p^2) [1/K-1]: {hhi:.6f}",
        f"1 - HHI (higher means more even): {gini_like:.6f}",
        f"Top-1 share: {top1_share:.4%}",
        f"Top-{top_k} coverage: {topk_share:.4%}",
    ]
    if chi2_stat is not None:
        summary_lines.append(f"Chi-square vs uniform: statistic={chi2_stat:.3f}, dof={chi2_df}")
        if chi2_pvalue is not None:
            summary_lines.append(f"Chi-square p-value: {chi2_pvalue:.3e}")
        else:
            summary_lines.append("Chi-square p-value: (install SciPy to compute)")
    summary_lines.extend(["", f"Diagnosis: {diagnosis}", "", f"Saved distribution CSV: {dist_csv_path}"])

    summary_path = os.path.join(run_dir, f"{model_name}_prediction_distribution_summary_{timestamp}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # Plots
    plot_paths = []
    if save_plot and unique_predicted > 0:
        # Full distribution (may be wide)
        plt.figure(figsize=(max(10, min(24, unique_predicted * 0.15)), 6))
        counts.plot(kind="bar")
        plt.title(f"Distribution of Predictions (All Glosses) — {model_name}")
        plt.xlabel("Predicted Gloss")
        plt.ylabel("Frequency")
        plt.xticks(rotation=90)
        plt.tight_layout()
        full_plot_path = os.path.join(run_dir, f"{model_name}_prediction_distribution_full_{timestamp}.png")
        plt.savefig(full_plot_path, dpi=200)
        plt.close()
        plot_paths.append(full_plot_path)

        # Top-N distribution
        top_n = min(30, unique_predicted)
        plt.figure(figsize=(max(10, top_n * 0.4), 6))
        counts.head(top_n).plot(kind="bar", color="#1f77b4")
        plt.title(f"Top {top_n} Predicted Glosses — {model_name}")
        plt.xlabel("Predicted Gloss")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        top_plot_path = os.path.join(run_dir, f"{model_name}_prediction_distribution_top{top_n}_{timestamp}.png")
        plt.savefig(top_plot_path, dpi=200)
        plt.close()
        plot_paths.append(top_plot_path)

    # Console output
    print("\n" + "=" * 50)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Total valid predictions: {total_predictions}")
    print(f"Unique predicted glosses: {unique_predicted}")
    print(f"Entropy (bits): {entropy:.4f}")
    print(f"Normalized entropy [0-1]: {normalized_entropy:.4f}")
    print(f"HHI (sum p^2): {hhi:.6f}")
    print(f"1 - HHI: {gini_like:.6f}")
    print(f"Top-1 share: {top1_share:.2%}")
    print(f"Top-{top_k} coverage: {topk_share:.2%}")
    if chi2_stat is not None:
        print(f"Chi-square vs uniform: statistic={chi2_stat:.3f}, dof={chi2_df}")
        if chi2_pvalue is not None:
            print(f"Chi-square p-value: {chi2_pvalue:.3e}")
        else:
            print("Chi-square p-value: (install SciPy to compute)")
    print(f"Diagnosis: {diagnosis}")
    print(f"Saved distribution CSV: {dist_csv_path}")
    print(f"Saved summary: {summary_path}")
    if plot_paths:
        for p in plot_paths:
            print(f"Saved plot: {p}")
    print(f"Run folder: {run_dir}")

    return {
        "total_valid_predictions": total_predictions,
        "unique_predicted": unique_predicted,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "hhi": hhi,
        "one_minus_hhi": gini_like,
        "top1_share": top1_share,
        "topk_share": topk_share,
        "chi2_stat": chi2_stat,
        "chi2_df": chi2_df,
        "chi2_pvalue": chi2_pvalue,
        "diagnosis": diagnosis,
        "distribution_csv": dist_csv_path,
        "summary_txt": summary_path,
        "plot_paths": plot_paths,
        "model_name": model_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze prediction distribution from a predictions CSV")
    parser.add_argument("csv_file", help="Path to the predictions CSV file")
    parser.add_argument(
        "--column",
        default="predicted_gloss",
        help="Name of the column containing predictions (default: predicted_gloss)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save analysis outputs (default: output/)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable saving plots",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K coverage to report (default: 10)",
    )

    args = parser.parse_args()

    print("Prediction Distribution Evaluation")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    results = analyze_prediction_distribution(
        csv_file_path=args.csv_file,
        prediction_column=args.column,
        output_dir=args.output_dir,
        save_plot=not args.no_plot,
        top_k=args.top_k,
    )

    if results is None:
        print("Analysis failed!")
    else:
        print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()


