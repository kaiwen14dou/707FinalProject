#!/usr/bin/env python3
"""
Comprehensive EDA utilities for the MIMIC-IV-ECG metadata.

The script reproducibly generates:
    * Demographic comparisons between the full ECG corpus and the analysis cohort
      (first ECG per subject).
    * ICD group level label prevalence, co-occurrence heatmaps, and age/sex stratified rates.
    * Optional signal-derived features (heart rate, QRS width, QT and PR intervals) plus
      dimensionality-reduction diagnostics when WFDB files are reachable.
    * Example 12-lead visualisations for manual sanity checks.

Outputs are written as PNG/CSV artifacts inside the chosen output directory so that the
entire flow can be re-run on servers without mutating the preprocessing codebase.
"""

from __future__ import annotations

import argparse
import ast
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")  # Ensure headless execution on servers.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ICD_COLUMNS = [
    "ed_diag_ed",
    "ed_diag_hosp",
    "hosp_diag_hosp",
    "all_diag_hosp",
    "all_diag_all",
]

ICD_GROUPS = {
    "MI": ["I21", "I22", "I25.2"],
    "ARRHYTHMIA": ["I48", "I46", "I47", "I49"],
    "CONDUCTION": ["I44", "I45"],
    "HEART_FAILURE": ["I50"],
    "HYPERTENSION": ["I10"],
    "MITRAL_STENOSIS": ["I34.0"],
    "CARDIOMYOPATHY": ["I42"],
}

AGE_BINS = [0, 30, 40, 50, 60, 70, 80, 120]
AGE_BIN_LABELS = ["<30", "30-39", "40-49", "50-59", "60-69", "70-79", ">=80"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end EDA for ECG metadata.")
    parser.add_argument(
        "--metadata-path",
        required=True,
        type=Path,
        help="Path to records_w_diag_icd10_folds.pkl (or csv/tsv/parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("EDA/outputs"),
        help="Directory that will store plots/tables.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Optional custom label column (binary or categorical). Leave unset to rely on ICD groups only.",
    )
    parser.add_argument(
        "--subject-column",
        default="subject_id",
        help="Column identifying unique patients.",
    )
    parser.add_argument(
        "--ecg-time-column",
        default="ecg_time",
        help="Timestamp column used to pick the first ECG per subject.",
    )
    parser.add_argument(
        "--file-column",
        default="file_name",
        help="Column containing WFDB relative paths.",
    )
    parser.add_argument(
        "--wfdb-root",
        type=Path,
        default=None,
        help="Root directory for WFDB files (set to skip signal-level analysis when absent).",
    )
    parser.add_argument(
        "--max-feature-records",
        type=int,
        default=200,
        help="Maximum number of ECGs to process for signal-derived features.",
    )
    parser.add_argument(
        "--max-plot-records",
        type=int,
        default=12,
        help="Number of ECGs to render for the signal visualisation grid.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=48,
        help="Random seed used when sampling ECGs for feature extraction and plotting.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )
    sns.set_theme(style="whitegrid", context="talk")


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".pkl", ".pickle"):
        df = pd.read_pickle(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported metadata format: {suffix}")

    for col in ICD_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_coerce_icd_list)
    return df


def _coerce_icd_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, str):
        val = value.strip()
        if val.startswith("[") and val.endswith("]"):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                return []
        # Fall back to single code strings separated by spaces/commas.
        if "," in val:
            return [item.strip() for item in val.split(",") if item.strip()]
        if val:
            return [val]
    return []


def add_datetime_columns(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in metadata.")
    return pd.to_datetime(df[column], errors="coerce")


def prepare_demographics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("age", "anchor_age"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.upper().replace({"FEMALE": "F", "MALE": "M"})
    return df


def get_first_ecg_per_subject(
    df: pd.DataFrame,
    subject_col: str,
    time_col: str,
) -> pd.DataFrame:
    if subject_col not in df.columns:
        raise KeyError(f"Subject column '{subject_col}' not found.")

    ordered = df.dropna(subset=[subject_col]).copy()
    ordered = ordered.sort_values([subject_col, time_col])
    first_idx = (
        ordered.groupby(subject_col)[time_col]
        .idxmin()
        .dropna()
        .astype(int)
        .tolist()
    )
    return ordered.loc[first_idx].copy()


def normalize_icd_code(code: str) -> str:
    return code.strip().upper().replace(".", "").replace(" ", "") if isinstance(code, str) else ""


def build_icd_code_sets(df: pd.DataFrame) -> pd.Series:
    diag_cols = [col for col in ICD_COLUMNS if col in df.columns]

    def collect_codes(row) -> set[str]:
        codes: set[str] = set()
        for col in diag_cols:
            values = row.get(col, [])
            if not values:
                continue
            for code in values:
                norm = normalize_icd_code(str(code))
                if norm:
                    codes.add(norm)
        return codes

    if not diag_cols:
        logging.warning("No ICD columns found; ICD group labels will stay empty.")
        return pd.Series([set() for _ in range(len(df))], index=df.index)

    return df[diag_cols].apply(collect_codes, axis=1)


def attach_icd_group_labels(df: pd.DataFrame) -> List[str]:
    normalized_targets: Dict[str, List[str]] = {
        group: [normalize_icd_code(code) for code in codes] for group, codes in ICD_GROUPS.items()
    }

    df["__icd_code_set"] = build_icd_code_sets(df)
    created_cols: List[str] = []

    for group, prefixes in normalized_targets.items():
        col_name = f"label_{group}"
        df[col_name] = df["__icd_code_set"].apply(lambda codes: _codes_match_prefixes(codes, prefixes))
        created_cols.append(col_name)

    df.drop(columns="__icd_code_set", inplace=True)
    return created_cols


def _codes_match_prefixes(codes: set[str], prefixes: Iterable[str]) -> bool:
    for code in codes:
        for prefix in prefixes:
            if code.startswith(prefix):
                return True
    return False


def maybe_normalize_binary_label(df: pd.DataFrame, label_col: str) -> Optional[str]:
    if label_col not in df.columns:
        logging.warning("Primary label column '%s' not found; skipping.", label_col)
        return None

    series = df[label_col]
    if pd.api.types.is_bool_dtype(series):
        df[label_col] = series.astype(int)
        return label_col

    unique = set(series.dropna().unique().tolist())
    if unique.issubset({0, 1, "0", "1"}):
        try:
            df[label_col] = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
            return label_col
        except Exception:
            logging.info("Could not coerce %s to numeric despite binary-looking values.", label_col)

    logging.info(
        "Primary label column '%s' is not binary (%s unique values); it will be handled separately.",
        label_col,
        min(len(unique), 20),
    )
    return None


def summarize_multiclass_label(
    df: pd.DataFrame,
    label_col: str,
    subset_name: str,
    output_dir: Path,
) -> None:
    series = df[label_col].astype(str)
    counts = series.value_counts(dropna=False)
    table_path = output_dir / f"{subset_name}_{label_col}_distribution.csv"
    counts.to_csv(table_path, header=["count"])

    top_counts = counts.head(20)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_counts.values, y=top_counts.index, orient="h", ax=ax, color="#4472C4")
    ax.set_title(f"{label_col} distribution ({subset_name})")
    ax.set_xlabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / f"{subset_name}_{label_col}_distribution.png", dpi=200)
    plt.close(fig)


def plot_age_gender_distributions(
    full_df: pd.DataFrame,
    first_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title in zip(
        axes,
        (full_df, first_df),
        ("Full ECG cohort", "Analysis cohort (first ECG per subject)"),
    ):
        sns.histplot(
            data=data,
            x="age",
            bins=30,
            ax=ax,
            kde=False,
            color="#4C78A8",
        )
        ax.set_title(title)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "age_distribution_full_vs_first.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    gender_counts = (
        pd.DataFrame(
            {
                "Full": full_df["gender"].value_counts(dropna=False),
                "FirstECG": first_df["gender"].value_counts(dropna=False),
            }
        )
        .fillna(0)
        .astype(int)
    )
    gender_counts.plot(kind="bar", ax=ax)
    ax.set_title("Gender distribution comparison")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "gender_distribution_full_vs_first.png", dpi=200)
    plt.close(fig)


def compute_label_prevalence(df: pd.DataFrame, label_cols: List[str]) -> pd.DataFrame:
    if not label_cols:
        return pd.DataFrame()
    totals = {col: df[col].astype(int).sum() for col in label_cols}
    prevalence = pd.DataFrame(
        {
            "count": totals,
            "prevalence": {col: totals[col] / max(len(df), 1) for col in label_cols},
        }
    ).sort_values("count", ascending=False)
    return prevalence


def plot_label_prevalence(
    prevalence: pd.DataFrame,
    subset_name: str,
    output_dir: Path,
) -> None:
    if prevalence.empty:
        logging.warning("No binary label columns available for prevalence plots (%s).", subset_name)
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    prevalence = prevalence.copy()
    prevalence["label"] = prevalence.index
    sns.barplot(
        data=prevalence,
        x="label",
        y="prevalence",
        color="#F28E2B",
        ax=ax,
    )
    ax.set_title(f"Binary label prevalence ({subset_name})")
    ax.set_ylabel("Prevalence (fraction)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / f"{subset_name}_label_prevalence.png", dpi=200)
    plt.close(fig)
    prevalence.to_csv(output_dir / f"{subset_name}_label_prevalence.csv")


def age_gender_prevalence(
    df: pd.DataFrame,
    label_cols: List[str],
    subset_name: str,
    output_dir: Path,
) -> None:
    if not label_cols or "age" not in df.columns or "gender" not in df.columns:
        logging.warning("Skipping age/gender prevalence for %s (missing columns).", subset_name)
        return

    working = df.copy()
    working["age_bin"] = pd.cut(working["age"], bins=AGE_BINS, labels=AGE_BIN_LABELS, right=False)
    age_table = (
        working.dropna(subset=["age_bin"])
        .groupby("age_bin", observed=False)[label_cols]
        .mean()
        .mul(100)
        .round(2)
    )
    age_table.to_csv(output_dir / f"{subset_name}_age_bin_prevalence.csv")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(age_table, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
    ax.set_title(f"Label prevalence by age bin (%) - {subset_name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{subset_name}_age_bin_prevalence.png", dpi=200)
    plt.close(fig)

    gender_table = (
        working.dropna(subset=["gender"])
        .groupby("gender", observed=False)[label_cols]
        .mean()
        .mul(100)
        .round(2)
    )
    gender_table.to_csv(output_dir / f"{subset_name}_gender_prevalence.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(gender_table, annot=True, fmt=".1f", cmap="RdPu", ax=ax)
    ax.set_title(f"Label prevalence by gender (%) - {subset_name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{subset_name}_gender_prevalence.png", dpi=200)
    plt.close(fig)


def plot_label_cooccurrence(
    df: pd.DataFrame,
    label_cols: List[str],
    subset_name: str,
    output_dir: Path,
) -> None:
    if len(label_cols) < 2:
        return
    matrix = df[label_cols].astype(int).T.dot(df[label_cols].astype(int))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Binary label co-occurrence counts ({subset_name})")
    fig.tight_layout()
    fig.savefig(output_dir / f"{subset_name}_label_cooccurrence.png", dpi=200)
    plt.close(fig)

    try:
        from upsetplot import UpSet, from_indicators

        binary = df[label_cols].astype(bool)
        upset_data = from_indicators(label_cols, binary)
        upset = UpSet(upset_data, subset_size="count", show_counts=True)
        fig = plt.figure(figsize=(10, 8))
        upset.plot(fig=fig)
        fig.suptitle(f"UpSet plot - {subset_name}")
        fig.savefig(output_dir / f"{subset_name}_label_upset.png", dpi=200)
        plt.close(fig)
    except ImportError:
        logging.info("upsetplot not installed; skipping UpSet plot for %s.", subset_name)
    except Exception as exc:
        logging.warning("Failed to build UpSet plot for %s (%s).", subset_name, exc)


def resolve_record_base(rel_path: str, wfdb_root: Path) -> Path:
    rel_path = str(rel_path or "").strip()
    if not rel_path:
        return wfdb_root

    candidate = Path(rel_path)
    if candidate.is_absolute():
        return candidate

    parts = candidate.parts
    if "files" in parts:
        idx = parts.index("files")
        candidate = Path(*parts[idx:])
    elif parts and parts[0] == wfdb_root.name:
        candidate = Path(*parts[1:])

    return wfdb_root / candidate


def compute_signal_features(
    df: pd.DataFrame,
    file_column: str,
    wfdb_root: Optional[Path],
    label_cols: List[str],
    max_records: int,
    rng: np.random.Generator,
) -> Optional[pd.DataFrame]:
    if wfdb_root is None:
        logging.info("WFDB root not provided; skipping signal feature extraction.")
        return None

    try:
        import wfdb
    except ImportError:
        logging.warning("wfdb is not installed. Install wfdb to enable signal features.")
        return None

    try:
        import neurokit2 as nk
    except ImportError:
        logging.warning("neurokit2 is not installed. Install neurokit2 to enable signal features.")
        return None

    available = df.dropna(subset=[file_column]).copy()
    if available.empty:
        logging.warning("File column '%s' is empty; cannot load signals.", file_column)
        return None

    sample_size = min(max_records, len(available))
    sampled = available.sample(n=sample_size, random_state=rng.integers(0, 1_000_000))
    feature_rows: List[dict] = []

    for _, row in sampled.iterrows():
        rel_path = str(row[file_column]).strip()
        if not rel_path:
            continue
        record_base = resolve_record_base(rel_path, wfdb_root)
        hea_path = record_base.with_suffix(".hea")
        if not hea_path.exists():
            logging.debug("Missing header for %s; skipping.", record_base)
            continue
        try:
            record = wfdb.rdrecord(str(record_base))
        except Exception as exc:  # pragma: no cover - wfdb errors depend on local files.
            logging.debug("Failed to read %s (%s).", record_base, exc)
            continue

        fs = getattr(record, "fs", None)
        if not fs or not math.isfinite(fs):
            logging.debug("Sampling frequency missing for %s.", record_base)
            continue

        lead_index = _select_lead_index(record)
        signal = record.p_signal[:, lead_index]

        try:
            signals, info = nk.ecg_process(signal, sampling_rate=fs)
            r_peaks = info.get("ECG_R_Peaks", [])
            if len(r_peaks) < 2:
                continue
            rates = info.get("ECG_Rate", [])
            if (rates is None or len(rates) == 0) and len(r_peaks) >= 2:
                try:
                    rates = nk.signal_rate(
                        r_peaks,
                        sampling_rate=fs,
                        desired_length=len(signal),
                    )
                except Exception:
                    rates = []
            if rates is None or len(rates) == 0:
                continue
            heart_rate = float(np.nanmedian(rates))
            _, delineate_info = nk.ecg_delineate(
                signals["ECG_Clean"],
                r_peaks,
                sampling_rate=fs,
                method="dwt",
                show=False,
            )
        except Exception as exc:  # pragma: no cover - depends on neurokit internals.
            logging.debug("neurokit2 failed on %s (%s).", record_base, exc)
            continue

        qrs_ms = _interval_duration(
            delineate_info.get("ECG_R_Onsets"),
            delineate_info.get("ECG_R_Offsets"),
            fs,
        )
        qt_ms = _interval_duration(
            delineate_info.get("ECG_R_Onsets"),
            delineate_info.get("ECG_T_Offsets"),
            fs,
        )
        pr_ms = _interval_duration(
            delineate_info.get("ECG_P_Onsets"),
            delineate_info.get("ECG_R_Onsets"),
            fs,
        )

        row_dict = {
            "record_path": rel_path,
            "heart_rate_bpm": heart_rate,
            "qrs_ms": qrs_ms,
            "qt_ms": qt_ms,
            "pr_ms": pr_ms,
        }
        for label in label_cols:
            row_dict[label] = row[label]
        feature_rows.append(row_dict)

    if not feature_rows:
        logging.warning("No signal features were extracted; check WFDB paths and dependencies.")
        return None

    features_df = pd.DataFrame(feature_rows)
    return features_df.dropna(subset=["heart_rate_bpm"])


def _select_lead_index(record) -> int:
    sig_names = getattr(record, "sig_name", []) or []
    for candidate in ("II", "V5", "V2", "I"):
        if candidate in sig_names:
            return sig_names.index(candidate)
    return 0


def _interval_duration(start_indices, end_indices, fs: float) -> float:
    if not start_indices or not end_indices or not fs:
        return float("nan")
    starts = _flatten_indices(start_indices)
    ends = _flatten_indices(end_indices)
    if not starts or not ends:
        return float("nan")
    count = min(len(starts), len(ends))
    durations = []
    for idx in range(count):
        start = starts[idx]
        end = ends[idx]
        if end is None or start is None:
            continue
        if math.isnan(start) or math.isnan(end):
            continue
        if end <= start:
            continue
        durations.append((end - start) * 1000.0 / fs)
    if not durations:
        return float("nan")
    return float(np.nanmedian(durations))


def _flatten_indices(values) -> List[float]:
    flattened: List[float] = []
    if isinstance(values, (list, tuple, np.ndarray, pd.Series)):
        iterable = values
    else:
        iterable = [values]

    for item in iterable:
        if item is None:
            continue
        if isinstance(item, (list, tuple, np.ndarray)):
            flattened.extend(_flatten_indices(item))
        else:
            flattened.append(float(item))
    return flattened


def plot_feature_distributions(
    features_df: pd.DataFrame,
    output_dir: Path,
    subset_name: str,
    reference_label: Optional[str],
) -> None:
    if features_df is None or features_df.empty:
        return

    numeric_cols = ["heart_rate_bpm", "qrs_ms", "qt_ms", "pr_ms"]
    melted = features_df.melt(id_vars=["record_path"], value_vars=numeric_cols, var_name="feature", value_name="value")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=melted, x="feature", y="value", ax=ax)
    ax.set_title(f"Signal feature distributions ({subset_name})")
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(output_dir / f"{subset_name}_signal_feature_boxplot.png", dpi=200)
    plt.close(fig)

    if reference_label and reference_label in features_df.columns:
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        axes = ax.flatten()
        for axis, feature in zip(axes, numeric_cols):
            sns.boxplot(
                data=features_df,
                x=reference_label,
                y=feature,
                ax=axis,
            )
            axis.set_title(f"{feature} vs {reference_label}")
            axis.set_xlabel(reference_label)
            axis.set_ylabel(feature)
        fig.tight_layout()
        fig.savefig(output_dir / f"{subset_name}_feature_vs_{reference_label}.png", dpi=200)
        plt.close(fig)


def plot_signal_examples(
    df: pd.DataFrame,
    file_column: str,
    wfdb_root: Optional[Path],
    output_dir: Path,
    max_examples: int,
    rng: np.random.Generator,
    label_cols: List[str],
) -> None:
    if wfdb_root is None:
        return
    try:
        import wfdb
    except ImportError:
        logging.info("wfdb missing; skipping signal visualisations.")
        return

    available = df.dropna(subset=[file_column])
    if available.empty:
        return

    sample_size = min(max_examples, len(available))
    sampled = available.sample(n=sample_size, random_state=rng.integers(0, 1_000_000))

    for idx, (_, row) in enumerate(sampled.iterrows(), start=1):
        rel_path = str(row[file_column]).strip()
        if not rel_path:
            continue
        record_base = resolve_record_base(rel_path, wfdb_root)
        hea_path = record_base.with_suffix(".hea")
        if not hea_path.exists():
            continue
        try:
            record = wfdb.rdrecord(str(record_base))
        except Exception:
            continue

        fig, axes = plt.subplots(4, 3, figsize=(18, 10), sharex=True)
        axes = axes.flatten()
        seconds = np.arange(record.p_signal.shape[0]) / record.fs
        for lead_idx, axis in enumerate(axes):
            if lead_idx >= record.p_signal.shape[1]:
                axis.axis("off")
                continue
            axis.plot(seconds, record.p_signal[:, lead_idx], linewidth=0.8)
            title = record.sig_name[lead_idx] if lead_idx < len(record.sig_name) else f"Lead {lead_idx+1}"
            axis.set_title(title)
        axes[-1].set_xlabel("Time (s)")
        label_summary = ", ".join(
            f"{col.split('label_')[-1]}={int(row.get(col, 0))}" for col in label_cols[:4]
        )
        fig.suptitle(f"ECG #{idx}: {rel_path}\n{label_summary}", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(output_dir / f"signal_example_{idx}.png", dpi=200)
        plt.close(fig)


def run_dimensionality_reduction(
    features_df: pd.DataFrame,
    reference_label: Optional[str],
    output_dir: Path,
    subset_name: str,
) -> None:
    if features_df is None or features_df.empty:
        return
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logging.info("scikit-learn missing; skipping dimensionality reduction plots.")
        return

    numeric_cols = ["heart_rate_bpm", "qrs_ms", "qt_ms", "pr_ms"]
    clean_df = features_df.dropna(subset=numeric_cols).copy()
    if clean_df.empty:
        return

    scaler = StandardScaler()
    X = scaler.fit_transform(clean_df[numeric_cols])

    pca = PCA(n_components=2, random_state=7)
    pca_coords = pca.fit_transform(X)
    tsne = TSNE(n_components=2, random_state=7, init="pca", learning_rate="auto")
    tsne_coords = tsne.fit_transform(X)

    label_series = clean_df[reference_label] if reference_label and reference_label in clean_df.columns else None

    def scatter(coords, title, filename):
        fig, ax = plt.subplots(figsize=(8, 6))
        if label_series is not None:
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=label_series, ax=ax, palette="coolwarm", s=40)
            ax.legend(title=reference_label, loc="best")
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=30)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)

    scatter(pca_coords, f"PCA embedding ({subset_name})", f"{subset_name}_pca.png")
    scatter(tsne_coords, f"t-SNE embedding ({subset_name})", f"{subset_name}_tsne.png")


def main() -> None:
    args = parse_args()
    configure_logging()
    if args.label_column:
        logging.info("Using custom label column '%s'.", args.label_column)
    else:
        logging.info("No custom label column provided; relying on ICD-derived groups only.")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(args.metadata_path)
    df = prepare_demographics(df)
    df[args.ecg_time_column] = add_datetime_columns(df, args.ecg_time_column)

    label_cols = attach_icd_group_labels(df)
    binary_primary = None
    if args.label_column:
        binary_primary = maybe_normalize_binary_label(df, args.label_column)
        if binary_primary:
            label_cols = [binary_primary] + label_cols

    logging.info("Loaded %d ECG rows.", len(df))
    first_df = get_first_ecg_per_subject(df, args.subject_column, args.ecg_time_column)
    logging.info("First-ECG cohort contains %d subjects.", len(first_df))

    plot_age_gender_distributions(df, first_df, output_dir)

    # Label prevalence, stratified for both cohorts.
    full_prev = compute_label_prevalence(df, label_cols)
    plot_label_prevalence(full_prev, "full_sample", output_dir)
    first_prev = compute_label_prevalence(first_df, label_cols)
    plot_label_prevalence(first_prev, "first_ecg", output_dir)

    age_gender_prevalence(df, label_cols, "full_sample", output_dir)
    age_gender_prevalence(first_df, label_cols, "first_ecg", output_dir)

    plot_label_cooccurrence(df, label_cols, "full_sample", output_dir)
    plot_label_cooccurrence(first_df, label_cols, "first_ecg", output_dir)

    if args.label_column and not binary_primary and args.label_column in df.columns:
        summarize_multiclass_label(df, args.label_column, "full_sample", output_dir)
        summarize_multiclass_label(first_df, args.label_column, "first_ecg", output_dir)

    rng = np.random.default_rng(args.random_seed)
    features_df = compute_signal_features(
        first_df,  # Prefer patient-level cohort for expensive computations.
        args.file_column,
        args.wfdb_root,
        label_cols,
        args.max_feature_records,
        rng,
    )
    reference_label = "label_ARRHYTHMIA" if "label_ARRHYTHMIA" in label_cols else (label_cols[0] if label_cols else None)
    plot_feature_distributions(features_df, output_dir, "first_ecg", reference_label)
    run_dimensionality_reduction(features_df, reference_label, output_dir, "first_ecg")

    plot_signal_examples(
        first_df,
        args.file_column,
        args.wfdb_root,
        output_dir,
        args.max_plot_records,
        rng,
        label_cols,
    )

    logging.info("EDA artifacts saved under %s", output_dir)


if __name__ == "__main__":
    main()
