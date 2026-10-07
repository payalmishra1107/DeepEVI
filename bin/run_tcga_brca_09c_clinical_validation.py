#!/usr/bin/env python3

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceWarning
except ImportError as exc:
    raise RuntimeError(
        "lifelines is required for Step 09C"
    ) from exc


SCORE_COLUMN = "deep_evi_tcga_surrogate"

STAGE_GROUP_ORDER = [
    "Stage 0/I",
    "Stage II",
    "Stage III",
    "Stage IV",
]


def load_table(path, sep=","):
    path = Path(path)

    if not path.exists():
        raise RuntimeError(f"Input file does not exist: {path}")

    df = pd.read_csv(
        path,
        sep=sep,
        dtype={"case_id": str},
    )

    df.columns = [str(c).strip() for c in df.columns]

    if "case_id" in df.columns:
        df["case_id"] = df["case_id"].astype(str).str.strip()

    return df


def require_columns(df, columns, name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def clean_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_stage(value):
    if pd.isna(value):
        return np.nan

    x = str(value).strip().upper()

    if x in {"", "NAN", "NONE", "UNKNOWN", "NOT REPORTED", "NOT AVAILABLE"}:
        return np.nan

    # Remove redundant whitespace.
    x = " ".join(x.split())

    # Normalize common representations.
    x = x.replace("STAGE", "").strip()

    mapping = {
        "0": "Stage 0/I",
        "I": "Stage 0/I",
        "IA": "Stage 0/I",
        "IB": "Stage 0/I",

        "II": "Stage II",
        "IIA": "Stage II",
        "IIB": "Stage II",

        "III": "Stage III",
        "IIIA": "Stage III",
        "IIIB": "Stage III",
        "IIIC": "Stage III",

        "IV": "Stage IV",
    }

    return mapping.get(x, np.nan)


def make_stage_group(df):
    if "ajcc_pathologic_stage" not in df.columns:
        df["ajcc_stage_group"] = np.nan
        return df

    df["ajcc_stage_group"] = df["ajcc_pathologic_stage"].apply(
        normalize_stage
    )

    df["ajcc_stage_group"] = pd.Categorical(
        df["ajcc_stage_group"],
        categories=STAGE_GROUP_ORDER,
        ordered=True,
    )

    return df


def cox_result_empty(status, note=None):
    result = {
        "status": status,
        "n": None,
        "events": None,
        "coef": None,
        "hazard_ratio": None,
        "p_value": None,
        "ci_lower_hr": None,
        "ci_upper_hr": None,
        "concordance": None,
    }

    if note is not None:
        result["note"] = note

    return result


def fit_univariate_cox(df):
    required = [
        "survival_time",
        "survival_event",
        SCORE_COLUMN,
    ]

    work = df.dropna(subset=required).copy()

    if len(work) < 30:
        return cox_result_empty(
            "insufficient_cases",
            f"Only {len(work)} complete cases available."
        )

    if work["survival_event"].sum() < 5:
        return cox_result_empty(
            "insufficient_events",
            f"Only {int(work['survival_event'].sum())} events available."
        )

    model_df = work[
        ["survival_time", "survival_event", SCORE_COLUMN]
    ].copy()

    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

    cph = CoxPHFitter()

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)

        try:
            cph.fit(
                model_df,
                duration_col="survival_time",
                event_col="survival_event",
            )
        except ConvergenceWarning as exc:
            return cox_result_empty(
                "convergence_warning",
                str(exc),
            )
        except Exception as exc:
            return cox_result_empty(
                "fit_failed",
                str(exc),
            )

    row = cph.summary.loc[SCORE_COLUMN]

    return {
        "status": "complete",
        "n": int(len(model_df)),
        "events": int(model_df["survival_event"].sum()),
        "coef": float(row["coef"]),
        "hazard_ratio": float(np.exp(row["coef"])),
        "p_value": float(row["p"]),
        "ci_lower_hr": float(
            np.exp(row["coef lower 95%"])
        ),
        "ci_upper_hr": float(
            np.exp(row["coef upper 95%"])
        ),
        "concordance": float(cph.concordance_index_),
    }


def fit_adjusted_cox(df):
    required = [
        "survival_time",
        "survival_event",
        SCORE_COLUMN,
        "age_at_index",
        "ajcc_stage_group",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        return cox_result_empty(
            "missing_covariates",
            f"Missing columns: {missing}",
        )

    work = df[required].copy()

    work["survival_time"] = pd.to_numeric(
        work["survival_time"], errors="coerce"
    )
    work["survival_event"] = pd.to_numeric(
        work["survival_event"], errors="coerce"
    )
    work[SCORE_COLUMN] = pd.to_numeric(
        work[SCORE_COLUMN], errors="coerce"
    )
    work["age_at_index"] = pd.to_numeric(
        work["age_at_index"], errors="coerce"
    )

    work = work.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(work) < 50:
        return cox_result_empty(
            "insufficient_complete_cases",
            f"Only {len(work)} complete adjusted cases available.",
        )

    if work["survival_event"].sum() < 10:
        return cox_result_empty(
            "insufficient_events",
            f"Only {int(work['survival_event'].sum())} events available.",
        )

    # Explicitly convert grouped AJCC stage to dummy variables.
    stage_dummies = pd.get_dummies(
        work["ajcc_stage_group"],
        prefix="ajcc_stage_group",
        drop_first=True,
        dtype=float,
    )

    model_df = pd.concat(
        [
            work[
                [
                    "survival_time",
                    "survival_event",
                    SCORE_COLUMN,
                    "age_at_index",
                ]
            ],
            stage_dummies,
        ],
        axis=1,
    )

    model_df = model_df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    # Remove zero-variance covariates defensively.
    predictor_columns = [
        c for c in model_df.columns
        if c not in {"survival_time", "survival_event"}
    ]

    zero_variance = [
        c for c in predictor_columns
        if model_df[c].nunique(dropna=True) <= 1
    ]

    if zero_variance:
        model_df = model_df.drop(columns=zero_variance)

    predictor_columns = [
        c for c in model_df.columns
        if c not in {"survival_time", "survival_event"}
    ]

    if SCORE_COLUMN not in predictor_columns:
        return cox_result_empty(
            "score_removed",
            "Deep-EVI surrogate was removed unexpectedly.",
        )

    cph = CoxPHFitter()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)

            cph.fit(
                model_df,
                duration_col="survival_time",
                event_col="survival_event",
            )

    except ConvergenceWarning as exc:
        return cox_result_empty(
            "convergence_warning",
            str(exc),
        )

    except Exception as exc:
        return cox_result_empty(
            "fit_failed",
            str(exc),
        )

    row = cph.summary.loc[SCORE_COLUMN]

    return {
        "status": "complete",
        "n": int(len(model_df)),
        "events": int(model_df["survival_event"].sum()),
        "coef": float(row["coef"]),
        "hazard_ratio": float(np.exp(row["coef"])),
        "p_value": float(row["p"]),
        "ci_lower_hr": float(
            np.exp(row["coef lower 95%"])
        ),
        "ci_upper_hr": float(
            np.exp(row["coef upper 95%"])
        ),
        "concordance": float(cph.concordance_index_),
        "covariates": predictor_columns,
        "stage_group_definition": {
            "Stage 0/I": [
                "Stage 0",
                "Stage I",
                "Stage IA",
                "Stage IB",
            ],
            "Stage II": [
                "Stage II",
                "Stage IIA",
                "Stage IIB",
            ],
            "Stage III": [
                "Stage III",
                "Stage IIIA",
                "Stage IIIB",
                "Stage IIIC",
            ],
            "Stage IV": [
                "Stage IV",
            ],
        },
        "excluded_stage_values_from_adjusted_model": [
            "Stage X",
            "unknown",
            "missing",
            "unrecognized",
        ],
        "zero_variance_covariates_removed": zero_variance,
    }


def proportional_hazards_check(df):
    required = [
        "survival_time",
        "survival_event",
        SCORE_COLUMN,
        "age_at_index",
        "ajcc_stage_group",
    ]

    if not all(c in df.columns for c in required):
        return {
            "status": "not_run",
            "note": "Required adjusted-model columns unavailable.",
        }

    work = df[required].copy()

    work["survival_time"] = pd.to_numeric(
        work["survival_time"], errors="coerce"
    )
    work["survival_event"] = pd.to_numeric(
        work["survival_event"], errors="coerce"
    )
    work[SCORE_COLUMN] = pd.to_numeric(
        work[SCORE_COLUMN], errors="coerce"
    )
    work["age_at_index"] = pd.to_numeric(
        work["age_at_index"], errors="coerce"
    )

    work = work.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(work) < 50:
        return {
            "status": "insufficient_cases",
            "n": int(len(work)),
        }

    if work["survival_event"].sum() < 10:
        return {
            "status": "insufficient_events",
            "n": int(len(work)),
        }

    stage_dummies = pd.get_dummies(
        work["ajcc_stage_group"],
        prefix="ajcc_stage_group",
        drop_first=True,
        dtype=float,
    )

    model_df = pd.concat(
        [
            work[
                [
                    "survival_time",
                    "survival_event",
                    SCORE_COLUMN,
                    "age_at_index",
                ]
            ],
            stage_dummies,
        ],
        axis=1,
    )

    model_df = model_df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    # Remove zero-variance predictors.
    for col in list(model_df.columns):
        if col in {"survival_time", "survival_event"}:
            continue
        if model_df[col].nunique(dropna=True) <= 1:
            model_df = model_df.drop(columns=col)

    cph = CoxPHFitter()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)

            cph.fit(
                model_df,
                duration_col="survival_time",
                event_col="survival_event",
            )

            cph.check_assumptions(
                model_df,
                p_value_threshold=0.05,
                show_plots=False,
            )

        return {
            "status": "completed",
            "n": int(len(model_df)),
            "events": int(model_df["survival_event"].sum()),
            "note": (
                "Proportional-hazards diagnostic executed "
                "without a convergence warning."
            ),
        }

    except ConvergenceWarning as exc:
        return {
            "status": "convergence_warning",
            "n": int(len(model_df)),
            "events": int(model_df["survival_event"].sum()),
            "note": str(exc),
        }

    except Exception as exc:
        return {
            "status": "diagnostic_failed",
            "n": int(len(model_df)),
            "events": int(model_df["survival_event"].sum()),
            "note": str(exc),
        }


def context_correlations(df, cohort_name, outdir):
    context_columns = [
        "tcell_context_score",
        "cytotoxic_context_score",
        "exhaustion_context_score",
    ]

    rows = []

    for col in context_columns:
        if col not in df.columns:
            continue

        work = df[[SCORE_COLUMN, col]].copy()
        work = work.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if len(work) < 10:
            continue

        rho, p = spearmanr(
            work[SCORE_COLUMN],
            work[col],
        )

        rows.append(
            {
                "cohort": cohort_name,
                "context_score": col,
                "n": int(len(work)),
                "spearman_rho": float(rho),
                "spearman_p": float(p),
            }
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        outdir / f"09C_{cohort_name}_context_correlations.csv",
        index=False,
    )

    return result


def characterize_clinical(df, outdir):
    rows = []

    for col in [
        "vital_status",
        "ajcc_pathologic_stage",
        "ajcc_stage_group",
        "primary_diagnosis",
        "classification_of_tumor",
    ]:
        if col not in df.columns:
            continue

        grouped = (
            df.groupby(
                col,
                dropna=False,
                observed=False,
            )[SCORE_COLUMN]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
        )

        for _, row in grouped.iterrows():
            rows.append(
                {
                    "variable": col,
                    "category": (
                        "NA"
                        if pd.isna(row[col])
                        else str(row[col])
                    ),
                    "n": int(row["count"]),
                    "score_mean": (
                        float(row["mean"])
                        if pd.notna(row["mean"])
                        else None
                    ),
                    "score_median": (
                        float(row["median"])
                        if pd.notna(row["median"])
                        else None
                    ),
                    "score_std": (
                        float(row["std"])
                        if pd.notna(row["std"])
                        else None
                    ),
                }
            )

    result = pd.DataFrame(rows)

    result.to_csv(
        outdir / "GSE176078_09C_clinical_characterization.csv",
        index=False,
    )

    return result


def analyze_cohort(df, cohort_name):
    result = {
        "cohort": cohort_name,
        "n_cases": int(df["case_id"].nunique()),
        "n_survival_time": int(
            df["survival_time"].notna().sum()
        ),
        "n_events": int(
            pd.to_numeric(
                df["survival_event"],
                errors="coerce",
            ).fillna(0).sum()
        ),
        "score_mean": float(
            df[SCORE_COLUMN].mean()
        ),
        "score_median": float(
            df[SCORE_COLUMN].median()
        ),
        "score_std": float(
            df[SCORE_COLUMN].std()
        ),
    }

    result["univariate_cox"] = fit_univariate_cox(df)

    if cohort_name == "multi_file_descriptive":
        result["adjusted_cox"] = {
            "status": "insufficient_cases",
            "n": int(len(df)),
            "note": (
                "Multi-file cases are retained for descriptive "
                "sensitivity analysis only."
            ),
        }

        result["ph_check"] = {
            "status": "insufficient_cases",
            "n": int(len(df)),
        }

    else:
        result["adjusted_cox"] = fit_adjusted_cox(df)
        result["ph_check"] = proportional_hazards_check(df)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Step 09C TCGA-BRCA clinical/prognostic validation"
    )

    parser.add_argument("--scores", required=True)
    parser.add_argument("--survival", required=True)
    parser.add_argument("--clinical", required=True)
    parser.add_argument("--multifile-audit", required=True)
    parser.add_argument("--outdir", required=True)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scores = load_table(args.scores)
    survival = load_table(args.survival)
    clinical = load_table(args.clinical, sep="\t")
    multifile_audit = load_table(args.multifile_audit)

    require_columns(
        scores,
        [
            "case_id",
            SCORE_COLUMN,
        ],
        "09B scores",
    )

    # The production 09B artifact uses survival_time_days.
    # Normalize it internally to survival_time for the 09C analysis.
    if "survival_time" not in survival.columns:
        if "survival_time_days" in survival.columns:
            survival = survival.rename(
                columns={"survival_time_days": "survival_time"}
            )
        else:
            raise RuntimeError(
                "09B survival table must contain either "
                "'survival_time' or 'survival_time_days'. "
                f"Available columns: {list(survival.columns)}"
            )

    require_columns(
        survival,
        [
            "case_id",
            "survival_time",
            "survival_event",
        ],
        "09B survival table",
    )

    require_columns(
        clinical,
        [
            "case_id",
            "age_at_index",
            "ajcc_pathologic_stage",
        ],
        "clinical table",
    )

    # Keep one row per case in the score table.
    if scores["case_id"].duplicated().any():
        raise RuntimeError(
            "09B score table contains duplicate case_id values."
        )

    # Merge frozen scores with survival.
    merged = scores.merge(
        survival[
            [
                "case_id",
                "survival_time",
                "survival_event",
            ]
        ],
        on="case_id",
        how="left",
        validate="one_to_one",
    )

    # Merge clinical covariates separately.
    clinical_cols = [
        "case_id",
        "age_at_index",
        "ajcc_pathologic_stage",
        "vital_status",
        "primary_diagnosis",
        "classification_of_tumor",
        "ajcc_pathologic_t",
        "ajcc_pathologic_n",
        "ajcc_pathologic_m",
    ]

    clinical_cols = [
        c for c in clinical_cols
        if c in clinical.columns
    ]

    merged = merged.merge(
        clinical[clinical_cols],
        on="case_id",
        how="left",
        validate="one_to_one",
    )

    merged = clean_numeric(
        merged,
        [
            SCORE_COLUMN,
            "tcell_context_score",
            "cytotoxic_context_score",
            "exhaustion_context_score",
            "survival_time",
            "survival_event",
            "age_at_index",
        ],
    )

    merged = make_stage_group(merged)

    # Determine multi-file cases from the audited case-level file.
    require_columns(
        multifile_audit,
        ["case_id"],
        "multi-file audit",
    )

    multi_file_cases = set(
        multifile_audit["case_id"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    merged["file_class"] = np.where(
        merged["case_id"].isin(multi_file_cases),
        "multi_file",
        "single_file",
    )

    # Primary cohort.
    all_df = merged.copy()

    # Single-file sensitivity cohort.
    single_df = merged[
        merged["file_class"] == "single_file"
    ].copy()

    # Multi-file descriptive cohort.
    multi_df = merged[
        merged["file_class"] == "multi_file"
    ].copy()

    # Save analysis cohort.
    merged.to_csv(
        outdir / "GSE176078_09C_analysis_cohort.csv",
        index=False,
    )

    clinical_characterization = characterize_clinical(
        merged,
        outdir,
    )

    all_result = analyze_cohort(
        all_df,
        "all",
    )

    single_result = analyze_cohort(
        single_df,
        "single_file_sensitivity",
    )

    multi_result = analyze_cohort(
        multi_df,
        "multi_file_descriptive",
    )

    # Context correlations.
    context_correlations(
        all_df,
        "all",
        outdir,
    )

    context_correlations(
        single_df,
        "single_file_sensitivity",
        outdir,
    )

    context_correlations(
        multi_df,
        "multi_file_descriptive",
        outdir,
    )

    # Cox summary table.
    cox_rows = []

    for result in [
        all_result,
        single_result,
        multi_result,
    ]:
        uni = result["univariate_cox"]
        adj = result["adjusted_cox"]

        cox_rows.append(
            {
                "cohort": result["cohort"],
                "n_cases": result["n_cases"],
                "n_survival_time": result["n_survival_time"],
                "n_events": result["n_events"],

                "univariate_status": uni.get("status"),
                "univariate_hr": uni.get("hazard_ratio"),
                "univariate_p": uni.get("p_value"),
                "univariate_ci_lower": uni.get("ci_lower_hr"),
                "univariate_ci_upper": uni.get("ci_upper_hr"),
                "univariate_concordance": uni.get(
                    "concordance"
                ),

                "adjusted_status": adj.get("status"),
                "adjusted_hr": adj.get("hazard_ratio"),
                "adjusted_p": adj.get("p_value"),
                "adjusted_ci_lower": adj.get("ci_lower_hr"),
                "adjusted_ci_upper": adj.get("ci_upper_hr"),
                "adjusted_concordance": adj.get(
                    "concordance"
                ),

                "adjusted_n": adj.get("n"),
                "adjusted_events": adj.get("events"),
            }
        )

    cox_summary = pd.DataFrame(cox_rows)

    cox_summary.to_csv(
        outdir / "GSE176078_09C_TCGA_BRCA_cox_summary.csv",
        index=False,
    )

    # Stage distribution.
    stage_counts = (
        merged["ajcc_stage_group"]
        .value_counts(dropna=False)
        .rename_axis("stage_group")
        .reset_index(name="n")
    )

    stage_counts["stage_group"] = stage_counts[
        "stage_group"
    ].astype(str)

    stage_counts.to_csv(
        outdir / "GSE176078_09C_stage_group_counts.csv",
        index=False,
    )

    adjusted_complete = all_result["adjusted_cox"].get(
        "status"
    ) == "complete"

    report = {
        "cohort": "TCGA-BRCA",
        "step": "09C_clinical_prognostic_validation",
        "status": "complete",

        "score_definition": (
            "Frozen 09A molecular surrogate computed from "
            "TCGA bulk expression"
        ),

        "signature_refit_on_tcga": False,
        "outcome_optimized_cutoff": False,
        "original_gnn_reused_directly": False,
        "rna_velocity": False,

        "independent_biological_validation": False,
        "external_cohort_projection": True,

        "primary_cohort": "all",
        "sensitivity_cohort": "single_file_sensitivity",
        "multi_file_cohort": (
            "descriptive_only_due_to_small_n"
        ),

        "cohorts": {
            "all": all_result,
            "single_file_sensitivity": single_result,
            "multi_file_descriptive": multi_result,
        },

        "adjusted_model_specification": {
            "predictor": SCORE_COLUMN,
            "covariates": [
                "age_at_index",
                "ajcc_stage_group",
            ],
            "stage_group_definition": {
                "Stage 0/I": [
                    "Stage 0",
                    "Stage I",
                    "Stage IA",
                    "Stage IB",
                ],
                "Stage II": [
                    "Stage II",
                    "Stage IIA",
                    "Stage IIB",
                ],
                "Stage III": [
                    "Stage III",
                    "Stage IIIA",
                    "Stage IIIB",
                    "Stage IIIC",
                ],
                "Stage IV": [
                    "Stage IV",
                ],
            },
            "Stage_X_and_unrecognized": (
                "excluded from adjusted model as missing stage"
            ),
            "convergence_required": True,
            "convergence_achieved": adjusted_complete,
        },
