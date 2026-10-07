#!/usr/bin/env python3

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning


SCORE = "deep_evi_tcga_surrogate"
TCELL = "tcell_context_score"
CYTOTOXIC = "cytotoxic_context_score"
EXHAUSTION = "exhaustion_context_score"

STAGE_GROUPS = [
    "Stage 0/I",
    "Stage II",
    "Stage III",
    "Stage IV",
]


def load_table(path, sep=","):
    path = Path(path)

    if not path.exists():
        raise RuntimeError(f"Missing input: {path}")

    df = pd.read_csv(path, sep=sep, dtype={"case_id": str})
    df.columns = [str(c).strip() for c in df.columns]

    if "case_id" in df.columns:
        df["case_id"] = (
            df["case_id"]
            .astype(str)
            .str.strip()
        )

    return df


def require_columns(df, columns, name):
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"{name} missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def normalize_stage(value):

    if pd.isna(value):
        return np.nan

    x = str(value).strip().upper()
    x = " ".join(x.split())
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


def prepare(df):

    numeric = [
        SCORE,
        TCELL,
        CYTOTOXIC,
        EXHAUSTION,
        "survival_time",
        "survival_event",
        "age_at_index",
        "n_expression_files",
    ]

    for col in numeric:

        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    require_columns(
        df,
        ["ajcc_pathologic_stage"],
        "Clinical table",
    )

    df["ajcc_stage_group"] = (
        df["ajcc_pathologic_stage"]
        .apply(normalize_stage)
    )

    return df


def standardize_continuous(model, columns):

    model = model.copy()

    for col in columns:

        if col not in model.columns:
            continue

        values = pd.to_numeric(
            model[col],
            errors="coerce",
        )

        sd = values.std()

        if not np.isfinite(sd) or sd <= 0:
            continue

        mean = values.mean()

        model[col] = (values - mean) / sd

    return model


def make_adjusted_model_frame(df, extra_covariates):

    required = [
        "survival_time",
        "survival_event",
        SCORE,
        "age_at_index",
        "ajcc_stage_group",
    ] + extra_covariates

    work = df[required].copy()

    work = (
        work
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    stage_dummies = pd.get_dummies(
        work["ajcc_stage_group"],
        prefix="ajcc_stage_group",
        drop_first=True,
        dtype=float,
    )

    model = pd.concat(
        [
            work[
                [
                    "survival_time",
                    "survival_event",
                    SCORE,
                    "age_at_index",
                ] + extra_covariates
            ],
            stage_dummies,
        ],
        axis=1,
    )

    for col in list(model.columns):

        if col in {
            "survival_time",
            "survival_event",
        }:
            continue

        if model[col].nunique(dropna=True) <= 1:
            model = model.drop(columns=col)

    continuous = [
        SCORE,
        "age_at_index",
    ] + extra_covariates

    model = standardize_continuous(
        model,
        continuous,
    )

    return model


def fit_adjusted_model(
    df,
    extra_covariates,
    model_name,
    penalizer=0.0,
):

    model = make_adjusted_model_frame(
        df,
        extra_covariates,
    )

    if len(model) < 50:

        return {
            "model": model_name,
            "status": "insufficient_cases",
            "n": int(len(model)),
            "events": int(
                model["survival_event"].sum()
            ) if len(model) else 0,
        }

    events = int(
        model["survival_event"].sum()
    )

    if events < 10:

        return {
            "model": model_name,
            "status": "insufficient_events",
            "n": int(len(model)),
            "events": events,
        }

    cph = CoxPHFitter(
        penalizer=penalizer,
        l1_ratio=0.0,
    )

    try:

        with warnings.catch_warnings():

            warnings.simplefilter(
                "error",
                ConvergenceWarning,
            )

            warnings.simplefilter(
                "error",
                RuntimeWarning,
            )

            cph.fit(
                model,
                duration_col="survival_time",
                event_col="survival_event",
            )

    except (
        ConvergenceWarning,
        RuntimeWarning,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as exc:

        return {
            "model": model_name,
            "status": "convergence_or_numeric_failure",
            "n": int(len(model)),
            "events": events,
            "penalizer": float(penalizer),
            "note": str(exc),
        }

    except Exception as exc:

        return {
            "model": model_name,
            "status": "fit_failed",
            "n": int(len(model)),
            "events": events,
            "penalizer": float(penalizer),
            "note": str(exc),
        }

    if SCORE not in cph.summary.index:

        return {
            "model": model_name,
            "status": "score_missing_from_model",
            "n": int(len(model)),
            "events": events,
            "penalizer": float(penalizer),
        }

    row = cph.summary.loc[SCORE]

    values = {
        "deep_evi_coef": row["coef"],
        "deep_evi_hr": np.exp(row["coef"]),
        "deep_evi_p": row["p"],
        "deep_evi_ci_lower": np.exp(
            row["coef lower 95%"]
        ),
        "deep_evi_ci_upper": np.exp(
            row["coef upper 95%"]
        ),
        "concordance": cph.concordance_index_,
    }

    if not all(
        np.isfinite(float(v))
        for v in values.values()
    ):

        return {
            "model": model_name,
            "status": "nonfinite_result",
            "n": int(len(model)),
            "events": events,
            "penalizer": float(penalizer),
            "note": (
                "Cox model returned a non-finite "
                "estimate and was not accepted."
            ),
        }

    return {
        "model": model_name,
        "status": "complete",
        "n": int(len(model)),
        "events": events,
        "penalizer": float(penalizer),
        "standardized_continuous_predictors": [
            SCORE,
            "age_at_index",
        ] + extra_covariates,
        "deep_evi_coef": float(values["deep_evi_coef"]),
        "deep_evi_hr": float(values["deep_evi_hr"]),
        "deep_evi_p": float(values["deep_evi_p"]),
        "deep_evi_ci_lower": float(
            values["deep_evi_ci_lower"]
        ),
        "deep_evi_ci_upper": float(
            values["deep_evi_ci_upper"]
        ),
        "concordance": float(
            values["concordance"]
        ),
        "covariates": [
            c
            for c in model.columns
            if c not in {
                "survival_time",
                "survival_event",
            }
        ],
    }


def fit_unadjusted(df, cohort_name):

    work = df[
        [
            "survival_time",
            "survival_event",
            SCORE,
        ]
    ].copy()

    work = (
        work
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(work) < 50:

        return {
            "cohort": cohort_name,
            "model": "M1_unadjusted",
            "status": "insufficient_cases",
            "n": int(len(work)),
        }

    events = int(
        work["survival_event"].sum()
    )

    if events < 10:

        return {
            "cohort": cohort_name,
            "model": "M1_unadjusted",
            "status": "insufficient_events",
            "n": int(len(work)),
            "events": events,
        }

    work = standardize_continuous(
        work,
        [SCORE],
    )

    cph = CoxPHFitter()

    try:

        with warnings.catch_warnings():

            warnings.simplefilter(
                "error",
                ConvergenceWarning,
            )

            warnings.simplefilter(
                "error",
                RuntimeWarning,
            )

            cph.fit(
                work,
                duration_col="survival_time",
                event_col="survival_event",
            )

    except Exception as exc:

        return {
            "cohort": cohort_name,
            "model": "M1_unadjusted",
            "status": "fit_failed",
            "n": int(len(work)),
            "events": events,
            "note": str(exc),
        }

    row = cph.summary.loc[SCORE]

    values = [
        row["coef"],
        np.exp(row["coef"]),
        row["p"],
        np.exp(row["coef lower 95%"]),
        np.exp(row["coef upper 95%"]),
        cph.concordance_index_,
    ]

    if not all(
        np.isfinite(float(v))
        for v in values
    ):

        return {
            "cohort": cohort_name,
            "model": "M1_unadjusted",
            "status": "nonfinite_result",
            "n": int(len(work)),
            "events": events,
        }

    return {
        "cohort": cohort_name,
        "model": "M1_unadjusted",
        "status": "complete",
        "n": int(len(work)),
        "events": events,
        "deep_evi_coef": float(values[0]),
        "deep_evi_hr": float(values[1]),
        "deep_evi_p": float(values[2]),
        "deep_evi_ci_lower": float(values[3]),
        "deep_evi_ci_upper": float(values[4]),
        "concordance": float(values[5]),
        "standardized_predictor": True,
        "interpretation": (
            "HR corresponds to a one-standard-deviation "
            "increase in the frozen TCGA molecular surrogate."
        ),
    }


def context_correlations(df, cohort_name):

    rows = []

    for context in [
        TCELL,
        CYTOTOXIC,
        EXHAUSTION,
    ]:

        work = (
            df[[SCORE, context]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(work) < 10:
            continue

        rho, p = spearmanr(
            work[SCORE],
            work[context],
        )

        rows.append(
            {
                "cohort": cohort_name,
                "context_score": context,
                "n": int(len(work)),
                "spearman_rho": float(rho),
                "spearman_p": float(p),
            }
        )

    return rows


def classify_expression_files(scores):

    require_columns(
        scores,
        [
            "case_id",
            "n_expression_files",
        ],
        "TCGA score table",
    )

    out = scores.copy()

    out["n_expression_files"] = pd.to_numeric(
        out["n_expression_files"],
        errors="coerce",
    )

    if out["n_expression_files"].isna().any():

        bad = out.loc[
            out["n_expression_files"].isna(),
            "case_id",
        ].tolist()

        raise RuntimeError(
            "Missing/non-numeric n_expression_files "
            f"for cases: {bad[:10]}"
        )

    if (out["n_expression_files"] < 1).any():

        bad = out.loc[
            out["n_expression_files"] < 1,
            ["case_id", "n_expression_files"],
        ]

        raise RuntimeError(
            "Invalid n_expression_files values:\n"
            f"{bad.to_string(index=False)}"
        )

    out["file_class"] = np.where(
        out["n_expression_files"] == 1,
        "single_file",
        "multi_file",
    )

    return out


def finite_json(obj):

    if isinstance(obj, dict):

        return {
            str(k): finite_json(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):

        return [
            finite_json(v)
            for v in obj
        ]

    if isinstance(obj, tuple):

        return [
            finite_json(v)
            for v in obj
        ]

    if isinstance(
        obj,
        (np.floating, float),
    ):

        value = float(obj)

        if not np.isfinite(value):
            return None

        return value

    if isinstance(
        obj,
        (np.integer, int),
    ):

        return int(obj)

    if isinstance(
        obj,
        (np.bool_, bool),
    ):

        return bool(obj)

    return obj


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scores",
        required=True,
    )

    parser.add_argument(
        "--survival",
        required=True,
    )

    parser.add_argument(
        "--clinical",
        required=True,
    )

    parser.add_argument(
        "--outdir",
        required=True,
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores = load_table(args.scores)
    survival = load_table(args.survival)
    clinical = load_table(
        args.clinical,
        sep="\t",
    )

    require_columns(
        scores,
        [
            "case_id",
            SCORE,
            TCELL,
            CYTOTOXIC,
            EXHAUSTION,
            "n_expression_files",
        ],
        "TCGA score table",
    )

    require_columns(
        survival,
        [
            "case_id",
            "survival_time_days",
            "survival_event",
        ],
        "TCGA survival table",
    )

    require_columns(
        clinical,
        [
            "case_id",
            "age_at_index",
            "ajcc_pathologic_stage",
        ],
        "TCGA clinical table",
    )

    survival = survival.copy()

    survival["survival_time"] = pd.to_numeric(
        survival["survival_time_days"],
        errors="coerce",
    )

    survival["survival_event"] = pd.to_numeric(
        survival["survival_event"],
        errors="coerce",
    )

    scores = classify_expression_files(scores)

    # Ensure one score row per case.
    if scores["case_id"].duplicated().any():

        dup = scores.loc[
            scores["case_id"].duplicated(keep=False),
            "case_id",
        ].unique()

        raise RuntimeError(
            "Duplicate case_id values in score table: "
            f"{dup[:10]}"
        )

    merged = (
        scores
        .merge(
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
        .merge(
            clinical,
            on="case_id",
            how="left",
            suffixes=("", "_clinical"),
            validate="one_to_one",
        )
    )

    merged = prepare(merged)

    # ============================================================
    # DEFINITIVE COHORT CONSTRUCTION
    # ============================================================

    all_df = merged.copy()

    single_file_df = merged.loc[
        merged["file_class"] == "single_file"
    ].copy()

    multi_file_df = merged.loc[
        merged["file_class"] == "multi_file"
    ].copy()

    # Analysis cohort requires valid survival time/event.
    def analysis_ready(df):

        return (
            df
            .replace([np.inf, -np.inf], np.nan)
            .dropna(
                subset=[
                    SCORE,
                    "survival_time",
                    "survival_event",
                ]
            )
            .copy()
        )

    all_analysis = analysis_ready(all_df)
    single_analysis = analysis_ready(single_file_df)
    multi_analysis = analysis_ready(multi_file_df)

    # ============================================================
    # MODEL RESULTS
    # ============================================================

    results = []

    # M1
    results.append(
        fit_unadjusted(
            all_analysis,
            "all",
        )
    )

    # M2
    m2 = fit_adjusted_model(
        all_analysis,
        [],
        "M2_age_stage",
        penalizer=0.0,
    )
    m2["cohort"] = "all"
    results.append(m2)

    # M3
    m3 = fit_adjusted_model(
        all_analysis,
        [TCELL],
        "M3_age_stage_tcell",
        penalizer=0.01,
    )
    m3["cohort"] = "all"
    results.append(m3)

    # M4
    m4 = fit_adjusted_model(
        all_analysis,
        [CYTOTOXIC],
        "M4_age_stage_cytotoxic",
        penalizer=0.01,
    )
    m4["cohort"] = "all"
    results.append(m4)

    # Single-file sensitivity
    m1s = fit_unadjusted(
        single_analysis,
        "single_file_sensitivity",
    )
    results.append(m1s)

    m2 = fit_adjusted_model(
        single_analysis,
        [],
        "M2_age_stage",
        penalizer=0.0,
    )
    m2["cohort"] = "single_file_sensitivity"
    results.append(m2)

    m3 = fit_adjusted_model(
        single_analysis,
        [TCELL],
        "M3_age_stage_tcell",
        penalizer=0.01,
    )
    m3["cohort"] = "single_file_sensitivity"
    results.append(m3)

    m4 = fit_adjusted_model(
        single_analysis,
        [CYTOTOXIC],
        "M4_age_stage_cytotoxic",
        penalizer=0.01,
    )
    m4["cohort"] = "single_file_sensitivity"
    results.append(m4)

    # ============================================================
    # CONTEXT CORRELATIONS
    # ============================================================

    correlation_rows = []

    correlation_rows.extend(
        context_correlations(
            all_analysis,
            "all",
        )
    )

    correlation_rows.extend(
        context_correlations(
            single_analysis,
            "single_file_sensitivity",
        )
    )

    # ============================================================
    # MULTI-FILE DESCRIPTIVE ANALYSIS
    # ============================================================

    multi_corr = context_correlations(
        multi_analysis,
        "multi_file_descriptive",
    )

    # ============================================================
    # ANALYSIS COHORT TABLE
    # ============================================================

    analysis_export = merged[
        [
            "case_id",
            SCORE,
            TCELL,
            CYTOTOXIC,
            EXHAUSTION,
            "n_expression_files",
            "file_class",
            "survival_time",
            "survival_event",
            "age_at_index",
            "ajcc_pathologic_stage",
            "ajcc_stage_group",
        ]
    ].copy()

    analysis_export.to_csv(
        outdir / "09D_analysis_cohort.csv",
        index=False,
    )

    pd.DataFrame(results).to_csv(
        outdir / "09D_bulk_composition_cox.csv",
        index=False,
    )

    pd.DataFrame(correlation_rows).to_csv(
        outdir / "09D_context_correlations.csv",
        index=False,
    )

    pd.DataFrame(multi_corr).to_csv(
        outdir
        / "09D_multi_file_descriptive_context_correlations.csv",
        index=False,
    )

    # ============================================================
    # COHORT AUDIT
    # ============================================================

    cohort_audit = {
        "all_score_cases": int(
            all_df["case_id"].nunique()
        ),
        "all_survival_cases": int(
            len(all_analysis)
        ),
        "single_file_score_cases": int(
            single_file_df["case_id"].nunique()
        ),
        "single_file_survival_cases": int(
            len(single_analysis)
        ),
        "multi_file_score_cases": int(
            multi_file_df["case_id"].nunique()
        ),
        "multi_file_survival_cases": int(
            len(multi_analysis)
        ),
        "single_file_expected_survival_cases": 1083,
        "multi_file_expected_cases": 11,
    }

    # Hard reproducibility guard.
    if (
        cohort_audit["single_file_survival_cases"]
        != 1083
    ):

        raise RuntimeError(
            "09D single-file cohort audit failed: "
            f"expected 1083 survival-available cases, "
            f"observed "
            f"{cohort_audit['single_file_survival_cases']}. "
            "The analysis was not accepted."
        )

    if (
        cohort_audit["multi_file_score_cases"]
        != 11
    ):

        raise RuntimeError(
            "09D multi-file cohort audit failed: "
            f"expected 11 cases, observed "
            f"{cohort_audit['multi_file_score_cases']}."
        )

    # ============================================================
    # REPORT
    # ============================================================

    report = {