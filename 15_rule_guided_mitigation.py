#!/usr/bin/env python3
"""
15_rule_guided_mitigation.py

Rule-guided mitigation on baseline (no-injection) outputs.

Inputs (from this repo):
- Baseline dataset (from 4_get_output_metrics.py):
    abstract_model_io/topic_{MODEL}_{DIFFICULTY}.csv
- RuleSHAP rules (from 6_extract_rules.py):
    xai_analyses_results/rules/shap_in_xgb={BOOL}+shap_in_lasso={BOOL}/
      association_rules_{MODEL}_{DIFFICULTY}_{METRIC}.csv

This script:
1) Selects a trigger rule from the rules CSV (default: highest-importance RULE row).
2) Finds all topics where the rule fires.
3) Applies a deterministic system-instruction patch to steer a chosen metric in a chosen direction.
4) Re-generates only those triggered outputs and recomputes metrics.
5) Saves per-topic deltas + a JSON summary with paired stats.

Supported target metrics:
- subjectivity_score_nn
- sentiment_score_nn
- gunning_fog
- explanation_length

Sign normalization:
- none:   use rule as-is
- absorb: keep trigger subset, but normalize sign reporting (treat as "positive")
- flip_mask: if coefficient_sign is negative, normalize it to positive by logically negating the firing set
             (i.e., patch topics where the original rule does NOT fire). This yields an exact complement set
             without needing OR syntax.

Direction/sign alignment:
After sign normalization, if the rule is expected to move the target metric in the SAME direction as the
requested intervention (e.g., rule is positive and --direction increase), we flip the firing set again.
This makes the trigger select the subset where the model is expected to move AGAINST the desired direction,
so the patch acts as a targeted correction.

Examples:
  python 15_rule_guided_mitigation.py \
    --model llama3.1 --difficulty baseline \
    --metric subjectivity_score_nn --direction decrease \
    --use_shap_in_xgb --use_shap_in_lasso \
    --sign_normalization absorb
"""

import os
import re
import glob
import json
import argparse
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from textstat import gunning_fog

from lib import instruct_model

# transformers is required only for NN metrics
try:
    from transformers import pipeline
except Exception:
    pipeline = None


###############################################################################
# Rule parsing / evaluation
###############################################################################

_COND_RE = re.compile(r'^\s*(?P<feat>.*?)\s*(?P<op><=|>=|<|>|=)\s*(?P<val>-?\d+(?:\.\d+)?)\s*$')


def parse_rule_expression(rule_expr: str) -> List[Tuple[str, str, float]]:
    """Parse a conjunction like: 'common > 0.8 & negative <= 0.6'."""
    if not isinstance(rule_expr, str) or not rule_expr.strip():
        return []
    parts = [p.strip() for p in rule_expr.split('&')]
    conds: List[Tuple[str, str, float]] = []
    for p in parts:
        m = _COND_RE.match(p)
        if not m:
            raise ValueError(f"Cannot parse condition: {p!r} (full rule: {rule_expr!r})")
        conds.append((m.group('feat').strip(), m.group('op'), float(m.group('val'))))
    return conds


def eval_conditions(df: pd.DataFrame, conds: List[Tuple[str, str, float]]) -> pd.Series:
    """Evaluate parsed conditions on df columns."""
    mask = pd.Series(True, index=df.index)
    for feat, op, val in conds:
        if feat not in df.columns:
            raise KeyError(
                f"Feature column not found in baseline dataset: {feat!r}. "
                f"Available cols (prefix): {list(df.columns)[:30]}..."
            )
        s = df[feat]
        if op == '<=':
            mask &= (s <= val)
        elif op == '>=':
            mask &= (s >= val)
        elif op == '<':
            mask &= (s < val)
        elif op == '>':
            mask &= (s > val)
        elif op == '=':
            mask &= (s == val)
        else:
            raise ValueError(f"Unsupported operator: {op}")
    return mask


def invert_operator(op: str) -> str:
    if op == '>':
        return '<='
    if op == '>=':
        return '<'
    if op == '<':
        return '>='
    if op == '<=':
        return '>'
    if op == '=':
        # would require '!=' for true negation; keep '=' to avoid invalid syntax
        return '='
    raise ValueError(f"Unsupported operator: {op}")


def flip_conjunction(rule_expr: str) -> str:
    """Flip each inequality while keeping conjunction-only syntax (not a logical negation)."""
    conds = parse_rule_expression(rule_expr)
    flipped = [(feat, invert_operator(op), val) for feat, op, val in conds]
    return " & ".join([f"{feat} {op} {val:g}" for feat, op, val in flipped])


def _expected_effect_from_sign(sign: str) -> str:
    s = (sign or "").strip().lower()
    if s == "positive":
        return "increase"
    if s == "negative":
        return "decrease"
    return "unknown"


###############################################################################
# Stats helpers
###############################################################################

def cohen_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples (d_z)."""
    d = y - x
    if d.size < 2:
        return float('nan')
    sd = np.nanstd(d, ddof=1)
    return float(np.nanmean(d) / sd) if sd and sd > 0 else float('nan')


def paired_ttest_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided paired t-test p-value (scipy if available)."""
    try:
        from scipy.stats import ttest_rel
        return float(ttest_rel(y, x, nan_policy="omit").pvalue)
    except Exception:
        return float('nan')


###############################################################################
# I/O helpers
###############################################################################

def _find_first(patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def infer_rules_csv(model: str, metric: str, difficulty: str, use_shap_in_xgb: bool, use_shap_in_lasso: bool) -> str:
    rules_dir = os.path.join(
        "xai_analyses_results",
        "rules",
        f"shap_in_xgb={use_shap_in_xgb}+shap_in_lasso={use_shap_in_lasso}",
    )
    return os.path.join(rules_dir, f"association_rules_{model}_{difficulty}_{metric}.csv")


def load_baseline_dataset(model: str, difficulty: str, baseline_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Preferred: abstract_model_io/topic_{model}_{difficulty}.csv
    Fallback: build from topic_scores + topic_explanations_{difficulty}_...
    """
    csv_file_dir = "abstract_model_io"
    if baseline_csv is None:
        baseline_csv = os.path.join(csv_file_dir, f"topic_{model}_{difficulty}.csv")
    if os.path.isfile(baseline_csv):
        return pd.read_csv(baseline_csv)

    # Fallback: auto-detect filenames
    scores_csv = _find_first([
        os.path.join(csv_file_dir, f"topic_scores_model-{model}_temperature-0_top_p-0.csv"),
        os.path.join(csv_file_dir, f"topic_scores_*model-{model}_*.csv"),
    ])
    expl_csv = _find_first([
        os.path.join(csv_file_dir, f"topic_explanations_{difficulty}_model-{model}_temperature-0_top_p-0.csv"),
        os.path.join(csv_file_dir, f"topic_explanations_{difficulty}_*model-{model}_*.csv"),
    ])
    if not scores_csv or not expl_csv:
        raise FileNotFoundError(
            "Cannot find baseline dataset. Expected either:\n"
            f"  {baseline_csv}\n"
            "or the pair:\n"
            f"  abstract_model_io/topic_scores_*model-{model}_*.csv\n"
            f"  abstract_model_io/topic_explanations_{difficulty}_*model-{model}_*.csv\n"
        )

    scores_df = pd.read_csv(scores_csv).drop_duplicates(subset=["topic", "score_type"])
    expl_df = pd.read_csv(expl_csv)
    pivot = scores_df.pivot(index="topic", columns="score_type", values="score_value").reset_index()
    df = pd.merge(pivot, expl_df, on="topic", how="inner")

    # Minimal metrics if missing
    df["gunning_fog"] = df["explanation"].astype(str).apply(gunning_fog)
    df["explanation_length"] = df["explanation"].astype(str).apply(len)
    return df


###############################################################################
# Metric calculators (match 4_get_output_metrics.py)
###############################################################################

_SENTIMENT_PIPE = None
_SUBJECTIVITY_PIPE = None


def _ensure_sentiment_pipeline():
    global _SENTIMENT_PIPE
    if _SENTIMENT_PIPE is None:
        if pipeline is None:
            raise ImportError("transformers is required to compute sentiment_score_nn.")
        _SENTIMENT_PIPE = pipeline(
            task="sentiment-analysis",
            model="tabularisai/multilingual-sentiment-analysis",
        )
    return _SENTIMENT_PIPE


def _ensure_subjectivity_pipeline():
    global _SUBJECTIVITY_PIPE
    if _SUBJECTIVITY_PIPE is None:
        if pipeline is None:
            raise ImportError("transformers is required to compute subjectivity_score_nn.")
        _SUBJECTIVITY_PIPE = pipeline(
            task="text-classification",
            model="GroNLP/mdebertav3-subjectivity-multilingual",
        )
    return _SUBJECTIVITY_PIPE


def calculate_sentiment_nn(text: str, max_tokens: int = 400, avg_chars_per_token: float = 3.5) -> float:
    sentiment_analyzer = _ensure_sentiment_pipeline()
    max_characters = int(max_tokens * avg_chars_per_token)
    chunks = [text[i:i + max_characters] for i in range(0, len(text), max_characters)]
    results = []
    for chunk in chunks:
        try:
            results.append(sentiment_analyzer(chunk)[0])
        except Exception:
            pass
    sentiment_map = {0: "negative", 1: "negative", 2: "neutral", 3: "positive", 4: "positive"}
    sentiment_dict = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for result in results:
        sentiment_type = str(result.get("label", "")).lower()
        if "negative" in sentiment_type:
            sentiment_type = "negative"
        elif "positive" in sentiment_type:
            sentiment_type = "positive"
        if sentiment_type not in sentiment_dict:
            try:
                sentiment_type_id = int(str(result.get("label", "")).split('_')[-1])
                sentiment_type = sentiment_map.get(sentiment_type_id, "neutral")
            except Exception:
                sentiment_type = "neutral"
        sentiment_dict[sentiment_type] = max(sentiment_dict[sentiment_type], float(result.get("score", 0.0)))
    label, score = max(sentiment_dict.items(), key=lambda x: x[-1])
    if label == "neutral":
        return 0.0
    return -float(score) if label == "negative" else float(score)


def calculate_subjectivity_nn(text: str, max_tokens: int = 400, avg_chars_per_token: float = 3.5) -> float:
    subjectivity_classifier = _ensure_subjectivity_pipeline()
    max_characters = int(max_tokens * avg_chars_per_token)
    chunks = [text[i:i + max_characters] for i in range(0, len(text), max_characters)]
    results = []
    for chunk in chunks:
        try:
            results.append(subjectivity_classifier(chunk)[0])
        except Exception:
            pass
    subj_map = {0: "objective", 1: "subjective"}
    subj_dict = {"objective": 0.0, "subjective": 0.0}
    for result in results:
        try:
            subj_type_id = int(str(result.get("label", "")).split('_')[-1])
            subj_type = subj_map[subj_type_id]
        except Exception:
            subj_type = "objective"
        subj_dict[subj_type] = max(subj_dict[subj_type], float(result.get("score", 0.0)))
    return float(subj_dict["subjective"])


def compute_metric(text: str, metric: str) -> float:
    if metric == "explanation_length":
        return float(len(text))
    if metric == "gunning_fog":
        return float(gunning_fog(text))
    if metric == "sentiment_score_nn":
        return float(calculate_sentiment_nn(text))
    if metric == "subjectivity_score_nn":
        return float(calculate_subjectivity_nn(text))
    raise ValueError(f"Unsupported metric: {metric}")


###############################################################################
# Patch instruction templates (two directions per metric)
###############################################################################

_PATCH_TEMPLATES: Dict[str, Dict[str, str]] = {
    "subjectivity_score_nn": {
        "increase": (
            "Write an explanation that includes interpretation and evaluative language where appropriate. "
            "Acknowledge uncertainties, present plausible perspectives, and make reasoned judgments. "
            "Use mild opinionated phrasing (e.g., 'arguably', 'it is reasonable to view...'), "
            "but remain grounded in facts."
        ),
        "decrease": (
            "Write an explanation in a strictly objective and neutral tone. "
            "Avoid opinions, value judgments, and emotive or persuasive language. "
            "Focus on verifiable facts, definitions, and causal mechanisms. "
            "Use impersonal phrasing and avoid taking sides."
        ),
    },
    "sentiment_score_nn": {
        "increase": (
            "Write an explanation with an overall positive and constructive framing. "
            "Highlight benefits, opportunities, and solutions when relevant, while staying truthful."
        ),
        "decrease": (
            "Write an explanation with an overall critical and cautionary framing. "
            "Emphasize risks, downsides, and failure modes when relevant, while staying truthful."
        ),
    },
    "gunning_fog": {
        "increase": (
            "Write a detailed, nuanced explanation using precise terminology and moderately complex sentence structures. "
            "Include caveats, trade-offs, and multiple facets. Avoid oversimplification."
        ),
        "decrease": (
            "Write a simple, easy-to-read explanation. Use short sentences and common words. "
            "Avoid jargon; if technical terms are necessary, define them plainly."
        ),
    },
    "explanation_length": {
        "increase": (
            "Write a longer explanation: at least three substantive paragraphs. "
            "Add context, examples, and clarifying details without repeating yourself."
        ),
        "decrease": (
            "Write a concise explanation in 4-6 sentences. "
            "Focus on the core idea; omit tangential details and avoid long lists."
        ),
    },
}


def normalize_direction(metric: str, direction: str) -> str:
    d = (direction or "").strip().lower()
    if d in {"increase", "up", "+", "higher", "more"}:
        return "increase"
    if d in {"decrease", "down", "-", "lower", "less"}:
        return "decrease"

    if metric == "sentiment_score_nn":
        if d in {"positive", "pos", "more_positive"}:
            return "increase"
        if d in {"negative", "neg", "more_negative"}:
            return "decrease"

    if metric == "subjectivity_score_nn":
        if d in {"subjective", "more_subjective"}:
            return "increase"
        if d in {"objective", "more_objective"}:
            return "decrease"

    if metric == "gunning_fog":
        if d in {"complex", "more_complex"}:
            return "increase"
        if d in {"simple", "simpler"}:
            return "decrease"

    if metric == "explanation_length":
        if d in {"long", "longer"}:
            return "increase"
        if d in {"short", "shorter"}:
            return "decrease"

    raise ValueError(f"Unrecognized direction {direction!r}.")


def build_patch_instruction(metric: str, direction: str, extra: Optional[str] = None, override_file: Optional[str] = None) -> str:
    if override_file:
        with open(override_file, "r", encoding="utf-8") as f:
            base = f.read().strip()
    else:
        direction = normalize_direction(metric, direction)
        base = _PATCH_TEMPLATES[metric][direction]
    if extra:
        extra = extra.strip()
        if extra:
            return base + " " + extra
    return base


###############################################################################
# Rules schema helpers
###############################################################################

def _norm_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def pick_rule_expr_col(df: pd.DataFrame) -> str:
    for c in ["rule_expression", "rule", "expression", "component", "pattern"]:
        if c in df.columns:
            return c
    # heuristic: column with most comparator-looking strings
    best_col, best_score = None, -1.0
    for c in df.columns:
        s = df[c].astype(str)
        score = float(s.str.contains(r"(<=|>=|<|>)", regex=True, na=False).mean())
        score += 0.25 * float(s.str.contains(r"&", regex=True, na=False).mean())
        if score > best_score:
            best_col, best_score = c, score
    if best_col is None:
        raise ValueError(f"Could not infer rule-expression column. Columns: {list(df.columns)}")
    return best_col


def pick_coeff_sign_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["coefficient_sign", "coef_sign", "sign", "impact_direction"]:
        if c in df.columns:
            return c
    return None


def filter_rule_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows where type != 'rule' (supports 'type' or 'component_type'), if the column exists."""
    for type_col in ["type", "component_type"]:
        if type_col in df.columns:
            s = _norm_str_series(df[type_col])
            if (s == "rule").any():
                return df.loc[s == "rule"].copy()
    return df.copy()


###############################################################################
# Main
###############################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-guided mitigation on baseline outputs.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--difficulty",
        type=str,
        default="baseline",
        choices=["baseline", "easy", "medium", "hard"],
        help="Dataset split (default: baseline).",
    )

    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=["subjectivity_score_nn", "sentiment_score_nn", "gunning_fog", "explanation_length"],
        help="Target metric to steer.",
    )
    parser.add_argument("--direction", type=str, required=True)

    parser.add_argument("--use_shap_in_xgb", action="store_true", help="Match rules folder flag.")
    parser.add_argument("--use_shap_in_lasso", action="store_true", help="Match rules folder flag.")

    parser.add_argument("--baseline_csv", type=str, default=None)
    parser.add_argument("--rules_csv", type=str, default=None)

    parser.add_argument("--rule_index", type=int, default=None, help="0-based index over RULE rows (after filtering).")
    parser.add_argument("--rule_expression", type=str, default=None, help="Exact rule expression to use.")
    parser.add_argument("--rule_contains", type=str, default=None, help="Substring filter over rule expressions.")
    parser.add_argument(
        "--sort_by",
        choices=["auto", "importance_weighted_by_gain", "importance", "dataset_coverage", "none"],
        default="auto",
    )
    parser.add_argument(
        "--prefer_sign",
        choices=["any", "positive", "negative"],
        default="any",
        help="Optionally restrict candidates by coefficient_sign (default: any).",
    )
    parser.add_argument(
        "--sign_normalization",
        # Backward compatible: "flip_conjunction" now behaves like "flip_mask".
        choices=["none", "absorb", "flip_mask", "flip_conjunction"],
        default="absorb",
    )

    parser.add_argument(
        "--max_patched_topics",
        "--max_patches",
        dest="max_patched_topics",
        type=int,
        default=300,
        help="Maximum number of patched topics (default: 200). Use 0 to patch all selected topics.",
    )
    parser.add_argument(
        "--selection",
        choices=["random", "highest_common", "lowest_fog"],
        default="random",
        help="If capping, how to pick topics.",
    )
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--patch_instruction_file", type=str, default=None)
    parser.add_argument("--patch_extra", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    # set_deterministic(args.random_seed)
    np.random.seed(args.random_seed)

    direction = normalize_direction(args.metric, args.direction)

    # Load baseline
    df = load_baseline_dataset(args.model, args.difficulty, baseline_csv=args.baseline_csv)
    if "topic" not in df.columns or "explanation" not in df.columns:
        raise KeyError("Baseline CSV must contain columns: 'topic' and 'explanation'.")

    # Rules CSV
    rules_path = args.rules_csv or infer_rules_csv(
        args.model, args.metric, args.difficulty, args.use_shap_in_xgb, args.use_shap_in_lasso
    )
    if not os.path.isfile(rules_path):
        raise FileNotFoundError(
            f"Rules CSV not found: {rules_path}\n"
            "Either run 6_extract_rules.py for this (model,difficulty,metric,flags) "
            "or pass --rules_csv explicitly."
        )
    rules_df = pd.read_csv(rules_path)

    # Filter to RULE rows
    cand = filter_rule_rows(rules_df)
    rule_expr_col = pick_rule_expr_col(cand)
    coeff_sign_col = pick_coeff_sign_col(cand)

    # Optional rule substring filter
    if args.rule_contains:
        cand = cand[_norm_str_series(cand[rule_expr_col]).str.contains(args.rule_contains.strip().lower(), na=False)].copy()

    # Optional sign restriction
    if args.prefer_sign != "any" and coeff_sign_col and coeff_sign_col in cand.columns:
        sub = cand[_norm_str_series(cand[coeff_sign_col]) == args.prefer_sign].copy()
        if len(sub) > 0:
            cand = sub

    if len(cand) == 0:
        raise ValueError("No candidate RULE rows left after filtering. Check --rule_contains/--prefer_sign.")

    # Select trigger row
    if args.rule_expression:
        m = cand[rule_expr_col].astype(str) == args.rule_expression
        if not m.any():
            raise ValueError(f"Provided --rule_expression not found (column {rule_expr_col!r}) after filtering.")
        trigger_row = cand[m].iloc[0]
    elif args.rule_index is not None:
        if args.rule_index < 0 or args.rule_index >= len(cand):
            raise IndexError(f"--rule_index out of bounds. Filtered rule rows have {len(cand)} rows.")
        trigger_row = cand.iloc[int(args.rule_index)]
    else:
        # Sort before taking the top
        if args.sort_by != "none":
            if args.sort_by == "auto":
                for col in ["importance_weighted_by_gain", "importance", "dataset_coverage"]:
                    if col in cand.columns:
                        cand = cand.sort_values(by=col, ascending=False)
                        break
            else:
                if args.sort_by in cand.columns:
                    cand = cand.sort_values(by=args.sort_by, ascending=False)
        trigger_row = cand.iloc[0]

    rule_raw = str(trigger_row.get(rule_expr_col, "")).strip()
    if not rule_raw:
        raise ValueError(f"Selected trigger rule expression is empty (column {rule_expr_col!r}).")

    coeff_sign_raw = ""
    if coeff_sign_col and coeff_sign_col in trigger_row.index:
        coeff_sign_raw = str(trigger_row.get(coeff_sign_col, "")).strip().lower()

    # Evaluate the original trigger first
    mask_fire_raw = eval_conditions(df, parse_rule_expression(rule_raw))

    # Normalize sign / determine which subset is considered "firing"
    # Backward compatible: "flip_conjunction" behaves like "flip_mask".
    norm_mode = "flip_mask" if args.sign_normalization == "flip_conjunction" else args.sign_normalization

    rule_used = rule_raw
    mask_fire = mask_fire_raw
    flipped_for_sign_normalization = False

    coeff_sign_norm = coeff_sign_raw or "unknown"
    expected_effect_raw = _expected_effect_from_sign(coeff_sign_raw)

    if norm_mode == "absorb" and coeff_sign_raw in {"positive", "negative"}:
        # Keep the same subset, but report the rule as "positive".
        coeff_sign_norm = "positive"
    elif norm_mode == "flip_mask" and coeff_sign_raw == "negative":
        # Exact complement of a conjunction without needing OR syntax.
        mask_fire = ~mask_fire_raw
        rule_used = f"NOT({rule_raw})"
        coeff_sign_norm = "positive"
        flipped_for_sign_normalization = True

    expected_effect_norm = _expected_effect_from_sign(coeff_sign_norm)

    # For direction alignment we must use the sign that corresponds to the CURRENT firing set.
    # - If we only "absorb" the sign, the subset is unchanged -> align using the raw sign.
    # - If we flip_mask (negate the firing set), the subset's effect sign flips -> align using "positive".
    coeff_sign_for_alignment = coeff_sign_raw or "unknown"
    if flipped_for_sign_normalization:
        coeff_sign_for_alignment = "positive"
    expected_effect_for_alignment = _expected_effect_from_sign(coeff_sign_for_alignment)

    # If the current firing set is expected to move the metric in the SAME direction as the requested
    # intervention (e.g., fires=>increase and --direction increase), flip it to target the opposite regime.
    flipped_for_direction_alignment = False
    if expected_effect_for_alignment in {"increase", "decrease"} and expected_effect_for_alignment == direction:
        mask_fire = ~mask_fire
        rule_used = f"NOT({rule_used})"
        flipped_for_direction_alignment = True

    n_fire = int(mask_fire.sum())
    if n_fire == 0:
        raise RuntimeError(f"Trigger rule fires on 0 topics.\nrule_used={rule_used}")

    df_fire = df[mask_fire].copy()

    # Cap patched topics if requested
    if args.max_patched_topics and args.max_patched_topics > 0 and n_fire > args.max_patched_topics:
        if args.selection == "highest_common" and "common" in df_fire.columns:
            df_fire = df_fire.sort_values("common", ascending=False).head(args.max_patched_topics)
        elif args.selection == "lowest_fog":
            if "gunning_fog" not in df_fire.columns:
                df_fire["gunning_fog"] = df_fire["explanation"].astype(str).apply(gunning_fog)
            df_fire = df_fire.sort_values("gunning_fog", ascending=True).head(args.max_patched_topics)
        else:
            df_fire = df_fire.sample(n=args.max_patched_topics, random_state=args.random_seed)

    # Patch instruction
    patch_instruction = build_patch_instruction(
        metric=args.metric,
        direction=direction,
        extra=args.patch_extra,
        override_file=args.patch_instruction_file,
    )

    prompts = [f'Explain "{t}".' for t in df_fire["topic"].astype(str).tolist()]
    # system_instructions = [patch_instruction] * len(prompts)

    print(f"[Mitigation] model={args.model} difficulty={args.difficulty} metric={args.metric} direction={direction}")
    print(f"[Mitigation] rules_csv={rules_path}")
    print(f"[Mitigation] rule_expr_col={rule_expr_col} coeff_sign_col={coeff_sign_col}")
    print(f"[Mitigation] rule_raw={rule_raw}")
    print(f"[Mitigation] rule_used={rule_used}")
    print(
        f"[Mitigation] coefficient_sign_raw={coeff_sign_raw} expected_effect_raw={expected_effect_raw} "
        f"sign_normalization={args.sign_normalization} coefficient_sign_normalized={coeff_sign_norm} "
        f"expected_effect_normalized={expected_effect_norm} "
        f"coefficient_sign_for_alignment={coeff_sign_for_alignment} "
        f"expected_effect_for_alignment={expected_effect_for_alignment} "
        f"flip_for_sign_normalization={flipped_for_sign_normalization} "
        f"flip_for_direction_alignment={flipped_for_direction_alignment}"
    )
    print(f"[Mitigation] fired={n_fire} patched={len(df_fire)}")

    patched = instruct_model(
        prompts,
        model=args.model,
        temperature=0,
        top_p=0,
        system_instruction=patch_instruction,
    )
    df_fire["patched_explanation"] = [p.strip() if isinstance(p, str) else "" for p in patched]

    # Baseline metric (use existing column if present)
    base_col = f"baseline_{args.metric}"
    patch_col = f"patched_{args.metric}"

    if args.metric in df_fire.columns:
        df_fire[base_col] = df_fire[args.metric].astype(float)
    else:
        df_fire[base_col] = df_fire["explanation"].astype(str).apply(lambda t: compute_metric(t, args.metric))

    df_fire[patch_col] = df_fire["patched_explanation"].astype(str).apply(lambda t: compute_metric(t, args.metric))

    # Always compute spillovers: length + fog
    df_fire["baseline_explanation_length"] = df_fire["explanation"].astype(str).apply(len)
    df_fire["patched_explanation_length"] = df_fire["patched_explanation"].astype(str).apply(len)
    df_fire["baseline_gunning_fog"] = df_fire["explanation"].astype(str).apply(gunning_fog)
    df_fire["patched_gunning_fog"] = df_fire["patched_explanation"].astype(str).apply(gunning_fog)

    # Deltas
    df_fire[f"delta_{args.metric}"] = df_fire[patch_col] - df_fire[base_col]
    df_fire["delta_explanation_length"] = df_fire["patched_explanation_length"] - df_fire["baseline_explanation_length"]
    df_fire["delta_gunning_fog"] = df_fire["patched_gunning_fog"] - df_fire["baseline_gunning_fog"]

    x = df_fire[base_col].to_numpy(dtype=float)
    y = df_fire[patch_col].to_numpy(dtype=float)

    out_dir = args.output_dir or os.path.join("xai_analyses_results", "case_study_mitigation", args.model)
    os.makedirs(out_dir, exist_ok=True)

    safe_rule_id = re.sub(r"[^a-zA-Z0-9]+", "_", rule_used).strip("_")[:80]
    base_name = f"mitigation_{args.metric}_{direction}_{safe_rule_id}"
    out_csv = os.path.join(out_dir, f"{base_name}.csv")
    out_json = os.path.join(out_dir, f"{base_name}.json")

    ex_topics = (
        df_fire.assign(_abs=np.abs(df_fire[f"delta_{args.metric}"]))
              .sort_values("_abs", ascending=False)["topic"]
              .astype(str)
              .head(8)
              .tolist()
    )

    summary: Dict[str, Any] = {
        "model": args.model,
        "difficulty": args.difficulty,
        "metric": args.metric,
        "direction": direction,
        "rules_csv": rules_path,
        "rule_expr_col": rule_expr_col,
        "coeff_sign_col": coeff_sign_col,
        "rule_raw": rule_raw,
        "rule_used": rule_used,
        "coefficient_sign_raw": coeff_sign_raw,
        "coefficient_sign_normalized": coeff_sign_norm,
        "sign_normalization": args.sign_normalization,
        "expected_effect_raw": expected_effect_raw,
        "expected_effect_normalized": expected_effect_norm,
        "coefficient_sign_for_alignment": coeff_sign_for_alignment,
        "expected_effect_for_alignment": expected_effect_for_alignment,
        "flip_for_sign_normalization": bool(flipped_for_sign_normalization),
        "flip_for_direction_alignment": bool(flipped_for_direction_alignment),
        "n_total_baseline": int(len(df)),
        "n_rule_fires": int(n_fire),
        "n_patched": int(len(df_fire)),
        "patch_instruction": patch_instruction,
        "metric_baseline_mean": float(np.nanmean(x)),
        "metric_baseline_std": float(np.nanstd(x, ddof=1)),
        "metric_patched_mean": float(np.nanmean(y)),
        "metric_patched_std": float(np.nanstd(y, ddof=1)),
        "metric_delta_mean": float(np.nanmean(y - x)),
        "metric_delta_median": float(np.nanmedian(y - x)),
        "paired_ttest_pvalue": paired_ttest_pvalue(x, y),
        "cohen_d_paired": cohen_d_paired(x, y),
        "len_baseline_mean": float(df_fire["baseline_explanation_length"].mean()),
        "len_patched_mean": float(df_fire["patched_explanation_length"].mean()),
        "fog_baseline_mean": float(df_fire["baseline_gunning_fog"].mean()),
        "fog_patched_mean": float(df_fire["patched_gunning_fog"].mean()),
        "example_topics_max_abs_delta": ex_topics,
    }

    df_fire.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Mitigation] saved_csv={out_csv}")
    print(f"[Mitigation] saved_json={out_json}")
    print("[Mitigation] summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
