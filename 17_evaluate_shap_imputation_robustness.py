#!/usr/bin/env python3
"""
17_evaluate_shap_imputation_robustness.py

Diagnostics for the nearest-neighbor SHAP-imputation workaround.

This version aligns the diagnostics with the actual abstracted_model(...) logic used
in 5_compute_shap_values.py:
- masked features are matched against rows that attain the *observed minimum* for those features,
  not against the synthetic background literal;
- exact / Hamming diagnostics are computed on *unmasked* coordinates only;
- candidate_count reports the true number of admissible candidates before truncation;
- tie variability is measured over the set of minimum-distance ties, which is what the real
  implementation samples from.

Outputs are written under:
  xai_analyses_results/rebuttal_imputation_robustness/
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from xai_eval_utils import INPUT_FEATURES, METRICS_LIST, load_topic_df

MINIMUM_SCORE = 1
MAXIMUM_SCORE = 5
MIN_VALUE = MINIMUM_SCORE / MAXIMUM_SCORE


def build_background(X: np.ndarray) -> np.ndarray:
    return np.min(X, axis=0) - MIN_VALUE / 10.0


def build_precomputed_rows_with_min(X: np.ndarray) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    mins = np.min(X, axis=0)
    for col in range(X.shape[1]):
        out[col] = np.where(np.isclose(X[:, col], mins[col]))[0]
    return out, mins


def admissible_candidate_indices(masked_features: np.ndarray, X: np.ndarray,
                                 precomputed_rows_with_min: Dict[int, np.ndarray]) -> Tuple[np.ndarray, bool]:
    """
    Return candidate row indices consistent with the workaround in abstracted_model(...).
    Also return whether we had to fall back to all rows because no admissible rows were found.
    """
    if len(masked_features) == 0:
        return np.arange(len(X), dtype=int), False
    counts = np.zeros(len(X), dtype=int)
    for feat_idx in masked_features:
        counts[precomputed_rows_with_min[int(feat_idx)]] += 1
    valid_rows = np.where(counts == len(masked_features))[0]
    if len(valid_rows) == 0:
        return np.arange(len(X), dtype=int), True
    return valid_rows, False


def closest_ties_for_perturbation(x_perturbed: np.ndarray, X: np.ndarray, candidate_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    X_candidates = X[candidate_indices]
    sq_distances = np.sum((X_candidates - x_perturbed) ** 2, axis=1)
    min_distance = float(np.min(sq_distances))
    closest_local = np.where(np.isclose(sq_distances, min_distance))[0]
    return candidate_indices[closest_local], sq_distances[closest_local], min_distance


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        val = float(spearmanr(x, y).statistic)
        return 0.0 if np.isnan(val) else val
    except Exception:
        return 0.0


def nanmean_or_zero(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
    return float(np.nanmean(arr)) if len(arr) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description='Robustness diagnostics for nearest-neighbor SHAP imputation.')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--difficulty', choices=['baseline', 'easy', 'medium', 'hard'], required=True)
    parser.add_argument('--n_samples', type=int, default=256, help='How many original rows to sample.')
    parser.add_argument('--n_masks_per_sample', type=int, default=32, help='How many SHAP-like perturbations per sampled row.')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='xai_analyses_results/rebuttal_imputation_robustness')
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_topic_df(args.model, args.difficulty)
    data_df = df[INPUT_FEATURES + METRICS_LIST].dropna().copy()
    X = data_df[INPUT_FEATURES].to_numpy(dtype=np.float32)
    Y = data_df[METRICS_LIST].to_numpy(dtype=np.float32)
    background = build_background(X)
    precomputed_rows, observed_mins = build_precomputed_rows_with_min(X)

    sample_indices = np.arange(len(X))
    if args.n_samples and args.n_samples < len(sample_indices):
        sample_indices = rng.choice(sample_indices, size=args.n_samples, replace=False)

    diagnostics_rows: List[Dict[str, Any]] = []
    tie_var_rows: List[Dict[str, Any]] = []

    for row_idx in sample_indices:
        x = X[int(row_idx)].copy()
        for perturb_id in range(args.n_masks_per_sample):
            mask_size = int(rng.integers(1, len(INPUT_FEATURES) + 1))
            masked_features = np.sort(rng.choice(len(INPUT_FEATURES), size=mask_size, replace=False))
            unmasked_features = np.array([i for i in range(len(INPUT_FEATURES)) if i not in set(masked_features)], dtype=int)

            x_perturbed = x.copy()
            x_perturbed[masked_features] = background[masked_features]

            candidate_indices, used_fallback = admissible_candidate_indices(masked_features, X, precomputed_rows)
            tie_indices, tie_sq_distances, min_sq_distance = closest_ties_for_perturbation(
                x_perturbed=x_perturbed,
                X=X,
                candidate_indices=candidate_indices,
            )
            selected_global_idx = int(tie_indices[rng.integers(0, len(tie_indices))])
            selected = X[selected_global_idx]

            # Diagnostics aligned with the actual workaround.
            masked_admissible = bool(np.all(np.isclose(selected[masked_features], observed_mins[masked_features]))) if len(masked_features) else True

            if len(unmasked_features) > 0:
                unmasked_diff = selected[unmasked_features] - x[unmasked_features]
                unmasked_l2 = float(np.sqrt(np.sum(unmasked_diff ** 2)))
                unmasked_hamming = int(np.sum(~np.isclose(selected[unmasked_features], x[unmasked_features])))
                unmasked_exact = bool(np.all(np.isclose(selected[unmasked_features], x[unmasked_features])))
                unmasked_near_005 = bool(unmasked_l2 <= 0.05)
                unmasked_near_010 = bool(unmasked_l2 <= 0.10)
            else:
                unmasked_l2 = float('nan')
                unmasked_hamming = float('nan')
                unmasked_exact = float('nan')
                unmasked_near_005 = float('nan')
                unmasked_near_010 = float('nan')

            # Full-space distances kept only as reference, not as the main success metric.
            ref_full_l2_to_perturbed = float(np.sqrt(np.sum((selected - x_perturbed) ** 2)))
            ref_full_hamming_to_perturbed = int(np.sum(~np.isclose(selected, x_perturbed)))

            diagnostics_rows.append({
                'source_row': int(row_idx),
                'perturbation_id': int(perturb_id),
                'masked_feature_count': int(mask_size),
                'unmasked_feature_count': int(len(unmasked_features)),
                'used_fallback_to_all_rows': bool(used_fallback),
                'candidate_count_true': int(len(candidate_indices)),
                'min_tie_count': int(len(tie_indices)),
                'masked_feature_admissible': masked_admissible,
                'exact_match_unmasked': unmasked_exact,
                'near_match_unmasked_le_0_05': unmasked_near_005,
                'near_match_unmasked_le_0_10': unmasked_near_010,
                'l2_distance_unmasked': unmasked_l2,
                'hamming_distance_unmasked': unmasked_hamming,
                'reference_full_l2_to_perturbed': ref_full_l2_to_perturbed,
                'reference_full_hamming_to_perturbed': ref_full_hamming_to_perturbed,
                'selected_index': selected_global_idx,
            })

            tie_outputs = Y[tie_indices]
            selected_output = Y[selected_global_idx]
            for metric_idx, metric in enumerate(METRICS_LIST):
                vals = tie_outputs[:, metric_idx].astype(float)
                tie_var_rows.append({
                    'source_row': int(row_idx),
                    'perturbation_id': int(perturb_id),
                    'metric': metric,
                    'min_tie_count': int(len(tie_indices)),
                    'tie_output_std': float(np.std(vals)),
                    'tie_output_range': float(np.max(vals) - np.min(vals)),
                    'selected_minus_tie_mean_abs': float(abs(float(selected_output[metric_idx]) - float(np.mean(vals)))),
                })

    diag_df = pd.DataFrame(diagnostics_rows)
    tie_df = pd.DataFrame(tie_var_rows)

    tie_summary_rows: List[Dict[str, Any]] = []
    for metric, gdf in tie_df.groupby('metric'):
        gdf_multi = gdf[gdf['min_tie_count'] > 1].copy()
        tie_summary_rows.append({
            'metric': metric,
            'mean_tie_output_std': float(gdf['tie_output_std'].mean()),
            'median_tie_output_std': float(gdf['tie_output_std'].median()),
            'mean_tie_output_range': float(gdf['tie_output_range'].mean()),
            'median_tie_output_range': float(gdf['tie_output_range'].median()),
            'mean_selected_minus_tie_mean_abs': float(gdf['selected_minus_tie_mean_abs'].mean()),
            'n_total_perturbations': int(len(gdf)),
            'n_multi_tie_perturbations': int(len(gdf_multi)),
            'multi_tie_rate': float(len(gdf_multi) / len(gdf)) if len(gdf) else 0.0,
            'mean_tie_output_std_when_gt1': float(gdf_multi['tie_output_std'].mean()) if len(gdf_multi) else 0.0,
            'median_tie_output_std_when_gt1': float(gdf_multi['tie_output_std'].median()) if len(gdf_multi) else 0.0,
            'mean_tie_output_range_when_gt1': float(gdf_multi['tie_output_range'].mean()) if len(gdf_multi) else 0.0,
            'median_tie_output_range_when_gt1': float(gdf_multi['tie_output_range'].median()) if len(gdf_multi) else 0.0,
            'mean_selected_minus_tie_mean_abs_when_gt1': float(gdf_multi['selected_minus_tie_mean_abs'].mean()) if len(gdf_multi) else 0.0,
        })
    tie_summary_df = pd.DataFrame(tie_summary_rows)

    base_name = f'{args.model}_{args.difficulty}_n{len(sample_indices)}_m{args.n_masks_per_sample}'
    diag_path = os.path.join(args.output_dir, f'imputation_diagnostics_{base_name}.csv')
    tie_path = os.path.join(args.output_dir, f'imputation_tie_variability_{base_name}.csv')
    tie_summary_path = os.path.join(args.output_dir, f'imputation_tie_variability_summary_{base_name}.csv')
    diag_df.to_csv(diag_path, index=False)
    tie_df.to_csv(tie_path, index=False)
    tie_summary_df.to_csv(tie_summary_path, index=False)

    diag_no_fallback = diag_df[~diag_df['used_fallback_to_all_rows']].copy() if len(diag_df) else pd.DataFrame()
    diag_multi_tie = diag_df[diag_df['min_tie_count'] > 1].copy() if len(diag_df) else pd.DataFrame()

    summary = {
        'model': args.model,
        'difficulty': args.difficulty,
        'n_source_rows': int(len(sample_indices)),
        'n_perturbations': int(len(diag_df)),
        'masked_feature_admissibility_rate': float(diag_df['masked_feature_admissible'].mean()) if len(diag_df) else 0.0,
        'fallback_to_all_rows_rate': float(diag_df['used_fallback_to_all_rows'].mean()) if len(diag_df) else 0.0,
        'mean_candidate_count_true': float(diag_df['candidate_count_true'].mean()) if len(diag_df) else 0.0,
        'median_candidate_count_true': float(diag_df['candidate_count_true'].median()) if len(diag_df) else 0.0,
        'mean_candidate_count_no_fallback': float(diag_no_fallback['candidate_count_true'].mean()) if len(diag_no_fallback) else 0.0,
        'median_candidate_count_no_fallback': float(diag_no_fallback['candidate_count_true'].median()) if len(diag_no_fallback) else 0.0,
        'mean_min_tie_count': float(diag_df['min_tie_count'].mean()) if len(diag_df) else 0.0,
        'median_min_tie_count': float(diag_df['min_tie_count'].median()) if len(diag_df) else 0.0,
        'mean_min_tie_count_when_gt1': float(diag_multi_tie['min_tie_count'].mean()) if len(diag_multi_tie) else 0.0,
        'median_min_tie_count_when_gt1': float(diag_multi_tie['min_tie_count'].median()) if len(diag_multi_tie) else 0.0,
        'multi_tie_rate': float((diag_df['min_tie_count'] > 1).mean()) if len(diag_df) else 0.0,
        'exact_match_unmasked_rate': nanmean_or_zero(diag_df['exact_match_unmasked']) if len(diag_df) else 0.0,
        'near_match_unmasked_rate_le_0_05': nanmean_or_zero(diag_df['near_match_unmasked_le_0_05']) if len(diag_df) else 0.0,
        'near_match_unmasked_rate_le_0_10': nanmean_or_zero(diag_df['near_match_unmasked_le_0_10']) if len(diag_df) else 0.0,
        'mean_l2_distance_unmasked': nanmean_or_zero(diag_df['l2_distance_unmasked']) if len(diag_df) else 0.0,
        'median_l2_distance_unmasked': float(np.nanmedian(pd.to_numeric(diag_df['l2_distance_unmasked'], errors='coerce').to_numpy(dtype=float))) if len(diag_df) else 0.0,
        'mean_hamming_distance_unmasked': nanmean_or_zero(diag_df['hamming_distance_unmasked']) if len(diag_df) else 0.0,
        'reference_mean_full_l2_to_perturbed': float(diag_df['reference_full_l2_to_perturbed'].mean()) if len(diag_df) else 0.0,
        'reference_mean_full_hamming_to_perturbed': float(diag_df['reference_full_hamming_to_perturbed'].mean()) if len(diag_df) else 0.0,
        'tie_output_variability': tie_summary_df.to_dict(orient='records'),
        'diagnostics_path': diag_path,
        'tie_variability_path': tie_path,
        'tie_variability_summary_path': tie_summary_path,
    }
    summary_path = os.path.join(args.output_dir, f'summary_{base_name}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
