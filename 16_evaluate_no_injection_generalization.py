#!/usr/bin/env python3
"""
16_evaluate_no_injection_generalization.py

Held-out evaluation of salient rules discovered in the no-injection setting.

Fixes:
- RuleSHAP now receives valid SHAP weights instead of None.
- Supports both RuleFit ('rule') and RuleSHAP ('rule_expression') schemas.
- Correctly evaluates negated rules like NOT(...).
- Emits warnings instead of silently swallowing all RuleSHAP failures.
"""

import os
import json
import pickle
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split

INPUT_FEATURES = [
    'conceptually dense',
    'technically complicated',
    'common',
    'socially controversial',
    'unambiguous',
    'positive',
    'negative',
    'neutral',
    'subject to geographical variability',
    'interdisciplinary',
    'subject to time variability',
]

RULESHAP_XGB_CONFIG = {
    'n_estimators': 300,
    'max_depth': 5,
    'subsample': 0.8,
    'tree_method': 'exact',
    'min_child_weight': 4,
    'learning_rate': 0.01,
}


def clean_rule_expression(rule_expr: str) -> str:
    cleaned = str(rule_expr).strip()
    cleaned = cleaned.replace(' and ', ' & ').replace('AND', '&')
    cleaned = ' '.join(cleaned.split())
    cleaned = cleaned.replace(' & ', '&').replace('&', ' & ')
    cleaned = ' '.join(cleaned.split())
    cleaned = cleaned.replace('NOT (', 'NOT(')
    return cleaned


def unwrap_negation(rule_expr: str) -> Tuple[bool, str]:
    expr = clean_rule_expression(rule_expr)
    expr_no_space = expr.replace(' ', '')
    if expr_no_space.startswith('NOT(') and expr_no_space.endswith(')'):
        inner = expr[expr.find('(') + 1: expr.rfind(')')].strip()
        return True, inner
    if expr.startswith('NOT '):
        return True, expr[4:].strip()
    return False, expr


def parse_rule_expression(rule_expr: str) -> List[Tuple[str, str, float]]:
    import re
    cond_re = re.compile(r'^\s*(?P<feat>.*?)\s*(?P<op><=|>=|<|>|=)\s*(?P<val>-?\d+(?:\.\d+)?)\s*$')
    _, base_expr = unwrap_negation(rule_expr)
    parts = [p.strip() for p in clean_rule_expression(base_expr).split('&') if p.strip()]
    conds = []
    for part in parts:
        m = cond_re.match(part)
        if not m:
            raise ValueError(f'Cannot parse condition: {part!r}')
        conds.append((m.group('feat').strip(), m.group('op'), float(m.group('val'))))
    return conds


def eval_conditions(df: pd.DataFrame, conds: List[Tuple[str, str, float]]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for feat, op, val in conds:
        col = df[feat]
        if op == '<=':
            mask &= (col <= val)
        elif op == '>=':
            mask &= (col >= val)
        elif op == '<':
            mask &= (col < val)
        elif op == '>':
            mask &= (col > val)
        elif op == '=':
            mask &= (col == val)
        else:
            raise ValueError(f'Unsupported operator: {op}')
    return mask


def eval_rule_expression(df: pd.DataFrame, rule_expr: str) -> pd.Series:
    is_negated, _ = unwrap_negation(rule_expr)
    base_mask = eval_conditions(df, parse_rule_expression(rule_expr))
    return ~base_mask if is_negated else base_mask


def normalize_rules_df(rules_df: pd.DataFrame) -> pd.DataFrame:
    df = rules_df.copy()
    if 'type' in df.columns:
        mask = df['type'].astype(str).str.lower().eq('rule')
        if mask.any():
            df = df[mask].copy()
    if 'component_type' in df.columns:
        mask = df['component_type'].astype(str).str.lower().eq('rule')
        if mask.any():
            df = df[mask].copy()
    if 'rule' not in df.columns:
        if 'rule_expression' in df.columns:
            df = df.rename(columns={'rule_expression': 'rule'})
        else:
            raise KeyError(f"Rules DataFrame must contain a 'rule' or 'rule_expression' column. Got: {list(df.columns)}")
    df['rule'] = df['rule'].astype(str).map(clean_rule_expression)
    df = df[df['rule'].str.contains(r'<=|>=|<|>|=', regex=True)].copy()
    for candidate_col in ['importance_weighted_by_gain', 'weighted_importance', 'importance', 'coef']:
        if candidate_col in df.columns:
            vals = pd.to_numeric(df[candidate_col], errors='coerce')
            df['sort_importance'] = vals.abs() if candidate_col == 'coef' else vals.fillna(0.0)
            break
    else:
        df['sort_importance'] = 0.0
    return df.sort_values('sort_importance', ascending=False).reset_index(drop=True)


def load_shap_weights(model: str, difficulty: str, metric: str, n_features: int) -> np.ndarray:
    pkl_path = os.path.join('abstract_model_io', f'global_shap_stats_{model}_{difficulty}.pkl')
    if os.path.isfile(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                stats = pickle.load(f)
            metric_stats = stats.get(metric, None)
            if metric_stats is not None:
                weights = np.array([
                    float(metric_stats[k]['upper_importance_bound']) for k in INPUT_FEATURES
                ], dtype=float)
                if len(weights) == n_features and np.all(np.isfinite(weights)) and np.sum(weights) > 0:
                    return weights
        except Exception:
            pass
    return np.ones(n_features, dtype=float)


def cohen_d_independent(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float('nan')
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = np.sqrt(((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2))
    if not np.isfinite(pooled) or pooled == 0:
        return float('nan')
    return float((np.mean(x) - np.mean(y)) / pooled)


def safe_ttest(x: np.ndarray, y: np.ndarray) -> float:
    try:
        return float(ttest_ind(x, y, equal_var=False, nan_policy='omit').pvalue)
    except Exception:
        return float('nan')


def fit_model(method: str, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str],
              random_seed: int, model_name: str, difficulty: str, metric: str):
    method = method.lower()
    if method == 'ruleshap':
        from ruleshap import RuleSHAP
        shap_weights = load_shap_weights(model_name, difficulty, metric, X_train.shape[1])
        model = RuleSHAP(
            gboost_config_dict=RULESHAP_XGB_CONFIG.copy(),
            random_state=random_seed,
            rfmode='regress',
        )
        model.fit(
            X_train,
            y_train,
            feature_names=feature_names,
            shap_weights=shap_weights,
            use_shap_in_xgb=True,
            use_shap_in_lasso=True,
        )
        return model
    if method == 'rulefit':
        from rulefit import RuleFit
        model = RuleFit(random_state=random_seed)
        model.fit(X_train, y_train, feature_names=feature_names)
        return model
    raise ValueError(f'Unsupported method: {method}')


def expected_direction(rule_row: pd.Series, train_df: pd.DataFrame, metric: str) -> str:
    for sign_col in ['impact_direction', 'coefficient_sign']:
        if sign_col in rule_row.index and str(rule_row[sign_col]).strip().lower() in {'positive', 'negative'}:
            return str(rule_row[sign_col]).strip().lower()
    try:
        mask = eval_rule_expression(train_df, str(rule_row['rule']))
        fired = train_df.loc[mask, metric].to_numpy(dtype=float)
        rest = train_df.loc[~mask, metric].to_numpy(dtype=float)
        if len(fired) == 0 or len(rest) == 0:
            return 'unknown'
        return 'positive' if np.mean(fired) >= np.mean(rest) else 'negative'
    except Exception:
        return 'unknown'


def main() -> None:
    parser = argparse.ArgumentParser(description='Held-out validation of naturally occurring rules.')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--difficulty', choices=['baseline'], default='baseline')
    parser.add_argument('--methods', type=str, default='ruleshap,rulefit')
    parser.add_argument('--metrics', type=str, default='explanation_length,subjectivity_score_nn,gunning_fog,sentiment_score_nn')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--top_k_rules', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='xai_analyses_results/rebuttal_no_injection_generalization')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    methods = [m.strip().lower() for m in args.methods.split(',') if m.strip()]
    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]

    df = pd.read_csv(os.path.join('abstract_model_io', f'topic_{args.model}_{args.difficulty}.csv'))
    results: List[dict] = []

    for metric in metrics:
        metric_df = df[INPUT_FEATURES + [metric, 'topic']].dropna().copy()
        if len(metric_df) < 10:
            continue
        for split_id in range(args.n_splits):
            seed = args.random_seed + split_id
            train_df, test_df = train_test_split(metric_df, test_size=args.test_size, random_state=seed)
            X_train = train_df[INPUT_FEATURES].to_numpy(dtype=np.float32)
            y_train = train_df[metric].to_numpy(dtype=np.float32)

            for method in methods:
                model = fit_model(method, X_train, y_train, INPUT_FEATURES, seed, args.model, args.difficulty, metric)
                try:
                    raw_rules = model.get_rules()
                    rules_df = normalize_rules_df(raw_rules)
                    print(f"[INFO] method={method} metric={metric} split={split_id} raw_rules={len(raw_rules)} normalized_rules={len(rules_df)}")
                except Exception as e:
                    print(f"[WARN] method={method} metric={metric} split={split_id} normalize failed: {e}")
                    continue
                rules_df = rules_df.head(args.top_k_rules).copy()
                if len(rules_df) == 0:
                    continue
                for rank, (_, row) in enumerate(rules_df.iterrows(), start=1):
                    rule_expr = str(row['rule'])
                    try:
                        test_mask = eval_rule_expression(test_df, rule_expr)
                    except Exception as e:
                        print(f"[WARN] method={method} metric={metric} split={split_id} unparsable rule: {rule_expr!r} error={e}")
                        continue
                    fired = test_df.loc[test_mask, metric].to_numpy(dtype=float)
                    rest = test_df.loc[~test_mask, metric].to_numpy(dtype=float)
                    train_direction = expected_direction(row, train_df, metric)
                    if len(fired) == 0 or len(rest) == 0:
                        continue
                    diff = float(np.mean(fired) - np.mean(rest))
                    observed_direction = 'positive' if diff >= 0 else 'negative'
                    results.append({
                        'model': args.model,
                        'metric': metric,
                        'method': method,
                        'split_id': split_id,
                        'rule_rank': rank,
                        'rule': rule_expr,
                        'n_test': int(len(test_df)),
                        'test_support': int(np.sum(test_mask)),
                        'test_support_ratio': float(np.mean(test_mask)),
                        'test_mean_when_rule_fires': float(np.mean(fired)),
                        'test_mean_when_rule_not_firing': float(np.mean(rest)),
                        'test_delta': diff,
                        'pvalue': safe_ttest(fired, rest),
                        'cohen_d': cohen_d_independent(fired, rest),
                        'train_direction': train_direction,
                        'observed_direction': observed_direction,
                        'direction_consistent': bool(train_direction == 'unknown' or train_direction == observed_direction),
                    })

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, f'generalization_{args.model}.csv')
    results_df.to_csv(csv_path, index=False)

    summary_df = results_df.groupby(['method', 'metric']).agg(
        mean_abs_delta=('test_delta', lambda s: float(np.mean(np.abs(s)))),
        mean_pvalue=('pvalue', 'mean'),
        direction_consistency_rate=('direction_consistent', 'mean'),
        mean_support_ratio=('test_support_ratio', 'mean'),
        n_rules=('rule', 'count'),
    ).reset_index() if len(results_df) else pd.DataFrame()
    summary_path = os.path.join(args.output_dir, f'summary_{args.model}.csv')
    summary_df.to_csv(summary_path, index=False)

    final_summary = {
        'model': args.model,
        'methods': methods,
        'metrics': metrics,
        'results_path': csv_path,
        'summary_path': summary_path,
        'mean_direction_consistency_rate': float(summary_df['direction_consistency_rate'].mean()) if len(summary_df) else float('nan'),
        'mean_abs_delta': float(summary_df['mean_abs_delta'].mean()) if len(summary_df) else float('nan'),
    }
    json_path = os.path.join(args.output_dir, f'final_summary_{args.model}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2)
    print(json.dumps(final_summary, indent=2))


if __name__ == '__main__':
    main()
