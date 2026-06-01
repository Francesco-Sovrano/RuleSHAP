import os
import re
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, matthews_corrcoef, f1_score
from sklearn.model_selection import train_test_split

K_LEVELS = [1, 3, 10]
LLM_MODELS = [
    'gpt-3.5-turbo',
    'gpt-4o-mini',
    'gpt-4o',
    'llama3.1',
    'llama3.1:70b',
]
COMPLEXITY_LEVELS = ['easy', 'medium', 'hard']
SURROGATE_TEST_SIZE = 0.2
SURROGATE_RANDOM_SEEDS = [42, 43, 44, 45, 46]
SURROGATE_MIN_TRAIN = 12
SURROGATE_MIN_TEST = 6

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

METRICS_LIST = [
    'explanation_length',
    'subjectivity_score_nn',
    'gunning_fog',
    'sentiment_score_nn',
    'framing_effect',
    'information_overload',
    'oversimplification',
]

GROUND_TRUTH_RULES: Dict[str, List[str]] = {
    'explanation_length_easy': [
        'common > 0.89',
        'common <= 0.89',
    ],
    'explanation_length_medium': [
        'common <= 0.5 & positive > 0.5',
        'common <= 0.5 & negative <= 0.70',
    ],
    'explanation_length_hard': [
        'common <= 0.5 & positive > 0.5',
        'common <= 0.5 & negative <= 0.70',
    ],
    'subjectivity_score_nn_medium': [
        'positive > 0.70',
        'negative <= 0.89',
        'positive <= 0.70',
        'negative > 0.89',
    ],
    'subjectivity_score_nn_hard': [
        'positive > 0.70',
        'negative <= 0.89',
        'positive <= 0.70',
        'negative > 0.89',
    ],
    'gunning_fog_hard': [
        'interdisciplinary <= 0.5',
        'interdisciplinary > 0.70 & interdisciplinary <= 0.89',
        'interdisciplinary > 0.5 & interdisciplinary <= 0.70',
        'interdisciplinary > 0.89',
    ],
    'oversimplification_easy': [
        'common > 0.89',
        'common <= 0.89',
    ],
    'information_overload_easy': [
        'common > 0.89',
        'common <= 0.89',
    ],
    'oversimplification_medium': [
        'common <= 0.5 & positive > 0.5',
        'common <= 0.5 & negative <= 0.70',
    ],
    'information_overload_medium': [
        'common <= 0.5 & positive > 0.5',
        'common <= 0.5 & negative <= 0.70',
        'positive > 0.70',
        'negative <= 0.89',
        'positive <= 0.70',
        'negative > 0.89',
    ],
    'framing_effect_medium': [
        'positive > 0.70',
        'negative <= 0.89',
        'positive <= 0.70',
        'negative > 0.89',
    ],
    'oversimplification_hard': [
        'common <= 0.5 & positive > 0.5',
        'common <= 0.5 & negative <= 0.70',
    ],
    'information_overload_hard': [
        'interdisciplinary <= 0.5',
        'interdisciplinary > 0.70 & interdisciplinary <= 0.89',
        'interdisciplinary > 0.5 & interdisciplinary <= 0.70',
        'interdisciplinary > 0.89',
        'common <= 0.5 & positive > 0.5',
        'common <= 0.5 & negative <= 0.70',
        'positive > 0.70',
        'negative <= 0.89',
        'positive <= 0.70',
        'negative > 0.89',
    ],
    'framing_effect_hard': [
        'positive > 0.70',
        'negative <= 0.89',
        'positive <= 0.70',
        'negative > 0.89',
    ],
}

_COND_RE = re.compile(r'^\s*(?P<feat>.*?)\s*(?P<op><=|>=|<|>|=)\s*(?P<val>-?\d+(?:\.\d+)?)\s*$')


def load_topic_df(model: str, difficulty: str, csv_dir: str = 'abstract_model_io') -> pd.DataFrame:
    path = os.path.join(csv_dir, f'topic_{model}_{difficulty}.csv')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Cannot find dataset: {path}')
    return pd.read_csv(path)


def load_global_shap_stats(model: str, difficulty: str, csv_dir: str = 'abstract_model_io') -> Dict[str, Dict[str, Dict[str, float]]]:
    path = os.path.join(csv_dir, f'global_shap_stats_{model}_{difficulty}.pkl')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Cannot find SHAP stats cache: {path}')
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_shap_weights(model: str, difficulty: str, metric: str, csv_dir: str = 'abstract_model_io') -> np.ndarray:
    stats = load_global_shap_stats(model, difficulty, csv_dir=csv_dir)
    metric_stats = stats[metric]
    return np.array([metric_stats[k]['upper_importance_bound'] for k in INPUT_FEATURES], dtype=float)


def rule_key(metric: str, difficulty: str) -> str:
    return f'{metric}_{difficulty}'


def clean_rule_expression(rule_expr: str) -> str:
    if not isinstance(rule_expr, str):
        return ''
    cleaned = rule_expr.strip()
    cleaned = cleaned.replace(' and ', ' & ')
    cleaned = cleaned.replace('AND', '&')
    cleaned = cleaned.replace('(', ' ').replace(')', ' ')
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.replace(' & ', '&')
    cleaned = cleaned.replace('&', ' & ')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def parse_rule_expression(rule_expr: str) -> List[Tuple[str, str, float]]:
    cleaned = clean_rule_expression(rule_expr)
    if not cleaned:
        return []
    parts = [p.strip() for p in cleaned.split('&') if p.strip()]
    conds: List[Tuple[str, str, float]] = []
    for part in parts:
        match = _COND_RE.match(part)
        if not match:
            raise ValueError(f'Cannot parse condition: {part!r} (rule={rule_expr!r})')
        conds.append((match.group('feat').strip(), match.group('op'), float(match.group('val'))))
    return conds


def eval_conditions(df: pd.DataFrame, conds: List[Tuple[str, str, float]]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for feat, op, val in conds:
        if feat not in df.columns:
            raise KeyError(f'Missing feature column {feat!r}')
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


def is_rule_like(rule_expr: str) -> bool:
    expr = clean_rule_expression(rule_expr)
    return any(op in expr for op in ('<=', '>=', '<', '>', '='))


def normalize_rules_df(rules_df: pd.DataFrame) -> pd.DataFrame:
    df = rules_df.copy()
    if 'type' in df.columns:
        mask = df['type'].astype(str).str.lower().eq('rule')
        if mask.any():
            df = df[mask].copy()

    # RuleSHAP exports `rule_expression`; RuleFit/baselines export `rule`.
    # Normalize both schemas to a canonical `rule` column for all evaluators.
    if 'rule' not in df.columns and 'rule_expression' in df.columns:
        df['rule'] = df['rule_expression']
    if 'rule' not in df.columns:
        raise KeyError(
            "Rules DataFrame must contain a 'rule' or 'rule_expression' column. "
            f"Got: {list(df.columns)}"
        )
    df['rule'] = df['rule'].astype(str).map(clean_rule_expression)
    df = df[df['rule'].map(is_rule_like)].copy()
    if 'weighted_importance' in df.columns:
        df['sort_importance'] = pd.to_numeric(df['weighted_importance'], errors='coerce').fillna(0.0)
    elif 'importance' in df.columns:
        df['sort_importance'] = pd.to_numeric(df['importance'], errors='coerce').fillna(0.0)
    elif 'coef' in df.columns:
        df['sort_importance'] = pd.to_numeric(df['coef'], errors='coerce').abs().fillna(0.0)
    else:
        df['sort_importance'] = 0.0
    return df.sort_values('sort_importance', ascending=False).reset_index(drop=True)


def f1_precision_recall_iou(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = float(np.sum(y_true & y_pred))
    fp = float(np.sum(~y_true & y_pred))
    fn = float(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def safe_mean(values: List[float]) -> float:
    vals = [float(v) for v in values if pd.notna(v)]
    return float(np.mean(vals)) if vals else float('nan')


def safe_median(values: List[float]) -> float:
    vals = [float(v) for v in values if pd.notna(v)]
    return float(np.median(vals)) if vals else float('nan')


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        val = float(spearmanr(y_true, y_pred).statistic)
        return 0.0 if np.isnan(val) else val
    except Exception:
        return 0.0


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return float('nan')
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if np.isclose(sst, 0.0):
        return 0.0
    sse = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (sse / sst)


def sort_rule_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'type' in out.columns:
        mask = out['type'].astype(str).str.lower().eq('rule')
        if mask.any():
            out = out[mask].copy()
    if 'rule' in out.columns:
        out['rule'] = out['rule'].astype(str)
    sort_col = None
    for col in ['weighted_importance', 'importance', 'abs_coef', 'coef']:
        if col in out.columns:
            sort_col = col
            break
    if sort_col is None:
        out['__sort__'] = 0.0
    else:
        vals = pd.to_numeric(out[sort_col], errors='coerce').fillna(0.0)
        if sort_col == 'coef':
            vals = vals.abs()
        out['__sort__'] = vals
    out = out.sort_values('__sort__', ascending=False).drop(columns='__sort__').reset_index(drop=True)
    return out


def parse_dtree_rule(rule_str: str) -> Tuple[List[Tuple[str, str, float]], Optional[float]]:
    text = str(rule_str).strip()
    response = None
    match = re.search(r'then\s+response\s*:\s*([-+]?\d*\.?\d+)', text, flags=re.I)
    if match:
        response = float(match.group(1))
        text = text[:match.start()].strip()
    text = text.replace('If ', '').replace('if ', '')
    text = text.replace(' and ', ' & ')
    text = text.replace('(', ' ').replace(')', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    conds = parse_rule_expression(text)
    return conds, response


def parse_rule_expression_auto(rule_str: str) -> List[Tuple[str, str, float]]:
    text = str(rule_str).strip()
    lowered = text.lower()
    parsers = []
    if lowered.startswith('if ') or ' then' in lowered or 'response:' in lowered:
        parsers = [parse_dtree_rule, parse_rule_expression]
    else:
        parsers = [parse_rule_expression, parse_dtree_rule]
    last_error = None
    for parser in parsers:
        try:
            parsed = parser(text)
            if isinstance(parsed, tuple):
                conds, _ = parsed
            else:
                conds = parsed
            if conds:
                return conds
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return []


def rule_mask_from_str(df: pd.DataFrame, rule_str: str, kind: str = 'generic') -> pd.Series:
    if kind == 'dtree':
        conds, _ = parse_dtree_rule(rule_str)
    elif kind in ('gelpe', 'auto'):
        conds = parse_rule_expression_auto(rule_str)
    else:
        conds = parse_rule_expression(rule_str)
    return eval_conditions(df, conds)


def first_exact_semantic_match_rank(
    df_features: pd.DataFrame,
    selected_rules: List[str],
    gt_rules: List[str],
    kind: str = 'generic',
) -> Optional[int]:
    if not selected_rules or not gt_rules:
        return None
    gt_masks = []
    for gt in gt_rules:
        try:
            gt_mask = rule_mask_from_str(df_features, gt, kind='generic').to_numpy(dtype=bool)
            gt_masks.append(gt_mask)
        except Exception:
            continue
    if not gt_masks:
        return None
    for rank, rule in enumerate(selected_rules, start=1):
        try:
            pred_mask = rule_mask_from_str(df_features, rule, kind=kind).to_numpy(dtype=bool)
        except Exception:
            continue
        if any(np.array_equal(pred_mask, gt_mask) for gt_mask in gt_masks):
            return rank
    return None


def get_topic_df(llm: str, complexity: str) -> pd.DataFrame:
    path = os.path.join('abstract_model_io', f'topic_{llm}_{complexity}.csv')
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def metric_complexity_key(metric: str, complexity: str) -> str:
    return f'{metric}_{complexity}'


def continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'r2': float(safe_r2(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'spearman': float(safe_spearman(y_true, y_pred)),
    }


def best_trigger_overlap_metrics(df_features: pd.DataFrame, selected_rules: List[str], gt_rules: List[str], kind: str = 'generic') -> Dict[str, float]:
    """
    Option A: semantic best-match scoring.

    For the selected top-k rules, score the *single best semantic match* against the
    available ground-truth triggers, rather than averaging coverage over all triggers.
    This makes TopK=1 comparable to MRR@1: if the first retrieved rule semantically
    matches one true trigger well, the overlap metric should reflect that.
    """
    if not gt_rules or not selected_rules:
        return {'mcc': float('nan'), 'f1': float('nan')}

    global_best_mcc = -1.0
    global_best_f1 = 0.0

    for gt in gt_rules:
        try:
            gt_mask = rule_mask_from_str(df_features, gt, kind='generic').to_numpy(dtype=bool)
        except Exception:
            continue
        for rule in selected_rules:
            try:
                pred_mask = rule_mask_from_str(df_features, rule, kind=kind).to_numpy(dtype=bool)
            except Exception:
                continue
            try:
                mcc = float(matthews_corrcoef(gt_mask, pred_mask))
            except Exception:
                mcc = 0.0
            try:
                f1 = float(f1_score(gt_mask, pred_mask, zero_division=0))
            except Exception:
                f1 = 0.0
            if mcc > global_best_mcc:
                global_best_mcc = mcc
                global_best_f1 = f1

    if global_best_mcc < -1.0:
        return {'mcc': float('nan'), 'f1': float('nan')}
    return {
        'mcc': float(global_best_mcc),
        'f1': float(global_best_f1),
    }


def parse_llm_complexity_metric(file_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    parts = file_name.split('_')
    if len(parts) < 5:
        return None, None, None
    llm = parts[2]
    complexity = parts[3]
    metric = '_'.join(parts[4:]).replace('.csv', '')
    return llm, complexity, metric


def _safe_train_test_split(X: np.ndarray, y: np.ndarray, seed: int):
    n = len(y)
    if n < (SURROGATE_MIN_TRAIN + SURROGATE_MIN_TEST):
        return None
    test_n = max(int(round(n * SURROGATE_TEST_SIZE)), SURROGATE_MIN_TEST)
    test_n = min(test_n, n - SURROGATE_MIN_TRAIN)
    if test_n <= 0 or (n - test_n) <= 0:
        return None
    return train_test_split(X, y, test_size=test_n, random_state=seed)


def _deduplicate_constant_columns(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2 or X.shape[1] == 0:
        return X
    keep = np.var(X, axis=0) > 0
    if not np.any(keep):
        return np.zeros((X.shape[0], 1), dtype=float)
    return X[:, keep]


def heldout_surrogate_metrics_from_design(X_design: np.ndarray, y: np.ndarray, seeds: List[int] = None) -> Dict[str, float]:
    if seeds is None:
        seeds = SURROGATE_RANDOM_SEEDS
    X_design = np.asarray(X_design, dtype=float)
    y = np.asarray(y, dtype=float)
    if X_design.ndim == 1:
        X_design = X_design.reshape(-1, 1)
    if len(y) == 0:
        return {'r2': float('nan'), 'mae': float('nan'), 'spearman': float('nan')}
    X_design = _deduplicate_constant_columns(X_design)

    r2_vals = []
    mae_vals = []
    spearman_vals = []
    successful = 0

    for seed in seeds:
        split = _safe_train_test_split(X_design, y, seed)
        if split is None:
            break
        X_train, X_test, y_train, y_test = split
        if X_train.shape[1] == 0:
            y_pred = np.full(len(y_test), float(np.mean(y_train)))
        else:
            try:
                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)
                y_pred = np.asarray(model.predict(X_test), dtype=float)
            except Exception:
                y_pred = np.full(len(y_test), float(np.mean(y_train)))
        metrics = continuous_metrics(y_test, y_pred)
        r2_vals.append(metrics['r2'])
        mae_vals.append(metrics['mae'])
        spearman_vals.append(metrics['spearman'])
        successful += 1

    if successful == 0:
        baseline = np.full(len(y), float(np.mean(y)))
        return continuous_metrics(y, baseline)

    return {
        'r2': safe_median(r2_vals),
        'mae': safe_median(mae_vals),
        'spearman': safe_median(spearman_vals),
    }


def build_rule_activation_matrix(df_features: pd.DataFrame, selected_rules: List[str], kind: str) -> np.ndarray:
    cols = []
    for rule in selected_rules:
        try:
            mask = rule_mask_from_str(df_features, rule, kind=kind).to_numpy(dtype=float)
        except Exception:
            continue
        cols.append(mask)
    if not cols:
        return np.zeros((len(df_features), 1), dtype=float)
    X = np.column_stack(cols).astype(float)
    return _deduplicate_constant_columns(X)


def evaluate_rule_file(file_path: str, llm: str, complexity: str, metric: str, kind: str, k_levels: List[int] = None) -> List[Dict[str, object]]:
    if k_levels is None:
        k_levels = K_LEVELS
    df_rules = pd.read_csv(file_path)
    df_rules = sort_rule_df(df_rules)
    topic_df = get_topic_df(llm, complexity)
    data_df = topic_df[INPUT_FEATURES + [metric]].dropna().copy()
    X_df = data_df[INPUT_FEATURES].copy()
    y = data_df[metric].to_numpy(dtype=float)
    key = metric_complexity_key(metric, complexity)
    gt_rules = GROUND_TRUTH_RULES.get(key, [])

    rows = []
    for k in k_levels:
        selected_rules = df_rules.head(k)['rule'].astype(str).tolist() if 'rule' in df_rules.columns else []
        X_design = build_rule_activation_matrix(X_df, selected_rules, kind=kind)
        metrics = heldout_surrogate_metrics_from_design(X_design, y)
        overlap = best_trigger_overlap_metrics(X_df, selected_rules, gt_rules, kind=kind)
        rows.append({
            'LLM': llm,
            'Complexity': complexity,
            'Metric': metric,
            'TopK': int(k),
            'RuleCountFile': int(len(df_rules)),
            'EvalMode': 'heldout_refit_on_rule_activations',
            'r2': metrics['r2'],
            'mae': metrics['mae'],
            'spearman': metrics['spearman'],
            'mcc': overlap['mcc'],
            'f1': overlap['f1'],
        })
    return rows


def build_topk_feature_design(df_features: pd.DataFrame, ranked_features: List[str], k: int) -> np.ndarray:
    selected = [f for f in ranked_features[:k] if f in df_features.columns]
    if not selected:
        return np.zeros((len(df_features), 1), dtype=float)
    X = df_features[selected].to_numpy(dtype=float)
    return _deduplicate_constant_columns(X)


def evaluate_shap_feature_fidelity(llm: str, complexity: str, metric: str, k_levels: List[int] = None) -> List[Dict[str, object]]:
    if k_levels is None:
        k_levels = K_LEVELS
    path = os.path.join('abstract_model_io', f'global_shap_stats_{llm}_{complexity}.pkl')
    with open(path, 'rb') as f:
        metric_global_feature_stats_dict = pickle.load(f)
    global_feature_stats = metric_global_feature_stats_dict[metric]
    ranked_features = sorted(global_feature_stats.keys(), key=lambda k: global_feature_stats[k]['upper_importance_bound'], reverse=True)
    topic_df = get_topic_df(llm, complexity)
    data_df = topic_df[INPUT_FEATURES + [metric]].dropna().copy()
    X_df = data_df[INPUT_FEATURES].copy()
    y = data_df[metric].to_numpy(dtype=float)

    rows = []
    for k in k_levels:
        X_design = build_topk_feature_design(X_df, ranked_features, k)
        metrics = heldout_surrogate_metrics_from_design(X_design, y)
        rows.append({
            'LLM': llm,
            'Complexity': complexity,
            'Metric': metric,
            'TopK': int(k),
            'FeatureCount': int(min(k, len(ranked_features))),
            'EvalMode': 'heldout_refit_on_topk_features',
            'r2': metrics['r2'],
            'mae': metrics['mae'],
            'spearman': metrics['spearman'],
            'mcc': float('nan'),
            'f1': float('nan'),
        })
    return rows


def save_summary_tables(detail_df: pd.DataFrame, evaluation_dir: str, prefix: str = '') -> None:
    os.makedirs(evaluation_dir, exist_ok=True)
    if len(detail_df) == 0:
        return
    detail_path = os.path.join(evaluation_dir, f'{prefix}topk_fidelity_detail.csv')
    detail_df.to_csv(detail_path, index=False)

    summary_df = detail_df.groupby(['LLM', 'Complexity', 'TopK'])[['r2', 'mae', 'spearman', 'mcc', 'f1']].mean(numeric_only=True).reset_index()
    summary_path = os.path.join(evaluation_dir, f'{prefix}topk_fidelity_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    overall_rows = []
    for topk_key, group in summary_df.groupby(['TopK']):
        topk = topk_key[0] if isinstance(topk_key, tuple) else topk_key
        mcc_values = pd.to_numeric(group['mcc'], errors='coerce').dropna()
        if len(mcc_values) == 0:
            overall_rows.append({
                'TopK': int(topk),
                'mcc_mean': float('nan'),
                'mcc_std': float('nan'),
                'mcc_q1': float('nan'),
                'mcc_median': float('nan'),
                'mcc_q3': float('nan'),
            })
        else:
            overall_rows.append({
                'TopK': int(topk),
                'mcc_mean': float(mcc_values.mean()),
                'mcc_std': float(mcc_values.std(ddof=1)) if len(mcc_values) > 1 else 0.0,
                'mcc_q1': float(mcc_values.quantile(0.25)),
                'mcc_median': float(mcc_values.median()),
                'mcc_q3': float(mcc_values.quantile(0.75)),
            })
    overall_df = pd.DataFrame(overall_rows).sort_values('TopK').reset_index(drop=True)
    overall_path = os.path.join(evaluation_dir, f'{prefix}topk_fidelity_overall.csv')
    overall_df.to_csv(overall_path, index=False)
