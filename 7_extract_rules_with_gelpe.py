import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import json
import argparse

import numpy as np
import pandas as pd

from gelpe import GELPE
from lib import load_cache


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Extract GELPE-inspired baseline rules for the structured abstraction benchmark.'
    )
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument(
        '--difficulty',
        choices=['hard', 'medium', 'easy', 'baseline'],
        required=True,
    )
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument(
        '--gelpe_top_k',
        type=int,
        default=6,
        help='Top-K abstractions selected by aggregated SHAP relevance.',
    )
    parser.add_argument(
        '--gelpe_max_depth',
        type=int,
        default=5,
        help='Maximum depth of the GELPE CART surrogate.',
    )
    return parser


def get_metric_shap_weights(metric_global_feature_stats_dict, metric):
    if not metric_global_feature_stats_dict or metric not in metric_global_feature_stats_dict:
        return np.ones(len(INPUT_FEATURES), dtype=float)
    global_feature_stats = metric_global_feature_stats_dict[metric]
    return np.array([
        float(global_feature_stats[k]['upper_importance_bound']) for k in INPUT_FEATURES
    ], dtype=float)


def main() -> None:
    args = build_parser().parse_args()
    model = args.model
    difficulty = args.difficulty
    random_seed = args.random_seed
    np.random.seed(random_seed)

    print('7_extract_rules_with_gelpe', args)

    csv_file = os.path.join('abstract_model_io', f'topic_{model}_{difficulty}.csv')
    if not os.path.exists(csv_file):
        raise FileNotFoundError(csv_file)
    df = pd.read_csv(csv_file)

    feature_stats_dir = os.path.join('xai_analyses_results', 'feature_stats')
    shap_stats_file = os.path.join(
        feature_stats_dir,
        f'global_feature_stats_{model}_{difficulty}.json',
    )
    metric_global_feature_stats_dict = load_cache(shap_stats_file)

    rule_output_dir = os.path.join('xai_analyses_results', 'baseline_rules')
    os.makedirs(rule_output_dir, exist_ok=True)

    for metric in METRICS_LIST:
        if metric not in df.columns:
            continue

        sub_df = df[INPUT_FEATURES + [metric]].dropna().copy()
        if sub_df.empty:
            continue

        X = sub_df[INPUT_FEATURES].to_numpy(dtype=np.float32)
        y = sub_df[metric].to_numpy(dtype=np.float32)
        shap_weights = get_metric_shap_weights(metric_global_feature_stats_dict, metric)

        gelpe_model = GELPE(
            top_k_features=args.gelpe_top_k,
            max_depth=args.gelpe_max_depth,
            random_state=random_seed,
            rfmode='regress',
        )
        gelpe_model.fit(X, y, feature_names=INPUT_FEATURES, shap_weights=shap_weights)
        gelpe_rules = gelpe_model.get_rules()

        out_file = os.path.join(
            rule_output_dir,
            f'gelpe_rules_{model}_{difficulty}_{metric}.csv',
        )
        gelpe_rules.to_csv(out_file, index=False)

        print(json.dumps({
            'metric': metric,
            'n_rows': int(len(sub_df)),
            'gelpe_top_k': args.gelpe_top_k,
            'gelpe_max_depth': args.gelpe_max_depth,
            'gelpe_selected_features': gelpe_model.selected_feature_names_,
            'gelpe_rules': int(len(gelpe_rules)),
            'output_file': out_file,
        }, indent=2))


if __name__ == '__main__':
    main()
