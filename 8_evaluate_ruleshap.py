import os
import json
import argparse
import numpy as np
import pandas as pd

from xai_eval_utils import (
    LLM_MODELS,
    COMPLEXITY_LEVELS,
    K_LEVELS,
    parse_llm_complexity_metric,
    evaluate_rule_file,
    save_summary_tables,
)

parser = argparse.ArgumentParser(description='Evaluate RuleSHAP with ranking and top-k fidelity metrics')
parser.add_argument('--use_shap_in_xgb', action='store_true')
parser.add_argument('--use_shap_in_lasso', action='store_true')
args = parser.parse_args()
use_shap_in_xgb, use_shap_in_lasso = args.use_shap_in_xgb, args.use_shap_in_lasso


directory_path = f'xai_analyses_results/rules/shap_in_xgb={use_shap_in_xgb}+shap_in_lasso={use_shap_in_lasso}'
evaluation_dir = f'xai_analyses_results/evaluation/shap_in_xgb={use_shap_in_xgb}+shap_in_lasso={use_shap_in_lasso}'
os.makedirs(evaluation_dir, exist_ok=True)

mrr_metrics = {
    'explanation_length_easy': ['common > 0.89', 'common <= 0.89'],
    'explanation_length_medium': ['common <= 0.5 & positive > 0.5', 'common <= 0.5 & negative <= 0.70'],
    'explanation_length_hard': ['common <= 0.5 & positive > 0.5', 'common <= 0.5 & negative <= 0.70'],
    'subjectivity_score_nn_medium': ['positive > 0.70', 'negative <= 0.89', 'positive <= 0.70', 'negative > 0.89'],
    'subjectivity_score_nn_hard': ['positive > 0.70', 'negative <= 0.89', 'positive <= 0.70', 'negative > 0.89'],
    'gunning_fog_hard': [
        'interdisciplinary <= 0.5',
        'interdisciplinary > 0.70 & interdisciplinary <= 0.89',
        'interdisciplinary > 0.5 & interdisciplinary <= 0.70',
        'interdisciplinary > 0.89',
    ],
    'oversimplification_easy': ['common > 0.89', 'common <= 0.89'],
    'information_overload_easy': ['common > 0.89', 'common <= 0.89'],
    'oversimplification_medium': ['common <= 0.5 & positive > 0.5', 'common <= 0.5 & negative <= 0.70'],
    'information_overload_medium': ['common <= 0.5 & positive > 0.5', 'common <= 0.5 & negative <= 0.70', 'positive > 0.70', 'negative <= 0.89', 'positive <= 0.70', 'negative > 0.89'],
    'framing_effect_medium': ['positive > 0.70', 'negative <= 0.89', 'positive <= 0.70', 'negative > 0.89'],
    'oversimplification_hard': ['common <= 0.5 & positive > 0.5', 'common <= 0.5 & negative <= 0.70'],
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
    'framing_effect_hard': ['positive > 0.70', 'negative <= 0.89', 'positive <= 0.70', 'negative > 0.89'],
}

rule_counts = {llm: {level: 0 for level in COMPLEXITY_LEVELS} for llm in LLM_MODELS}
rr_results = {llm: {level: {f'RR@{k}': [] for k in K_LEVELS} for level in COMPLEXITY_LEVELS} for llm in LLM_MODELS}
fidelity_rows = []

for file_name in os.listdir(directory_path):
    if not file_name.endswith('.csv'):
        continue
    llm, complexity, metric = parse_llm_complexity_metric(file_name)
    if llm not in LLM_MODELS or complexity not in COMPLEXITY_LEVELS or metric is None:
        continue

    file_path = os.path.join(directory_path, file_name)
    df = pd.read_csv(file_path)
    rule_counts[llm][complexity] += len(df)
    rule_counts[llm]['total'] = rule_counts[llm].get('total', 0) + len(df)

    metric_complexity = f'{metric}_{complexity}'
    if metric_complexity in mrr_metrics:
        sort_col = 'weighted_importance' if 'weighted_importance' in df.columns else ('importance' if 'importance' in df.columns else 'coef')
        df = df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
        for k in K_LEVELS:
            top_k_rules = df.head(k)['rule'].astype(str).tolist()
            j = 0
            found = False
            while not found and j < len(top_k_rules):
                rule = top_k_rules[j]
                found = rule in mrr_metrics[metric_complexity]
                j += 1
            reciprocal_rank = 1 / j if found else 0
            rr_results[llm][complexity][f'RR@{k}'].append(reciprocal_rank)
        fidelity_rows.extend(evaluate_rule_file(file_path, llm, complexity, metric, kind='ruleshap'))

print('rule_counts:', json.dumps(rule_counts, indent=4))
mrr_results = {llm: {level: {f'MRR@{k}': np.mean(rr_results[llm][level][f'RR@{k}']) for k in K_LEVELS} for level in COMPLEXITY_LEVELS} for llm in LLM_MODELS}
for llm in LLM_MODELS:
    mrr_results[llm]['all'] = {f'MRR@{k}': np.mean(sum((rr_results[llm][level][f'RR@{k}'] for level in COMPLEXITY_LEVELS), [])) for k in K_LEVELS}
print('MRR:', json.dumps(mrr_results, indent=4))

pd.DataFrame.from_dict(rule_counts, orient='index').rename_axis('LLM').to_csv(os.path.join(evaluation_dir, 'rule_counts.csv'))
mrr_results_df = pd.DataFrame.from_dict({(llm, level): mrr_results[llm][level] for llm in mrr_results for level in mrr_results[llm]}, orient='index')
mrr_results_df.index = pd.MultiIndex.from_tuples(mrr_results_df.index, names=['LLM', 'Complexity'])
mrr_results_df.to_csv(os.path.join(evaluation_dir, 'mrr_results.csv'))
rr_results_df = pd.DataFrame.from_dict({(llm, complexity): {f'RR@{k}': rr_results[llm][complexity][f'RR@{k}'] for k in K_LEVELS} for llm in rr_results for complexity in rr_results[llm]}, orient='index')
rr_results_df.index = pd.MultiIndex.from_tuples(rr_results_df.index, names=['LLM', 'Complexity'])
rr_results_df.to_csv(os.path.join(evaluation_dir, 'rr_results.csv'))

fidelity_df = pd.DataFrame(fidelity_rows)
save_summary_tables(fidelity_df, evaluation_dir)
print(f'Top-k fidelity results saved under: {evaluation_dir}')
