import os
import json
import numpy as np
import pandas as pd

from xai_eval_utils import (
    LLM_MODELS,
    COMPLEXITY_LEVELS,
    K_LEVELS,
    GROUND_TRUTH_RULES,
    INPUT_FEATURES,
    parse_llm_complexity_metric,
    evaluate_rule_file,
    save_summary_tables,
    get_topic_df,
    metric_complexity_key,
    first_exact_semantic_match_rank,
)


directory_path = 'xai_analyses_results/baseline_rules'
evaluation_dir = 'xai_analyses_results/evaluation/gelpe'
os.makedirs(evaluation_dir, exist_ok=True)

rule_counts = {llm: {level: 0 for level in COMPLEXITY_LEVELS} for llm in LLM_MODELS}
rr_results = {llm: {level: {f'RR@{k}': [] for k in K_LEVELS} for level in COMPLEXITY_LEVELS} for llm in LLM_MODELS}
fidelity_rows = []

for file_name in os.listdir(directory_path):
    if not file_name.endswith('.csv') or not file_name.startswith('gelpe_rules_'):
        continue
    llm, complexity, metric = parse_llm_complexity_metric(file_name)
    if llm not in LLM_MODELS or complexity not in COMPLEXITY_LEVELS or metric is None:
        continue

    file_path = os.path.join(directory_path, file_name)
    df = pd.read_csv(file_path)
    rule_counts[llm][complexity] += len(df)
    rule_counts[llm]['total'] = rule_counts[llm].get('total', 0) + len(df)

    key = metric_complexity_key(metric, complexity)
    gt_rules = GROUND_TRUTH_RULES.get(key, [])
    if gt_rules and 'rule' in df.columns:
        topic_df = get_topic_df(llm, complexity)
        data_df = topic_df[INPUT_FEATURES + [metric]].dropna().copy()
        X_df = data_df[INPUT_FEATURES].copy()

        if 'weighted_importance' in df.columns:
            sort_col = 'weighted_importance'
        elif 'importance' in df.columns:
            sort_col = 'importance'
        else:
            sort_col = None
        if sort_col is not None:
            df = df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

        for k in K_LEVELS:
            top_k_rules = df.head(k)['rule'].astype(str).tolist()
            rank = first_exact_semantic_match_rank(X_df, top_k_rules, gt_rules, kind='gelpe')
            rr_results[llm][complexity][f'RR@{k}'].append(0.0 if rank is None else 1.0 / rank)

    fidelity_rows.extend(evaluate_rule_file(file_path, llm, complexity, metric, kind='gelpe'))

print('rule_counts:', json.dumps(rule_counts, indent=4))
mrr_results = {
    llm: {
        level: {f'MRR@{k}': np.mean(rr_results[llm][level][f'RR@{k}']) for k in K_LEVELS}
        for level in COMPLEXITY_LEVELS
    }
    for llm in LLM_MODELS
}
for llm in LLM_MODELS:
    mrr_results[llm]['all'] = {
        f'MRR@{k}': np.mean(sum((rr_results[llm][level][f'RR@{k}'] for level in COMPLEXITY_LEVELS), []))
        for k in K_LEVELS
    }
print('MRR:', json.dumps(mrr_results, indent=4))

pd.DataFrame.from_dict(rule_counts, orient='index').rename_axis('LLM').to_csv(
    os.path.join(evaluation_dir, 'rule_counts.csv')
)
mrr_results_df = pd.DataFrame.from_dict(
    {(llm, level): mrr_results[llm][level] for llm in mrr_results for level in mrr_results[llm]},
    orient='index',
)
mrr_results_df.index = pd.MultiIndex.from_tuples(mrr_results_df.index, names=['LLM', 'Complexity'])
mrr_results_df.to_csv(os.path.join(evaluation_dir, 'mrr_results.csv'))
rr_results_df = pd.DataFrame.from_dict(
    {
        (llm, complexity): {f'RR@{k}': rr_results[llm][complexity][f'RR@{k}'] for k in K_LEVELS}
        for llm in rr_results for complexity in rr_results[llm]
    },
    orient='index',
)
rr_results_df.index = pd.MultiIndex.from_tuples(rr_results_df.index, names=['LLM', 'Complexity'])
rr_results_df.to_csv(os.path.join(evaluation_dir, 'rr_results.csv'))

fidelity_df = pd.DataFrame(fidelity_rows)
save_summary_tables(fidelity_df, evaluation_dir)
print(f'Top-k fidelity results saved under: {evaluation_dir}')
