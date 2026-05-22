import os
import json
import numpy as np
import pandas as pd

from xai_eval_utils import (
    LLM_MODELS,
    COMPLEXITY_LEVELS,
    K_LEVELS,
    evaluate_shap_feature_fidelity,
    save_summary_tables,
)

evaluation_dir = 'xai_analyses_results/evaluation/shap'
os.makedirs(evaluation_dir, exist_ok=True)

mrr_metrics = {
    'explanation_length_easy': ['common'],
    'explanation_length_medium': ['common', 'positive', 'negative'],
    'explanation_length_hard': ['common', 'positive', 'negative'],
    'subjectivity_score_nn_medium': ['positive', 'negative'],
    'subjectivity_score_nn_hard': ['positive', 'negative'],
    'gunning_fog_hard': ['interdisciplinary'],
    'oversimplification_easy': ['common'],
    'information_overload_easy': ['common'],
    'oversimplification_medium': ['common', 'positive', 'negative'],
    'information_overload_medium': ['common', 'positive', 'negative'],
    'framing_effect_medium': ['positive', 'negative'],
    'oversimplification_hard': ['common', 'positive', 'negative'],
    'information_overload_hard': ['interdisciplinary', 'common', 'positive', 'negative'],
    'framing_effect_hard': ['positive', 'negative'],
}

rr_results = {llm: {level: {f'RR@{k}': [] for k in K_LEVELS} for level in COMPLEXITY_LEVELS} for llm in LLM_MODELS}
fidelity_rows = []

for llm in LLM_MODELS:
    for complexity in COMPLEXITY_LEVELS:
        for metric in ['explanation_length', 'subjectivity_score_nn', 'gunning_fog']:
            metric_complexity = f'{metric}_{complexity}'
            if metric_complexity not in mrr_metrics:
                continue
            file_path = f'abstract_model_io/global_shap_stats_{llm}_{complexity}.pkl'
            if not os.path.isfile(file_path):
                continue
            import pickle
            with open(file_path, 'rb') as f:
                metric_global_feature_stats_dict = pickle.load(f)
            global_feature_stats = metric_global_feature_stats_dict[metric]
            input_features = sorted(global_feature_stats.keys(), key=lambda k: global_feature_stats[k]['upper_importance_bound'], reverse=True)
            for k in K_LEVELS:
                top_k_features = input_features[:k]
                rr_results[llm][complexity][f'RR@{k}'] += [1 / (top_k_features.index(f) + 1) if f in top_k_features else 0 for f in mrr_metrics[metric_complexity]]
            fidelity_rows.extend(evaluate_shap_feature_fidelity(llm, complexity, metric))

mrr_results = {llm: {level: {f'MRR@{k}': np.mean(rr_results[llm][level][f'RR@{k}']) for k in K_LEVELS} for level in COMPLEXITY_LEVELS} for llm in LLM_MODELS}
for llm in LLM_MODELS:
    mrr_results[llm]['all'] = {f'MRR@{k}': np.mean(sum((rr_results[llm][level][f'RR@{k}'] for level in COMPLEXITY_LEVELS), [])) for k in K_LEVELS}
print('MRR:', json.dumps(mrr_results, indent=4))

mrr_results_df = pd.DataFrame.from_dict({(llm, level): mrr_results[llm][level] for llm in mrr_results for level in mrr_results[llm]}, orient='index')
mrr_results_df.index = pd.MultiIndex.from_tuples(mrr_results_df.index, names=['LLM', 'Complexity'])
mrr_results_df.to_csv(os.path.join(evaluation_dir, 'mrr_results.csv'))

fidelity_df = pd.DataFrame(fidelity_rows)
save_summary_tables(fidelity_df, evaluation_dir, prefix='feature_')
print(f'Top-k feature fidelity results saved under: {evaluation_dir}')
