import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import json

evaluation_dir = f"xai_analyses_results/evaluation/shap"

# Ensure evaluation directory exists
os.makedirs(evaluation_dir, exist_ok=True)

# Define parameters
llm_models = [
	"gpt-3.5-turbo", 
	"gpt-4o-mini", 
	"gpt-4o", 
	'llama3.1', 
]
complexity_levels = ["easy", "medium", "hard"]
k_levels = [1, 3, 10]

# Define the relevant metrics for MRR computation
mrr_metrics = {
	"explanation_length_easy": ["common"],
	"explanation_length_medium": ["common", "positive", "negative"],
	"explanation_length_hard": ["common", "positive", "negative"],
	"subjectivity_score_nn_medium": ["positive", "negative"],
	"subjectivity_score_nn_hard": ["positive", "negative"],
	"gunning_fog_hard": ["interdisciplinary"],
}

def compute_mrr(shap_weights, target_features, k_levels):
	"""Computes the Mean Reciprocal Rank (MRR) for a given set of SHAP weights."""
	sorted_indices = np.argsort(shap_weights)[::-1]  # Sort in descending order
	feature_ranks = {feature: rank + 1 for rank, feature in enumerate(sorted_indices)}
	
	reciprocal_ranks = []
	for feature in target_features:
		rank = feature_ranks.get(feature, None)
		if rank:
			reciprocal_ranks.append(1 / rank)
	
	return {k: np.mean([rr for rr in reciprocal_ranks if rr <= 1/k]) for k in k_levels}

# Initialize a dictionary to store MRR results
rr_results = {llm: {level: {f"RR@{k}": [] for k in k_levels} for level in complexity_levels} for llm in llm_models}

for llm in llm_models:
	for complexity in complexity_levels:
		for metric in ["explanation_length", "subjectivity_score_nn", "gunning_fog"]:
			metric_complexity = f"{metric}_{complexity}"
			if metric_complexity not in mrr_metrics:
				continue
			
			
			# Load SHAP statistics from pickle file
			file_path = f"abstract_model_io/global_shap_stats_{llm}_{complexity}.pkl"
			with open(file_path, "rb") as f:
				metric_global_feature_stats_dict = pickle.load(f)
			
			global_feature_stats = metric_global_feature_stats_dict[metric]
			input_features = sorted(global_feature_stats.keys(), key=lambda k: global_feature_stats[k]['upper_importance_bound'], reverse=True)
			for k in k_levels:
				top_k_features = input_features[:k]
				rr_results[llm][complexity][f"RR@{k}"] += [
					1/(top_k_features.index(f)+1) if f in top_k_features else 0
					for f in mrr_metrics[metric_complexity]
				]

mrr_results = {llm: {level: {f"MRR@{k}": np.mean(rr_results[llm][level][f"RR@{k}"]) for k in k_levels} for level in complexity_levels} for llm in llm_models}
for llm in llm_models:
	mrr_results[llm]["all"] = {
		f"MRR@{k}": np.mean(sum((rr_results[llm][level][f"RR@{k}"] for level in complexity_levels), []))
		for k in k_levels
	}

print('MRR:', json.dumps(mrr_results, indent=4))

# Save MRR results to CSV
mrr_results_df = pd.DataFrame.from_dict({(llm, level): mrr_results[llm][level] for llm in mrr_results for level in mrr_results[llm]}, orient='index')
mrr_results_df.index = pd.MultiIndex.from_tuples(mrr_results_df.index, names=["LLM", "Complexity"]) # Convert the index into a proper MultiIndex
mrr_results_csv_path = os.path.join(evaluation_dir, "mrr_results.csv")
mrr_results_df.to_csv(mrr_results_csv_path, index=True)

print(f'MRR results saved to: {mrr_results_csv_path}')
