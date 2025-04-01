import os
import pandas as pd
import numpy as np
import json

directory_path = f'xai_analyses_results/baseline_rules'
evaluation_dir = f"xai_analyses_results/evaluation/rulefit"

# Ensure evaluation directory exists
os.makedirs(evaluation_dir, exist_ok=True)

# Define the relevant LLM models
llm_models = [
	"gpt-3.5-turbo", 
	"gpt-4o-mini", 
	"gpt-4o", 
	'llama3.1', 
]
complexity_levels = ["easy", "medium", "hard"]
k_levels = [1,3,10]

# Define the relevant metrics for MRR computation
mrr_metrics = {
	"explanation_length_easy": [
		"common > 0.9000000059604645", 
		"common <= 0.9000000059604645"
	],
	"explanation_length_medium": [
		"common <= 0.5000000149011612 & positive > 0.5000000149011612", 
		"common <= 0.5000000149011612 & negative <= 0.7000000178813934"
	],
	"explanation_length_hard": [
		"common <= 0.5000000149011612 & positive > 0.5000000149011612", 
		"common <= 0.5000000149011612 & negative <= 0.7000000178813934"
	],
	"subjectivity_score_nn_medium": [
		"positive > 0.7000000178813934", 
		"negative <= 0.9000000059604645", 
		"positive <= 0.7000000178813934",
		"negative > 0.9000000059604645", 
	],
	"subjectivity_score_nn_hard": [
		"positive > 0.7000000178813934", 
		"negative <= 0.9000000059604645", 
		"positive <= 0.7000000178813934",
		"negative > 0.9000000059604645", 
	],
	"gunning_fog_hard": [
		"interdisciplinary <= 0.5000000149011612",
		"interdisciplinary <= 0.9000000059604645 & interdisciplinary > 0.7000000178813934",
		"interdisciplinary <= 0.7000000178813934 & interdisciplinary > 0.5000000149011612",
		"interdisciplinary > 0.9000000059604645"
	],
	#####
	## LLM as a judge
	"oversimplification_easy": [
		"common > 0.9000000059604645", 
		"common <= 0.9000000059604645"
	],
	"information_overload_easy": [
		"common > 0.9000000059604645", 
		"common <= 0.9000000059604645"
	],
	"oversimplification_medium": [
		"common <= 0.5000000149011612 & positive > 0.5000000149011612", 
		"common <= 0.5000000149011612 & negative <= 0.7000000178813934",
	],
	"information_overload_medium": [
		"common <= 0.5000000149011612 & positive > 0.5000000149011612", 
		"common <= 0.5000000149011612 & negative <= 0.7000000178813934",
		"positive > 0.7000000178813934", 
		"negative <= 0.9000000059604645", 
		"positive <= 0.7000000178813934",
		"negative > 0.9000000059604645", 
	],
	"framing_effect_medium": [
		"positive > 0.7000000178813934", 
		"negative <= 0.9000000059604645", 
		"positive <= 0.7000000178813934",
		"negative > 0.9000000059604645", 
	],
	"oversimplification_hard": [
		"common <= 0.5000000149011612 & positive > 0.5000000149011612", 
		"common <= 0.5000000149011612 & negative <= 0.7000000178813934",
	],
	"information_overload_hard": [
		"interdisciplinary <= 0.5000000149011612",
		"interdisciplinary <= 0.9000000059604645 & interdisciplinary > 0.7000000178813934",
		"interdisciplinary <= 0.7000000178813934 & interdisciplinary > 0.5000000149011612",
		"interdisciplinary > 0.9000000059604645",
		"common <= 0.5000000149011612 & positive > 0.5000000149011612", 
		"common <= 0.5000000149011612 & negative <= 0.7000000178813934",
		"positive > 0.7000000178813934", 
		"negative <= 0.9000000059604645", 
		"positive <= 0.7000000178813934",
		"negative > 0.9000000059604645", 
	],
	"framing_effect_hard": [
		"positive > 0.7000000178813934", 
		"negative <= 0.9000000059604645", 
		"positive <= 0.7000000178813934",
		"negative > 0.9000000059604645", 
	],
}


# Initialize a dictionary to store rules counts
rule_counts = {llm: {level: 0 for level in complexity_levels} for llm in llm_models}

# Initialize a dictionary to store MRR results
rr_results = {llm: {level: {f"RR@{k}": [] for k in k_levels} for level in complexity_levels} for llm in llm_models}

# Compute rule counts and MRRs
for file_name in os.listdir(directory_path):
	if not file_name.endswith('.csv'):  # Check if the file is a CSV
		continue
	if not file_name.startswith('rulefit_rules_'):
		continue
	# Parse file details
	parts = file_name.split("_")
	if len(parts) < 5:
		continue

	llm = parts[2]
	complexity = parts[3]
	metric = "_".join(parts[4:]).replace(".csv", "")

	# Only process relevant LLMs and complexity levels
	if llm not in llm_models or complexity not in complexity_levels:
		continue

	# Load CSV
	file_path = os.path.join(directory_path, file_name)
	df = pd.read_csv(file_path)

	# Compute Rule-based Conciseness: Count rules
	rule_counts[llm][complexity] += len(df)
	if "total" not in rule_counts[llm]:
		rule_counts[llm]["total"] = 0
	rule_counts[llm]["total"] += len(df)

	metric_complexity = f"{metric}_{complexity}"

	# Compute MRR only for relevant conditions
	if metric_complexity in mrr_metrics:
		# Extract rank positions based on importance (assuming higher importance = better ranking)
		df = df.sort_values(by="importance", ascending=False).reset_index(drop=True)
		for k in k_levels:
			top_k_rules = df.head(k)["rule"].tolist()
			j=0
			found=False
			while not found and j < len(top_k_rules):
				rule = top_k_rules[j]
				rule = ' & '.join(sorted(map(lambda x: x.strip(), rule.split('&'))))
				found = rule in mrr_metrics[metric_complexity]
				# found = any((
				# 	all(map(lambda x: x.strip() in rule, gt_rule.split('&')))
				# 	for gt_rule in mrr_metrics[metric_complexity]
				# ))
				j+=1
				# if found:
				# 	print(rule, j, llm, complexity, metric)
			reciprocal_rank = 1/j if found else 0
			rr_results[llm][complexity][f"RR@{k}"].append(reciprocal_rank)

print('rule_counts:', json.dumps(rule_counts, indent=4))

# print('RR:', json.dumps(rr_results, indent=4))

mrr_results = {llm: {level: {f"MRR@{k}": np.mean(rr_results[llm][level][f"RR@{k}"]) for k in k_levels} for level in complexity_levels} for llm in llm_models}
for llm in llm_models:
	mrr_results[llm]["all"] = {
		f"MRR@{k}": np.mean(sum((rr_results[llm][level][f"RR@{k}"] for level in complexity_levels), []))
		for k in k_levels
	}

print('MRR:', json.dumps(mrr_results, indent=4))

# Save rule counts to CSV
rule_counts_df = pd.DataFrame.from_dict(rule_counts, orient='index')
rule_counts_df.index.name = "LLM"
rule_counts_csv_path = os.path.join(evaluation_dir, "rule_counts.csv")
rule_counts_df.to_csv(rule_counts_csv_path, index=True)

# Save MRR results to CSV
mrr_results_df = pd.DataFrame.from_dict({(llm, level): mrr_results[llm][level] for llm in mrr_results for level in mrr_results[llm]}, orient='index')
mrr_results_df.index = pd.MultiIndex.from_tuples(mrr_results_df.index, names=["LLM", "Complexity"]) # Convert the index into a proper MultiIndex
mrr_results_csv_path = os.path.join(evaluation_dir, "mrr_results.csv")
mrr_results_df.to_csv(mrr_results_csv_path, index=True)

# --- Save raw RR results to CSV ---
rr_results_df = pd.DataFrame.from_dict(
	{(llm, complexity): {f"RR@{k}": rr_results[llm][complexity][f"RR@{k}"] for k in k_levels}
	 for llm in rr_results for complexity in rr_results[llm]}, orient='index'
)
rr_results_df.index = pd.MultiIndex.from_tuples(rr_results_df.index, names=["LLM", "Complexity"])
rr_results_csv_path = os.path.join(evaluation_dir, "rr_results.csv")
rr_results_df.to_csv(rr_results_csv_path, index=True)

print(f'Rule counts saved to: {rule_counts_csv_path}')
print(f'MRR results saved to: {mrr_results_csv_path}')
print(f'RR results saved to: {rr_results_csv_path}')