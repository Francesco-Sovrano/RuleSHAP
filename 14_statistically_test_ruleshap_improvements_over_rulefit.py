import ast
import os
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


def flatten_rr_column(df_col):
	flat_list = []
	# Process each cell in the column.
	for cell in df_col:
		try:
			# Convert the string representation of the list to an actual list.
			lst = ast.literal_eval(cell)
			flat_list.extend(lst)
		except (ValueError, SyntaxError):
			print(f"Warning: Could not parse {cell}")
	return flat_list


def load_rr_scores(folder_path):
	"""
	Loads ALL 'rr_results.csv' files from the given folder_path.
	Concatenates them into one DataFrame. If there are multiple CSVs,
	they will be appended row-wise. If no CSVs are found, returns an
	empty DataFrame.
	"""
	all_dfs = []
	for file_name in os.listdir(folder_path):
		if file_name.lower() == "rr_results.csv":
			csv_path = os.path.join(folder_path, file_name)
			print(csv_path)
			df = pd.read_csv(csv_path, header=0, index_col=[0, 1])
			# df's index has LLM, Complexity as MultiIndex.
			all_dfs.append(df)

	if not all_dfs:
		print(f"No 'rr_results.csv' found in: {folder_path}")
		return pd.DataFrame()

	return pd.concat(all_dfs)


def holm_bonferroni(p_values):
	"""
	Return Holm-Bonferroni adjusted p-values in the original order.

	Holm's step-down procedure sorts raw p-values increasingly, multiplies
	each by the number of remaining hypotheses, and enforces monotonicity of
	the adjusted p-values over the sorted order.
	"""
	m = len(p_values)
	indexed = sorted(enumerate(p_values), key=lambda item: item[1])
	adjusted = [None] * m
	running_max = 0.0

	for rank, (original_idx, p_value) in enumerate(indexed, start=1):
		multiplier = m - rank + 1
		adj_p = min(1.0, multiplier * p_value)
		running_max = max(running_max, adj_p)
		adjusted[original_idx] = running_max

	return adjusted


# Folders containing the rr_results.csv files that we want to compare.
shap_lasso_folder = "./xai_analyses_results/evaluation/shap_in_xgb=True+shap_in_lasso=True"
rulefit_folder = "./xai_analyses_results/evaluation/rulefit"

# Load the data.
shap_lasso_df = load_rr_scores(shap_lasso_folder)
rulefit_df = load_rr_scores(rulefit_folder)

# The columns are: RR@1, RR@3, RR@10.
# If there are more RR columns, include them automatically.
rr_columns = [col for col in shap_lasso_df.columns if col.startswith("RR@")]

print("Comparing distributions between:")
print("  [shap_in_xgb=True+shap_in_lasso=True] vs [rulefit]")
print("Using one-sided paired Wilcoxon signed-rank tests")
print("with Holm-Bonferroni correction across RR columns.\n")

results = []

# For each RR column, gather all values from RuleSHAP and RuleFit.
for rr_col in rr_columns:
	shap_lasso_values = flatten_rr_column(shap_lasso_df[rr_col])
	rulefit_values = flatten_rr_column(rulefit_df[rr_col])

	if len(shap_lasso_values) != len(rulefit_values):
		raise ValueError(
			f"Mismatched paired sample sizes for {rr_col}: "
			f"RuleSHAP={len(shap_lasso_values)}, RuleFit={len(rulefit_values)}"
		)

	statistic, raw_p_value = wilcoxon(
		shap_lasso_values,
		rulefit_values,
		zero_method="zsplit",
		alternative="greater",
	)

	n = len(shap_lasso_values)
	# Matched-pairs rank-biserial correlation.
	# For n pairs, the maximum possible positive-rank sum is n*(n+1)/2.
	effect_size = (4 * statistic) / (n * (n + 1)) - 1

	results.append(
		{
			"metric": rr_col,
			"n_pairs": n,
			"wilcoxon_statistic": statistic,
			"raw_p_value": raw_p_value,
			"holm_bonferroni_p_value": None,  # filled below
			"rank_biserial_correlation": effect_size,
		}
	)

adjusted_p_values = holm_bonferroni([row["raw_p_value"] for row in results])
for row, adjusted_p_value in zip(results, adjusted_p_values):
	row["holm_bonferroni_p_value"] = adjusted_p_value

# Print results after correction so that raw and adjusted values are side by side.
for row in results:
	print(f"=== {row['metric']} ===")
	print(f"  n_pairs = {row['n_pairs']}")
	print(f"  Wilcoxon statistic = {row['wilcoxon_statistic']:.3f}")
	print(f"  raw p-value = {row['raw_p_value']:.6f}")
	print(f"  Holm-Bonferroni adjusted p-value = {row['holm_bonferroni_p_value']:.6f}")
	print(
		"  Effect size (rank-biserial correlation) = "
		f"{row['rank_biserial_correlation']:.3f}\n"
	)

# Save a machine-readable summary for paper reporting and reproducibility.
output_path = Path("./xai_analyses_results/evaluation/ruleshap_vs_rulefit_wilcoxon_holm.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(output_path, index=False)
print(f"Saved corrected statistical test summary to: {output_path}")
