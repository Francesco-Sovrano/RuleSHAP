import os
import pandas as pd
import ast
from scipy.stats import mannwhitneyu, wilcoxon

def flatten_rr_column(df_col):
	flat_list = []
	# Process each cell in the column
	for cell in df_col:
		try:
			# Convert the string representation of the list to an actual list
			lst = ast.literal_eval(cell)
			# Extend the flat list with the list items
			flat_list.extend(lst)
		except (ValueError, SyntaxError):
			# If conversion fails, skip or handle error accordingly
			print(f"Warning: Could not parse {cell}")
	return flat_list

def load_rr_scores(folder_path):
	"""
	Loads ALL 'rr_results.csv' files from the given folder_path.
	Concatenates them into one DataFrame. If there are multiple CSVs,
	they will be appended row-wise. If no CSVs found, returns empty DataFrame.
	"""
	all_dfs = []
	for file_name in os.listdir(folder_path):
		if file_name.lower() == "rr_results.csv":
			csv_path = os.path.join(folder_path, file_name)
			print(csv_path)
			df = pd.read_csv(csv_path, header=0, index_col=[0,1])
			# df’s index has LLM, Complexity as MultiIndex
			all_dfs.append(df)

	if not all_dfs:
		print(f"No 'rr_results.csv' found in: {folder_path}")
		return pd.DataFrame()  # empty

	return pd.concat(all_dfs)

# Folders containing the rr_results.csv that we want to compare
shap_lasso_folder = "./xai_analyses_results/evaluation/shap_in_xgb=True+shap_in_lasso=True"
rulefit_folder     = "./xai_analyses_results/evaluation/rulefit"

# Load the data
shap_lasso_df = load_rr_scores(shap_lasso_folder)
rulefit_df    = load_rr_scores(rulefit_folder)

# The columns are: RR@1, RR@3, RR@10
# If you have more RR columns, list them here or read them dynamically:
rr_columns = [col for col in shap_lasso_df.columns if col.startswith("RR@")]

print("Comparing distributions between:")
print(f"  [shap_in_xgb=True+shap_in_lasso=True] vs [rulefit]")
print("Using Wilcoxon test.\n")

# For each RR column, gather all values from shap_lasso, gather all from rulefit
for rr_col in rr_columns:
	shap_lasso_values = flatten_rr_column(shap_lasso_df[rr_col])
	rulefit_values    = flatten_rr_column(rulefit_df[rr_col])

	u_statistic, p_value = wilcoxon(shap_lasso_values, rulefit_values, zero_method='zsplit', alternative='greater')

	# Compute effect size (matched pairs rank biserial correlation)
	n = len(shap_lasso_values)
	# Maximum possible sum of ranks for n pairs is n*(n+1)/2.
	# The effect size is computed as: r = (4*u_statistic)/(n*(n+1)) - 1
	effect_size = (4 * u_statistic) / (n * (n + 1)) - 1

	# Print results
	print(f"=== {rr_col} ===")
	print(f"  n_shap_lasso = {len(shap_lasso_values)}, n_rulefit = {len(rulefit_values)}")
	print(f"  U statistic = {u_statistic:.3f}, p-value = {p_value:.6f}")
	print(f"  Effect size (rank biserial correlation) = {effect_size:.3f}\n")
