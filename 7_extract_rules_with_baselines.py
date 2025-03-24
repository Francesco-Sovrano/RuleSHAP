import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Force single-threaded usage in BLAS/OpenBLAS/MKL/NumExpr
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# import re
import pandas as pd
import numpy as np
import json

from rulefit import RuleFit
# from imodels import SkopeRulesClassifier, BayesianRuleSetClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, _tree

from lib import create_cache, load_cache

################################################################

import argparse
parser = argparse.ArgumentParser(description="Provide model and random seed")

# Model string input
parser.add_argument(
	"--model",
	type=str,
	required=True,
	help="Specify the model name (e.g., 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'llama3.1', 'llama3.1:70b')"
)

# Model string input
parser.add_argument(
	"--difficulty",
	choices=[
		'hard',
		'medium', 
		'easy', 
		'baseline', 
	],
	required=True,
	help="Choose the rule detection difficulty: baseline, easy, medium, hard"
)

# Integer input for random seed
parser.add_argument(
	"--random_seed",
	type=int,
	default=42,
	help="Specify the random seed (integer)"
)

args = parser.parse_args()
model, difficulty, random_seed = args.model, args.difficulty, args.random_seed
np.random.seed(random_seed)

print('7_extract_rules_with_baselines', args)

################################################################

csv_file_dir = 'abstract_model_io/'

input_features = [
	'conceptually dense',
	'technically complicated',
	'common',
	'socially controversial',
	'unambiguous',
	# 'open to interpretation',
	'positive',
	'negative',
	'neutral',
	'subject to geographical variability',
	'interdisciplinary',
	'subject to time variability'
]

metrics_list = [
	'explanation_length',
	'subjectivity_score',
	'subjectivity_score_nn',
	'gunning_fog',
	'sentiment_score',
	'sentiment_score_nn',
	# 'dox_score',
	##############################
	### Other readability scores
	'flesch_score',
	'smog_index',
	'coleman_liau_index',
]

rule_output_dir = f'xai_analyses_results/baseline_rules/'
os.makedirs(rule_output_dir, exist_ok=True)

################################################################

# def denormalize_rule(rule_str, feature_weights):
# 	# Regex to match conditions like: feature_name <= 0.25, feature_name > -1.5, etc.
# 	# This pattern assumes feature names are word characters and conditions are space-separated.
# 	pattern = r"([^&]+)\s+(<=|>=|<|>|=)\s+(-?\d*\.?\d*)"

# 	def replace_threshold(match_tuple):
# 		feature, op, value = match_tuple
# 		# print(feature)
# 		# Multiply the threshold by the corresponding alpha_weight to revert to original scale
# 		original_value = float(value) * feature_weights[feature.strip()]
# 		return f"{feature} {op} {original_value}"

# 	# Find all conditions in the rule_str
# 	conditions = re.findall(pattern, rule_str)
# 	if not conditions:
# 		return rule_str
# 	return ' & '.join(map(replace_threshold, conditions))

################################################################

merged_df = pd.read_csv(os.path.join(csv_file_dir, f'topic_{model}_{difficulty}.csv'))
X = merged_df[input_features].dropna().values.astype(np.float32)


for metric in metrics_list:
	y = merged_df[[metric]].dropna().values.astype(np.float32)

	# ----------------------- Decision Tree Regressor ---------------------------
	# Initialize the model with desired parameters
	tree_model = DecisionTreeRegressor(random_state=random_seed, max_depth=5)  # adjust max_depth as needed
	# Fit the model
	tree_model.fit(X, y)
	# Function to extract decision rules
	def extract_rules(tree, feature_names):
		tree_ = tree.tree_
		feature_name = [
			feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
			for i in tree_.feature
		]
		paths = []
		def recurse(node, path, paths):
			if tree_.feature[node] != _tree.TREE_UNDEFINED:
				name = feature_name[node]
				threshold = tree_.threshold[node]
				# Left child
				recurse(tree_.children_left[node],
						path + [f"({name} <= {threshold:.3f})"], paths)
				# Right child
				recurse(tree_.children_right[node],
						path + [f"({name} > {threshold:.3f})"], paths)
			else:
				value = tree_.value[node]
				path_str = " and ".join(path)
				paths.append(f"If {path_str} then response: {value[0][0]:.3f}")
		recurse(0, [], paths)
		return paths
	# Extract rules
	rules = extract_rules(tree_model, input_features)
	# Create a DataFrame to store the rules
	rules_df = pd.DataFrame(rules, columns=["rule"])
	rules_df.to_csv(os.path.join(rule_output_dir, f'dtree_rules_{model}_{difficulty}_{metric}.csv'))

	# ----------------------- Linear Regression ---------------------------
	linreg_model = LinearRegression()
	linreg_model.fit(X, y)
	# Extract coefficients
	coefs = linreg_model.coef_[0]
	intercept = linreg_model.intercept_[0]
	# Create a DataFrame to mimic the "rules" output
	# Note: With a plain linear model, each feature has exactly one coefficient,
	# so we can treat the "rule" as simply the feature name.
	coefs_df = pd.DataFrame({
		"rule": input_features,
		"coef": coefs,
		# "intercept": intercept,
	})
	# If you only want non-zero coefficients:
	coefs_df = coefs_df[coefs_df.coef != 0]
	# Sort by absolute value of coefficients in descending order
	coefs_df["abs_coef"] = coefs_df["coef"].abs()
	coefs_df = coefs_df.sort_values(by="abs_coef", ascending=False).drop(columns=["abs_coef"])
	# Save the rules to a CSV file
	coefs_df.to_csv(os.path.join(rule_output_dir, f'linear_rules_{model}_{difficulty}_{metric}.csv'))

	# ----------------------- RuleFit ---------------------------
	rulefit_model = RuleFit(random_state=random_seed)
	rulefit_model.fit(X, y, feature_names=input_features)
	rulefit_rules = rulefit_model.get_rules()
	rulefit_rules = rulefit_rules[rulefit_rules.coef != 0]  # Only keep rules with non-zero coefficients
	rulefit_rules.to_csv(os.path.join(rule_output_dir, f'rulefit_rules_{model}_{difficulty}_{metric}.csv'))

	### The methods below do not work with any ordinal target, only with discrete or categorical targets, which is not our case
	# # ----------------------- SkopeRules ------------------------
	# skope_model = SkopeRulesClassifier(random_state=random_seed)
	# skope_model.fit(X, y>np.median(y), feature_names=input_features)
	# skope_rules = skope_model.rules_  # Extracts rules from the model
	# skope_rules_df = pd.DataFrame({
	# 	'rule': skope_rules,
	# })
	# skope_rules_df.to_csv(os.path.join(rule_output_dir, f'skope_rules_{model}_{difficulty}_{metric}.csv'))

	# # ----------------------- Bayesian Rule Set ----------------
	# brs_model = BayesianRuleSetClassifier()
	# brs_model.fit(X, y>np.median(y), feature_names=input_features)
	# brs_rules = brs_model.rules()
	# brs_rules_df = pd.DataFrame({
	# 	'rule': brs_rules,
	# })
	# brs_rules_df.to_csv(os.path.join(rule_output_dir, f'brs_rules_{model}_{difficulty}_{metric}.csv'))
