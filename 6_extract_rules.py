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

from ruleshap import RuleSHAP
# from rulefit import RuleFit

from lib import create_cache, load_cache

################################################################

import argparse
parser = argparse.ArgumentParser(description="Provide model, SHAP boolean options, and random seed")

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

# Boolean inputs
parser.add_argument(
	"--use_shap_in_xgb",
	action="store_true",
	help="Set this flag to use SHAP in XGB"
)
parser.add_argument(
	"--use_shap_in_lasso",
	action="store_true",
	help="Set this flag to use SHAP in Lasso"
)

# Integer input for random seed
parser.add_argument(
	"--random_seed",
	type=int,
	default=42,
	help="Specify the random seed (integer)"
)

args = parser.parse_args()
model, difficulty, use_shap_in_xgb, use_shap_in_lasso, random_seed = args.model, args.difficulty, args.use_shap_in_xgb, args.use_shap_in_lasso, args.random_seed
np.random.seed(random_seed)

print('6_extract_rules', args)

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
	# 'subjectivity_score',
	'subjectivity_score_nn',
	'gunning_fog',
	# 'sentiment_score',
	'sentiment_score_nn',
	# 'dox_score',
	# ##############################
	# ### Other readability scores
	# 'flesch_score',
	# 'smog_index',
	# 'coleman_liau_index',
	# ##############################
	### LLM-as-a-judge
	'framing_effect',
	'information_overload',
	'oversimplification',
]

rule_output_dir = f'xai_analyses_results/rules/shap_in_xgb={use_shap_in_xgb}+shap_in_lasso={use_shap_in_lasso}'
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

metric_global_feature_stats_dict = load_cache(os.path.join(csv_file_dir, f'global_shap_stats_{model}_{difficulty}.pkl'))
for metric in metrics_list:
	X_and_y = merged_df[input_features+[metric]].dropna().values.astype(np.float32)
	y = X_and_y[:, -1]
	X = X_and_y[:, :-1]
	
	global_feature_stats = metric_global_feature_stats_dict[metric]
	shap_weights = np.array([global_feature_stats[k]['upper_importance_bound'] for k in input_features])

	rf_model = RuleSHAP(
		gboost_config_dict = { # Details about parameters: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor
			'n_estimators': 300, # Number of trees in the ensemble. Fewer trees mean fewer rules. Use a lower number (e.g., 50–200).
			'max_depth': 5, # Limits the maximum depth of a tree. Restricts the number of splits, leading to simpler trees. Start with small values (e.g., 2–4).
			'subsample': 0.8, # Fraction of training instances used to build each tree. Smaller values introduce randomness, reducing overfitting and simplifying models. Use values around 0.5–0.8.
			# 'sampling_method': 'uniform', # Each training instance has an equal probability of being selected. Typically set subsample >= 0.5 for good results.
			'tree_method': 'exact', # The tree construction algorithm used in XGBoost. See description in https://xgboost.readthedocs.io/en/stable/treemethod.html
			# 'max_leaves': 20, # Maximum number of terminal nodes (leaves)
			'min_child_weight': 4, # Minimum sum of weights required in a child node. Larger values make it harder for the model to create splits, effectively limiting the number of nodes in a tree. Try higher values (e.g., 5–10).
			# 'alpha': 1, # L1 regularization term. Regularization terms penalize more complex models. Encourage simpler models by penalizing complex trees.
			'learning_rate': 0.01, # Step size shrinkage to prevent overfitting. Slower learning can prevent overly complex rules. Use moderately low values (e.g., 0.1–0.3).
			# 'objective': 'reg:pseudohubererror',
			# 'gamma': 0, # Allow splits with minimal gain
		},
		random_state=random_seed, # For reproducibility
		rfmode='regress', # 'regress' for regression or 'classify' for binary classification.
		# max_rules=4000,
		# tree_size=10,
	)
	
	rf_model.fit(X, y, 
		feature_names=input_features, 
		shap_weights=shap_weights,
		use_shap_in_xgb=use_shap_in_xgb, 
		use_shap_in_lasso=use_shap_in_lasso,
	)
	rules = rf_model.get_rules()  # Extracts rules from the model
	rules = rules.sort_values("importance", ascending=False)  # Only keep rules with non-zero coefficients

	# Save the rules to a file for inspection
	rules.to_csv(os.path.join(rule_output_dir, f'association_rules_{model}_{difficulty}_{metric}.csv'), index=False)
