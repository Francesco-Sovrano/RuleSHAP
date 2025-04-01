import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Force single-threaded usage in BLAS/OpenBLAS/MKL/NumExpr
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np

from scipy.stats import pearsonr, spearmanr

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

args = parser.parse_args()
model, difficulty = args.model, args.difficulty

print('12_input_output_correlation_analysis', args)

################################################################

csv_file_dir = 'correlation_analysis/'
os.makedirs(csv_file_dir, exist_ok=True)

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

proxy_metrics_list = [
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
]

llm_as_a_judge_list = [
	### LLM-as-a-judge
	'framing_effect',
	'information_overload',
	'oversimplification',
]

################################################################

merged_df = pd.read_csv(os.path.join('abstract_model_io/', f'topic_{model}_{difficulty}.csv'))

# SHAP analysis for each metric
correlation_analysis_stats = []
for metric in proxy_metrics_list+llm_as_a_judge_list:
	for feature in input_features:
		X_and_y = merged_df[[feature,metric]].dropna().values.astype(np.float32)
		Y = X_and_y[:, -1]
		X = X_and_y[:, :-1]
		corr, p_value = spearmanr(X, Y)
		effect_size = corr ** 2  # Using r^2 as an effect size measure
		correlation_analysis_stats.append({
			'metric': metric,
			'feature': feature,
			'correlation': corr,
			'p_value': p_value,
			'effect_size': effect_size, # This value can be interpreted as the proportion of variance in the metric that is explained by the feature
		})

df = pd.DataFrame(correlation_analysis_stats)
df = df.sort_values(by="effect_size", ascending=False).reset_index(drop=True)
# Define the filename for the CSV file
csv_filename = os.path.join(csv_file_dir, f'analysis_{model}_{difficulty}.csv')
# Save the DataFrame to a CSV file
df.to_csv(csv_filename, index=False)
