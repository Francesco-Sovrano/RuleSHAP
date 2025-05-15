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

print('13_llm_estimate_proxy_metrics_correlation_analysis')

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

llm_models = [
	"gpt-3.5-turbo", 
	"gpt-4o-mini", 
	"gpt-4o", 
	'llama3.1', 
	'llama3.1:70b'
]
complexity_levels = ["easy", "medium", "hard"]

# Create a list to store all the X_and_y arrays.
all_df = []
for difficulty in complexity_levels:
	for model in llm_models:
		all_df.append(pd.read_csv(os.path.join('abstract_model_io/', f'topic_{model}_{difficulty}.csv')))
# Concatenate all arrays vertically.
merged_df = pd.concat(all_df, ignore_index=True)

correlation_analysis_stats = []
for metric in llm_as_a_judge_list:
	for feature in proxy_metrics_list:
		X_and_y = merged_df[[feature,metric]].dropna().values.astype(np.float32)
		Y = X_and_y[:, -1]
		X = X_and_y[:, :-1]
		corr, p_value = spearmanr(X, Y)
		effect_size = corr ** 2  # Using r^2 as an effect size measure
		correlation_analysis_stats.append({
			'llm_as_a_judge': metric,
			'proxy_metric': feature,
			'correlation': corr,
			'p_value': p_value,
			'effect_size': effect_size, # This value can be interpreted as the proportion of variance in the metric that is explained by the feature
		})

df = pd.DataFrame(correlation_analysis_stats)
df = df.sort_values(by="effect_size", ascending=False).reset_index(drop=True)
# Define the filename for the CSV file
csv_filename = os.path.join(csv_file_dir, f'analysis_llm_as_a_judge.csv')
# Save the DataFrame to a CSV file
df.to_csv(csv_filename, index=False)
