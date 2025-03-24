import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Force single-threaded usage in BLAS/OpenBLAS/MKL/NumExpr
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from numba import njit, types
from numba.typed import Dict, List

from scipy.stats import pearsonr, spearmanr

# from sklearn.cluster import KMeans
# from sklearn_extra.cluster import KMedoids
# from mlxtend.frequent_patterns import apriori, association_rules
import shap
from ruleshap import rand_int

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

# Integer input for random seed
parser.add_argument(
	"--fast_shap_estimate",
	action="store_true",
	help="Set this flag to compute Shapley values in a faster, approximated way when len(input_features) < 20"
)

args = parser.parse_args()
model, difficulty, random_seed, fast_shap_estimate = args.model, args.difficulty, args.random_seed, args.fast_shap_estimate
np.random.seed(random_seed)

print('5_compute_shap_values', args)

################################################################

csv_file_dir = 'abstract_model_io/'

minimum_score = 1
maximum_score = 5
min_value = minimum_score/maximum_score

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
]

# Create the plot directories if they don't exist
summary_plot_dir = f'xai_analyses_results/summary_plot'
os.makedirs(summary_plot_dir, exist_ok=True)
# dependence_plot_dir = 'xai_analyses_results/dependence_plot/'
# os.makedirs(dependence_plot_dir, exist_ok=True)

################################################################

def get_global_feature_stats_from_shap_values(shap_values, features, target):
	'''
	Prints the feature importances based on SHAP values in an ordered way
	shap_values -> The SHAP values calculated from a shap.Explainer object
	features -> The name of the features, on the order presented to the explainer
	target -> the target value, i.e., the model output metric
	'''
	# Convert SHAP values and target into a DataFrame for easier manipulation
	shap_df = pd.DataFrame(shap_values, columns=features)
	shap_df['target'] = target
	# Compute correlation between each feature's SHAP values and the target variable
	correlation_with_target = {}
	for col in features:
		corr, p_value = spearmanr(shap_df[col], shap_df["target"])
		correlation_with_target[col] = corr
	# Calculates the feature importance (mean absolute shap value) for each feature
	feature_details = {
		features[i]: {
			'max': np.max(abs_shap_values_i),
			'min': np.min(abs_shap_values_i),
			'mean': np.mean(abs_shap_values_i),
			'std': np.std(abs_shap_values_i),
			'median': np.median(abs_shap_values_i),
			'percentile_75th': np.percentile(abs_shap_values_i, 75),
			'spearman_correlation': correlation_with_target[features[i]],
			'upper_importance_bound': np.mean(abs_shap_values_i)+np.std(abs_shap_values_i),
		}
		for i, abs_shap_values_i in map(lambda x: (x, np.abs(shap_values[:, x])), range(shap_values.shape[1]))
	}
	# Organize the importances and columns in a dictionary
	feature_details = dict(sorted(feature_details.items(), key=lambda item: item[1]['upper_importance_bound'], reverse=True))
	return feature_details

################################################################

merged_df = pd.read_csv(os.path.join(csv_file_dir, f'topic_{model}_{difficulty}.csv'))
X = merged_df[input_features].dropna().values.astype(np.float32)

####################
### zero_background
# _background = np.full((1, X.shape[1]), 0., dtype=np.float32)
####################
### min_background
_background = np.min(X, axis=0).reshape(1, -1) - min_value/10 # slightly less than the minimum
print('min_background:', _background)
####################
### median_background
# _background = np.median(X, axis=0).reshape(1, -1)
# print('median_background:', _background)
####################
### random_sample_background
# sample_size = 100  # Choose an appropriate size based on your dataset
# _background = X[np.random.choice(X.shape[0], sample_size, replace=False), :]
####################
### kmedoids_background
# # kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
# kmedoids = KMedoids(n_clusters=len(input_features), metric='cityblock', method='pam', init='k-medoids++', max_iter=300, random_state=0).fit(X) # https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
# _background = kmedoids.cluster_centers_

# ------------------------------------------------------------------------
# Precompute indices for each column in X that have the minimum
#         absolute value. This saves computation time later.
# ------------------------------------------------------------------------
# Create a Numba typed dictionary
precomputed_rows_with_min = Dict.empty( # Will store, for each column, which row(s) in X minimize abs(X[col])
	key_type=types.int64,  # String keys
	value_type=types.Array(types.int64, 1, "C")  # 1D NumPy arrays of integers as values
)
for col in range(X.shape[1]):
	col_X = X[:, col]
	# Find the minimum absolute value in this column
	col_min = np.min(col_X)
	# Find all row indices where this minimum occurs
	rows_with_min = np.where(col_X == col_min)[0]
	# Append this list of rows (indices) to our precomputed list
	precomputed_rows_with_min[col] = rows_with_min

@njit(fastmath=True)
def abstracted_model(x, X, y, background, precomputed_rows_with_min, seed):

	# # If the model is fed with a pre-computed output in the last column (i.e., x has one extra column),
	# # we simply return that last column as the result. This is often a quick bypass if x already
	# # contains the desired output.
	# if x.shape[-1] == X.shape[-1] + 1:
	# 	return x[:, -1]

	# Number of samples in the input array 'x'
	n_samples = x.shape[0]

	# Allocate an array to hold the global indices in X of the chosen rows
	selected_indices = np.empty(n_samples, dtype=np.int64)

	# ------------------------------------------------------------------------
	# STEP 1: For each sample in x, determine which row in X provides
	#         the best match based on the logic described below.
	# ------------------------------------------------------------------------
	
	for i in range(n_samples):
		# We'll assume we might use all rows of X unless we find a smaller subset
		use_all_Xs = True
		
		# Find feature indices in x[i] that match the "background trigger" value, background[0].
		# This tells us which features (columns) to consider for the "minimum absolute value" rule.
		feature_indices = np.where(x[i] == background[0])[0]

		# If there are any features that match the background value:
		if len(feature_indices) > 0:
			row_counts = np.zeros(X.shape[0], dtype=np.int64)
			# Gather all rows that minimize absolute value for each of those feature indices
			for f in feature_indices:
				for row in precomputed_rows_with_min[f]:
					row_counts[row] += 1
			
			# Filter rows where count equals len(feature_indices)
			valid_rows = np.where(row_counts == len(feature_indices))[0]
			# If we found any valid rows
			if len(valid_rows) > 0:
				X_candidates = X[valid_rows]
				# Since we do have valid candidates, we won't use all rows in X
				use_all_Xs = False

		# If no valid candidates were found (or no matching features):
		# we simply use all rows in X.
		if use_all_Xs:
			X_candidates = X

		# --------------------------------------------------------------------
		# STEP 2: Compute a distance metric between the current sample x[i]
		#         and each candidate row (or all rows, if no candidates).
		#         Here we use the Euclidean distance by default.
		# --------------------------------------------------------------------
		distances = np.empty(X_candidates.shape[0], dtype=np.float32)
		for j in range(X_candidates.shape[0]):
			# Euclidean distance; no need to compute squared root of the sum of squares since we're only looking for the minimum
			distances[j] = np.sum((x[i] - X_candidates[j]) ** 2, dtype=np.float32)

			# -- Alternative distance examples (commented out) --
			# Hamming distance:
			# distances[j] = np.sum(x[i] != X_candidates[j])
			#
			# Manhattan distance:
			# distances[j] = np.sum(np.abs(x[i] - X_candidates[j]))

		# Identify the minimum distance value and find all candidate rows that achieve it
		min_distance = np.min(distances)
		closest_indices = np.where(distances == min_distance)[0] # Indices of closest candidates

		# --------------------------------------------------------------------
		# STEP 3: Randomly select one among the rows with that minimum distance.
		#         We'll use a helper function rand_int(...) that presumably
		#         returns an integer in the specified range, seeded for reproducibility.
		# --------------------------------------------------------------------
		selected_index = closest_indices[rand_int(len(closest_indices), seed+n_samples+i)]

		# Map back to the global index in `X`
		selected_indices[i] = valid_rows[selected_index] if not use_all_Xs else selected_index
		########################################
		# Alternative selection: select a random candidate (commented out)
		# selected_indices[i] = rand_int(len(X_candidates), seed+i+n_samples)
	# Return the corresponding `y` values for the selected indices
	return y[selected_indices]


# SHAP analysis for each metric
metric_global_feature_stats_dict = {}
for metric in metrics_list:
	# X are the score_type columns (features), y is the metric value
	X_with_y = merged_df[input_features+[metric]].dropna().values.astype(np.float32)
	# X_with_y = merged_df[input_features+[metric]].fillna(1).values
	y = X_with_y[:, -1]
	X = X_with_y[:, :-1]

	# print(abstracted_model(X[:10], X, _background, y, precomputed_rows_with_min, random_seed))

	print(f"SHAP summary plot for metric {metric} and model {model} {difficulty}")

	# SHAP Explainer
	# explainer = shap.KernelExplainer(lambda x: abstracted_model(x, X, y, _background, precomputed_rows_with_min, random_seed), _background, seed=random_seed)
	# shap_values = explainer.shap_values(X_with_y, l1_reg=False) # The number of features considered is small. No need for L1 regularization
	# explainer = shap.PermutationExplainer(lambda x: abstracted_model(x, X, y, _background, precomputed_rows_with_min, random_seed), _background, seed=random_seed) # PermutationExplainer: Suitable for models where an efficient approximation of Shapley values is acceptable, and when feature independence is a reasonable assumption.
	# shap_values = explainer.shap_values(X_with_y[:, :-1])

	# build a clustering of the features based on shared information about y
	clustering = shap.utils.hclust(X, y)
	# above we implicitly used shap.maskers.Independent by passing a raw dataframe as the masker now we explicitly use a Partition masker that uses the clustering we just computed
	masker = shap.maskers.Partition(_background, clustering=clustering)

	model_fn = lambda x: abstracted_model(x, X, y, _background, precomputed_rows_with_min, random_seed)
	if fast_shap_estimate or len(input_features) > 20:
		explainer = shap.PermutationExplainer( # PermutationExplainer: Suitable for models where an efficient approximation of Shapley values is acceptable, and when feature independence is a reasonable assumption.
			model_fn, 
			masker, 
			seed=random_seed,
		)
		shap_values = explainer.shap_values(X)
	else:
		explainer = shap.explainers.Exact(
			model_fn, 
			masker, 
		)
		shap_values = explainer(X).values

	# print('shap_values', shap_values.shape, len(input_features), shap_values)

	# # Plot the feature importance (global explanation)
	# shap.plots.bar(shap_values, show=False)
	# plt.savefig(f'xai_analyses_results/shap_bar_plot_{model}_{metric}.png')
	# plt.close()

	# SHAP summary plot showing feature importance (score_types contributing to each metric)
	shap.summary_plot(shap_values, X, feature_names=input_features, plot_type="violin", show=False)
	plt.savefig(os.path.join(summary_plot_dir,f'shap_summary_plot_{model}_{difficulty}_{metric}.png'))
	plt.close()

	# try:
	# 	# Create the force plot for the first 100 samples (you can adjust the range)
	# 	shap.force_plot(explainer.expected_value, shap_values[:100], X.iloc[:100], show=False)
	# 	plt.savefig(f'xai_analyses_results/shap_force_plot_{model}_{metric}.png')
	# 	plt.close()
	# except Exception as e:
	# 	print(e)

	# # You can also add SHAP dependence plots for individual features if needed
	# for score_feature in input_features:
	# 	shap.dependence_plot(score_feature, shap_values, X, feature_names=input_features, show=False)
	# 	plt.savefig(os.path.join(dependence_plot_dir,f'shap_dependence_plot_{model}_{difficulty}_{metric}_{score_feature}.png'))
	# 	plt.close()

	global_feature_stats = metric_global_feature_stats_dict[metric] = get_global_feature_stats_from_shap_values(shap_values, input_features, y)
	print(f'global_feature_stats for {metric} {difficulty}: {json.dumps(global_feature_stats, indent=4)}')

create_cache(os.path.join(csv_file_dir, f'global_shap_stats_{model}_{difficulty}.pkl'), lambda: metric_global_feature_stats_dict)
