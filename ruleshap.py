# import os
# # Force single-threaded usage in BLAS/OpenBLAS/MKL/NumExpr
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
from more_itertools import unique_everseen
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, Lasso, LogisticRegressionCV
from functools import reduce

import numbers
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import validate_data, _check_sample_weight
from sklearn.linear_model._coordinate_descent import _pre_fit, _set_order, _alpha_grid, cd_fast
from sklearn.utils.validation import check_random_state
from scipy import sparse

from rulefit.rulefit import RuleFit, Winsorizer, FriedScale

from collections import defaultdict
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from numba import jit, njit
from numba import int32, int64, float64, optional, types
from numba.experimental import jitclass
from numba.typed import List, Dict

@njit
def float_to_str(value, n_decimals=6):
	if value < 0:
		is_negative = True
		value = -value
	else:
		is_negative = False

	integer_part = int(value)
	decimal_part = value - integer_part

	# Convert integer part to string
	int_str = ""
	if integer_part == 0:
		int_str = "0"
	else:
		while integer_part > 0:
			int_str = chr(48 + integer_part % 10) + int_str
			integer_part //= 10

	# Convert decimal part to string
	dec_str = ""
	if decimal_part > 0:
		dec_str = "."
		count = 0
		while decimal_part > 0 and count < n_decimals:  # Limit decimal places
			decimal_part *= 10
			digit = int(decimal_part)
			dec_str += chr(48 + digit)
			decimal_part -= digit
			count += 1

	# Combine parts
	result = int_str + dec_str
	if is_negative:
		result = "-" + result

	return result

@njit(fastmath=True)
def rand_int(max_value, seed):
	"""
	Generate a random integer between 0 and max_value (exclusive).

	Parameters
	----------
	max_value : int
		The upper bound of the range (exclusive).
	rng_state : numpy.random.Generator
		The random number generator state (NumPy Generator instance).
		Must be seeded and passed externally for reproducibility.

	Returns
	-------
	int
		Random integer between 0 and max_value - 1.
	"""
	# Linear Congruential Generator (LCG) for simple random number generation
	seed = (1103515245 * seed + 12345) % (2**31)
	return seed % max_value

@njit(fastmath=True)
def shap_enet_coordinate_descent( # original implementation at https://github.com/scikit-learn/scikit-learn/blob/d666202a9349893c1bd106cc9ee0ff0a807c7cf3/sklearn/linear_model/_cd_fast.pyx
	w,
	alpha,
	shap_weights,
	beta,
	X,
	y,
	max_iter=1000,
	tol=1e-4,
	seed=None,
	random=False,
	positive=False
):
	"""
	Numba version of the coordinate descent algorithm
	for Elastic-Net regression

	We minimize:

	(1/2) * norm(y - X w, 2)^2 + alpha * norm(w, 1) + (beta/2) * norm(w, 2)^2

	Returns
	-------
	w : ndarray of shape (n_features,)
		ElasticNet coefficients.
	gap : float
		Achieved dual gap.
	tol : float
		Tolerance used for the dual gap.
	n_iter : int
		Number of coordinate descent iterations.
	"""

	dtype = w.dtype
	n_samples, n_features = X.shape
	if random and seed:
		x_sum = int(np.sum(X))

	# compute norms of the columns of X
	norm_cols_X = np.square(X).sum(axis=0)

	# R = y - Xw
	R = y - X @ w

	# Adjust tolerance
	yy = np.dot(y, y)
	tol *= yy

	gap = tol + 1.0
	d_w_tol = tol

	# Temporary array
	XtA = np.empty(n_features, dtype=dtype)

	for n_iter in range(max_iter):
		w_max = 0.0
		d_w_max = 0.0

		for f_iter in range(n_features):
			if random and seed:
				ii = rand_int(n_features, seed + f_iter + x_sum)
				# print(random)
			else:
				ii = f_iter

			if norm_cols_X[ii] == 0.0:
				continue

			w_ii = w[ii]

			# If w[ii] != 0, remove its contribution from R
			if w_ii != 0.0:
				R += w_ii * X[:, ii]

			# tmp = X[:,ii]^T R
			tmp = np.dot(X[:, ii], R)

			if positive and tmp < 0:
				w[ii] = 0.0
			else:
				# Compute the adjusted L1 penalty term
				alpha_adjusted = alpha / shap_weights[ii]
				# Compute the adjusted L2 penalty term
				beta_adjusted = beta / shap_weights[ii]**2 if beta else 0

				# Soft-thresholding step
				val = abs(tmp) - alpha_adjusted
				if val < 0:
					w[ii] = 0.0
				else:
					w[ii] = np.sign(tmp) * (val / (norm_cols_X[ii] + beta_adjusted))

			# Update R
			if w[ii] != 0.0:
				R -= w[ii] * X[:, ii]

			d_w_ii = abs(w[ii] - w_ii)
			d_w_max = max(d_w_max, d_w_ii)
			w_max = max(w_max, abs(w[ii]))

		# Check for early stopping
		if (w_max == 0.0) or (d_w_max / w_max < d_w_tol) or (n_iter == max_iter - 1):
			# Compute duality gap

			# XtA = X^T R - beta * w
			# We'll do this in two steps:
			# First: XtA = X^T R
			XtA = X.T @ R - beta * w

			if positive:
				dual_norm_XtA = np.max(XtA)
			else:
				dual_norm_XtA = np.max(np.abs(XtA))

			R_norm2 = np.dot(R, R)
			adjusted_w = w / shap_weights
			w_norm2 = np.dot(adjusted_w, adjusted_w) if beta else 0

			if dual_norm_XtA > alpha:
				const = alpha / dual_norm_XtA
				A_norm2 = R_norm2 * (const ** 2)
				gap = 0.5 * (R_norm2 + A_norm2)
			else:
				const = 1.0
				gap = R_norm2

			l1_norm = np.sum(np.abs(w) / shap_weights)
			# print(0, l1_norm, np.sum(np.abs(w)))
			# print(1, shap_weights)

			gap += (alpha * l1_norm
					- const * np.dot(R, y)
					+ 0.5 * beta * (1 + const ** 2) * w_norm2)

			if gap < tol:
				# Converged
				return w, gap, tol, n_iter + 1

	# If not returned within the loop, we've not converged fully
	# Numba doesn't allow warnings.warn or custom warnings easily in nopython mode.
	# Just print a message or handle outside this function.
	# print("Objective did not converge. Increase max_iter, scale features, or increase regularization.")

	return w, gap, tol, max_iter

def lasso_path(
	X,
	y,
	*,
	eps=1e-3,
	n_alphas=100,
	alphas=None,
	precompute="auto",
	Xy=None,
	copy_X=True,
	coef_init=None,
	verbose=False,
	return_n_iter=False,
	positive=False,
	**params,
):
	return enet_path(
		X,
		y,
		l1_ratio=1.0,
		eps=eps,
		n_alphas=n_alphas,
		alphas=alphas,
		precompute=precompute,
		Xy=Xy,
		copy_X=copy_X,
		coef_init=coef_init,
		verbose=verbose,
		positive=positive,
		return_n_iter=return_n_iter,
		**params,
	)

def enet_path(
	X,
	y,
	*,
	l1_ratio=0.5,
	eps=1e-3,
	n_alphas=100,
	alphas=None,
	precompute="auto",
	Xy=None,
	copy_X=True,
	coef_init=None,
	verbose=False,
	return_n_iter=False,
	positive=False,
	check_input=True,
	**params,
):
	print('enet_path with SHAP weights')
	X_offset_param = params.pop("X_offset", None)
	X_scale_param = params.pop("X_scale", None)
	sample_weight = params.pop("sample_weight", None)
	tol = params.pop("tol", 1e-4)
	max_iter = params.pop("max_iter", 1000)
	random_state = params.pop("random_state", None)
	selection = params.pop("selection", "cyclic")
	shap_weights = params.pop("shap_weights", None)

	if len(params) > 0:
		raise ValueError("Unexpected parameters in params", params.keys())

	# Validate shap_weights
	if shap_weights is None: ### My edit
		shap_weights = np.ones(X.shape[1])

	if not isinstance(shap_weights, np.ndarray): ### My edit
		shap_weights = np.array(shap_weights)
	# if len(shap_weights) != X.shape[1]:
	#   raise ValueError(f"shap_weights must have the same length as the number of features. Current shape is {shap_weights.shape} but it should be {X.shape}")


	# We expect X and y to be already Fortran ordered when bypassing
	# checks
	if check_input:
		X = check_array(
			X,
			accept_sparse="csc",
			dtype=[np.float64, np.float32],
			order="F",
			copy=copy_X,
		)
		y = check_array(
			y,
			accept_sparse="csc",
			dtype=X.dtype.type,
			order="F",
			copy=False,
			ensure_2d=False,
		)
		if Xy is not None:
			# Xy should be a 1d contiguous array or a 2D C ordered array
			Xy = check_array(
				Xy, dtype=X.dtype.type, order="C", copy=False, ensure_2d=False
			)

	n_samples, n_features = X.shape

	multi_output = False
	if y.ndim != 1:
		multi_output = True
		n_targets = y.shape[1]

	if multi_output and positive:
		raise ValueError("positive=True is not allowed for multi-output (y.ndim != 1)")

	# MultiTaskElasticNet does not support sparse matrices
	if not multi_output and sparse.issparse(X):
		if X_offset_param is not None:
			# As sparse matrices are not actually centered we need this to be passed to
			# the CD solver.
			X_sparse_scaling = X_offset_param / X_scale_param
			X_sparse_scaling = np.asarray(X_sparse_scaling, dtype=X.dtype)
		else:
			X_sparse_scaling = np.zeros(n_features, dtype=X.dtype)

	# X should have been passed through _pre_fit already if function is called
	# from ElasticNet.fit
	if check_input:
		X, y, _, _, _, precompute, Xy = _pre_fit(
			X,
			y,
			Xy,
			precompute,
			fit_intercept=False,
			copy=False,
			check_input=check_input,
		)
	if alphas is None:
		# No need to normalize of fit_intercept: it has been done
		# above
		alphas = _alpha_grid(
			X,
			y,
			Xy=Xy,
			l1_ratio=l1_ratio,
			fit_intercept=False,
			eps=eps,
			n_alphas=n_alphas,
			copy_X=False,
		)
	elif len(alphas) > 1:
		alphas = np.sort(alphas)[::-1]  # make sure alphas are properly ordered

	n_alphas = len(alphas)
	dual_gaps = np.empty(n_alphas)
	n_iters = []

	random = selection == "random"

	if not multi_output:
		coefs = np.empty((n_features, n_alphas), dtype=X.dtype)
	else:
		coefs = np.empty((n_targets, n_features, n_alphas), dtype=X.dtype)

	if coef_init is None:
		coef_ = np.zeros(coefs.shape[:-1], dtype=X.dtype, order="F")
	else:
		coef_ = np.asfortranarray(coef_init, dtype=X.dtype)

	# print(2, shap_weights.shape, n_features)
	# print(0, alphas)
	# print(1, alphas.shape)

	for i, alpha in enumerate(alphas):
		# account for n_samples scaling in objectives between here and cd_fast
		l1_reg = alpha * l1_ratio * n_samples
		l2_reg = alpha * (1.0 - l1_ratio) * n_samples

		model = shap_enet_coordinate_descent( ### My edit
			coef_, 
			l1_reg, 
			shap_weights, 
			l2_reg, 
			X, 
			y, 
			max_iter=max_iter, 
			tol=tol, 
			seed=random_state, 
			random=random, 
			positive=positive
		)
		
		coef_, dual_gap_, eps_, n_iter_ = model
		coefs[..., i] = coef_
		# we correct the scale of the returned dual gap, as the objective
		# in cd_fast is n_samples * the objective in this docstring.
		dual_gaps[i] = dual_gap_ / n_samples
		n_iters.append(n_iter_)

		if verbose:
			if verbose > 2:
				print(model)
			elif verbose > 1:
				print("Path: %03i out of %03i" % (i, n_alphas))
			else:
				sys.stderr.write(".")

	if return_n_iter:
		return alphas, coefs, dual_gaps, n_iters
	return alphas, coefs, dual_gaps

class SHAPLasso(Lasso):

	path = staticmethod(lasso_path)
	def __init__(
		self,
		alpha=1.0,
		*,
		fit_intercept=True,
		precompute=False,
		copy_X=True,
		max_iter=1000,
		tol=1e-4,
		warm_start=False,
		positive=False,
		random_state=None,
		selection="cyclic",
		shap_weights=None
	):
		super().__init__(
			alpha=alpha,
			fit_intercept=fit_intercept,
			precompute=precompute,
			copy_X=copy_X,
			max_iter=max_iter,
			tol=tol,
			warm_start=warm_start,
			positive=positive,
			random_state=random_state,
			selection=selection,
		)
		self.shap_weights = shap_weights
		print('SHAPLasso::__init__')

	def fit(self, X, y, sample_weight=None, check_input=True):
		print('SHAPLasso::fit')
		if self.alpha == 0:
			warnings.warn(
				(
					"With alpha=0, this algorithm does not converge "
					"well. You are advised to use the LinearRegression "
					"estimator"
				),
				stacklevel=2,
			)

		# Remember if X is copied
		X_copied = False
		# We expect X and y to be float64 or float32 Fortran ordered arrays
		# when bypassing checks
		if check_input:
			X_copied = self.copy_X and self.fit_intercept
			X, y = validate_data(
				self,
				X,
				y,
				accept_sparse="csc",
				order="F",
				dtype=[np.float64, np.float32],
				force_writeable=True,
				accept_large_sparse=False,
				copy=X_copied,
				multi_output=True,
				y_numeric=True,
			)
			y = check_array(
				y, order="F", copy=False, dtype=X.dtype.type, ensure_2d=False
			)

		n_samples, n_features = X.shape
		alpha = self.alpha

		if isinstance(sample_weight, numbers.Number):
			sample_weight = None
		if sample_weight is not None:
			if check_input:
				sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
			# TLDR: Rescale sw to sum up to n_samples.
			# Long: The objective function of Enet
			#
			#    1/2 * np.average(squared error, weights=sw)
			#    + alpha * penalty                                             (1)
			#
			# is invariant under rescaling of sw.
			# But enet_path coordinate descent minimizes
			#
			#     1/2 * sum(squared error) + alpha' * penalty                  (2)
			#
			# and therefore sets
			#
			#     alpha' = n_samples * alpha                                   (3)
			#
			# inside its function body, which results in objective (2) being
			# equivalent to (1) in case of no sw.
			# With sw, however, enet_path should set
			#
			#     alpha' = sum(sw) * alpha                                     (4)
			#
			# Therefore, we use the freedom of Eq. (1) to rescale sw before
			# calling enet_path, i.e.
			#
			#     sw *= n_samples / sum(sw)
			#
			# such that sum(sw) = n_samples. This way, (3) and (4) are the same.
			sample_weight = sample_weight * (n_samples / np.sum(sample_weight))
			# Note: Alternatively, we could also have rescaled alpha instead
			# of sample_weight:
			#
			#     alpha *= np.sum(sample_weight) / n_samples

		# Ensure copying happens only once, don't do it again if done above.
		# X and y will be rescaled if sample_weight is not None, order='F'
		# ensures that the returned X and y are still F-contiguous.
		should_copy = self.copy_X and not X_copied
		X, y, X_offset, y_offset, X_scale, precompute, Xy = _pre_fit(
			X,
			y,
			None,
			self.precompute,
			fit_intercept=self.fit_intercept,
			copy=should_copy,
			check_input=check_input,
			sample_weight=sample_weight,
		)
		# coordinate descent needs F-ordered arrays and _pre_fit might have
		# called _rescale_data
		if check_input or sample_weight is not None:
			X, y = _set_order(X, y, order="F")
		if y.ndim == 1:
			y = y[:, np.newaxis]
		if Xy is not None and Xy.ndim == 1:
			Xy = Xy[:, np.newaxis]

		n_targets = y.shape[1]

		if not self.warm_start or not hasattr(self, "coef_"):
			coef_ = np.zeros((n_targets, n_features), dtype=X.dtype, order="F")
		else:
			coef_ = self.coef_
			if coef_.ndim == 1:
				coef_ = coef_[np.newaxis, :]

		dual_gaps_ = np.zeros(n_targets, dtype=X.dtype)
		self.n_iter_ = []

		for k in range(n_targets):
			if Xy is not None:
				this_Xy = Xy[:, k]
			else:
				this_Xy = None
			# print(k, shap_weights[k], coef_[k].shape)
			_, this_coef, this_dual_gap, this_iter = self.path(
				X,
				y[:, k],
				eps=None,
				n_alphas=None,
				alphas=[alpha],
				precompute=precompute,
				Xy=this_Xy,
				copy_X=True,
				coef_init=coef_[k],
				verbose=False,
				return_n_iter=True,
				positive=self.positive,
				check_input=False,
				# from here on **params
				tol=self.tol,
				X_offset=X_offset,
				X_scale=X_scale,
				max_iter=self.max_iter,
				random_state=self.random_state,
				selection=self.selection,
				sample_weight=sample_weight,
				shap_weights=self.shap_weights, ### My edit
			)
			coef_[k] = this_coef[:, 0]
			dual_gaps_[k] = this_dual_gap[0]
			self.n_iter_.append(this_iter[0])

		if n_targets == 1:
			self.n_iter_ = self.n_iter_[0]
			self.coef_ = coef_[0]
			self.dual_gap_ = dual_gaps_[0]
		else:
			self.coef_ = coef_
			self.dual_gap_ = dual_gaps_

		self._set_intercept(X_offset, y_offset, X_scale)

		# check for finiteness of coefficients
		if not all(np.isfinite(w).all() for w in [self.coef_, self.intercept_]):
			raise ValueError(
				"Coordinate descent iterations resulted in non-finite parameter"
				" values. The input data may contain large values and need to"
				" be preprocessed."
			)

		# return self for chaining fit and predict calls
		return self

class SHAPLassoCV(LassoCV):
	
	path = staticmethod(lasso_path)
	def __init__(
		self,
		*,
		eps=1e-3,
		n_alphas=100,
		alphas=None,
		fit_intercept=True,
		precompute="auto",
		max_iter=1000,
		tol=1e-4,
		copy_X=True,
		cv=None,
		verbose=False,
		n_jobs=None,
		positive=False,
		random_state=None,
		selection="cyclic",
		shap_weights=None,
	):
		super().__init__(
			eps=eps,
			n_alphas=n_alphas,
			alphas=alphas,
			fit_intercept=fit_intercept,
			precompute=precompute,
			max_iter=max_iter,
			tol=tol,
			copy_X=copy_X,
			cv=cv,
			verbose=verbose,
			n_jobs=n_jobs,
			positive=positive,
			random_state=random_state,
			selection=selection,
		)
		self.shap_weights = shap_weights

	def _get_estimator(self):
		return SHAPLasso(shap_weights=self.shap_weights)

# class Winsorizer():
#   """Performs Winsorization 1->1*

#   Warning: this class should not be used directly.
#   """    
#   def __init__(self,trim_quantile=0.0):
#       self.trim_quantile=trim_quantile
#       self.winsor_lims=None
		
#   def train(self,X):
#       # get winsor limits
#       self.winsor_lims=np.ones([2,X.shape[1]])*np.inf
#       self.winsor_lims[0,:]=-np.inf
#       if self.trim_quantile>0:
#           for i_col in np.arange(X.shape[1]):
#               lower=np.percentile(X[:,i_col],self.trim_quantile*100)
#               upper=np.percentile(X[:,i_col],100-self.trim_quantile*100)
#               self.winsor_lims[:,i_col]=[lower,upper]
		
#   def trim(self,X):
#       X_=X.copy()
#       X_=np.where(X>self.winsor_lims[1,:],np.tile(self.winsor_lims[1,:],[X.shape[0],1]),np.where(X<self.winsor_lims[0,:],np.tile(self.winsor_lims[0,:],[X.shape[0],1]),X))
#       return X_

# class FriedScale():
#   """Performs scaling of linear variables according to Friedman et al. 2005 Sec 5

#   Each variable is first Winsorized l->l*, then standardised as 0.4 x l* / std(l*)
#   Warning: this class should not be used directly.
#   """    
#   def __init__(self, winsorizer = None):
#       self.scale_multipliers=None
#       self.winsorizer = winsorizer
		
#   def train(self,X):
#       # get multipliers
#       if self.winsorizer != None:
#           X_trimmed= self.winsorizer.trim(X)
#       else:
#           X_trimmed = X

#       scale_multipliers=np.ones(X.shape[1])
#       for i_col in np.arange(X.shape[1]):
#           num_uniq_vals=len(np.unique(X[:,i_col]))
#           if num_uniq_vals>2: # don't scale binary variables which are effectively already rules
#               scale_multipliers[i_col]=0.4/(1.0e-12 + np.std(X_trimmed[:,i_col]))
#       self.scale_multipliers=scale_multipliers
		
#   def scale(self,X):
#       if self.winsorizer != None:
#           return self.winsorizer.trim(X)*self.scale_multipliers
#       else:
#           return X*self.scale_multipliers

# Define the data types for Numba jitclass
type_spec_rule_condition = [
	('feature_index', int32),
	('threshold', float64),
	('operator', types.unicode_type),
	('support', optional(float64)),
	('gain', optional(float64)),
	('feature_name', optional(types.unicode_type))
]

@jitclass(type_spec_rule_condition)
class RuleCondition:
	"""Class for binary rule condition."""

	def __init__(self, feature_index, threshold, operator, support=0.0, gain=0.0, feature_name=''):
		self.feature_index = feature_index
		self.threshold = threshold
		self.operator = operator
		self.support = support
		self.gain = gain
		self.feature_name = feature_name

	def transform_weight(self, weight):
		return weight[self.feature_index]

	def transform(self, X):
		"""Transform dataset.

		Parameters
		----------
		X: array-like matrix, shape=(n_samples, n_features)

		Returns
		-------
		X_transformed: array-like matrix, shape=(n_samples, 1)
		"""
		if self.operator == "<=":
			res = 1 * (X[:, self.feature_index] <= self.threshold)
		elif self.operator == ">":
			res = 1 * (X[:, self.feature_index] > self.threshold)
		elif self.operator == ">=":
			res = 1 * (X[:, self.feature_index] >= self.threshold)
		elif self.operator == "<":
			res = 1 * (X[:, self.feature_index] < self.threshold)
		else:
			raise ValueError("Unsupported operator")
		return res

	def get_unique_key(self):
		return (self.threshold, self.operator, self.feature_name)

	def __hash__(self):
		return hash(self.get_unique_key())

	def __str__(self):
		feature = self.feature_name if self.feature_name else f"f{self.feature_index}"
		# Use Numba-compatible formatting for the threshold
		threshold_str = float_to_str(self.threshold, 2)  # Format float as a concise string
		# Combine strings into the final representation
		return feature + " " + self.operator + " " + threshold_str

# # Make sure RuleCondition is compiled and ready
# _ = RuleCondition.class_type.instance_type

# Define the data types for Numba jitclass
type_spec_rule = [
	('conditions', types.ListType(RuleCondition.class_type.instance_type)),
	('prediction_value', optional(float64)),
	('is_negated', optional(types.boolean)),
	('max_gain', float64),
	('total_gain', float64),
	('min_rule_coverage', float64),
	('global_coverage', float64),
]

@jitclass(type_spec_rule)
class Rule():
	"""Class for binary Rules from list of conditions"""

	def __init__(self, rule_conditions, prediction_value=0., is_negated=False, global_coverage=-1):
		# Convert the input list to a numba.typed.List
		self.conditions = rule_conditions
		self.conditions.sort(key=lambda x: (x.feature_index, x.threshold))
		self.max_gain = np.max(np.array([x.gain for x in rule_conditions]))
		self.total_gain = sum([x.gain for x in rule_conditions])
		self.min_rule_coverage = np.min(np.array([x.support for x in rule_conditions]))
		# self.total_rule_coverage = sum((x.support for x in rule_conditions))
		self.prediction_value = prediction_value
		self.is_negated = is_negated
		self.global_coverage = global_coverage

	def transform_weight(self, weight): ### My edit
		unique_conditions = list({c.feature_index: c for c in self.conditions}.values())  # Avoid duplicates by feature_index
		# if len(self.conditions) != len(unique_conditions):
		# 	print(len(self.conditions), len(unique_conditions))
		weight_applies = [condition.transform_weight(weight) for condition in unique_conditions]
		return reduce(lambda x,y: x + y, weight_applies)/len(weight_applies) # Prioritize rules with fewer more important conditions/features over rules with many less important conditions/features

	def transform(self, X):
		rule_applies = [condition.transform(X) for condition in self.conditions]
		if not self.is_negated:
			return reduce(lambda x,y: x * y, rule_applies)
		return reduce(lambda x,y: np.maximum(x,y), rule_applies)

	def get_coverage(self, X):
		return np.mean(self.transform(X))

	def __str__(self):
		if not self.is_negated:
			return  " & ".join([str(x) for x in self.conditions])
		return  " | ".join([str(x) for x in self.conditions])

	def __hash__(self):
		return sum([hash(condition) for condition in self.conditions])

	def __gt__(self, other):
		if self.is_negated and other.is_negated:
			return len(set([hash(cond) for cond in other.conditions]) - set([hash(cond) for cond in self.conditions])) == 0
		if not self.is_negated and not other.is_negated:
			return len(set([hash(cond) for cond in self.conditions]) - set([hash(cond) for cond in other.conditions])) == 0
		return False

	def __lt__(self, other):
		if self.is_negated and other.is_negated:
			return len(set([hash(cond) for cond in self.conditions]) - set([hash(cond) for cond in other.conditions])) == 0
		if not self.is_negated and not other.is_negated:
			return len(set([hash(cond) for cond in other.conditions]) - set([hash(cond) for cond in self.conditions])) == 0
		return False

def negate_operator(operator):
	if operator == '>=':
		return '<'
	if operator == '<=':
		return '>'
	if operator == '>':
		return '<='
	if operator == '<':
		return '>='
	raise ValueError("Unsupported operator")

def negate_rule(rule):
	conditions = [
		RuleCondition(cond.feature_index, cond.threshold, negate_operator(cond.operator), support=cond.support, gain=cond.gain, feature_name=cond.feature_name)
		for cond in rule.conditions
	]
	negated_rule = Rule(List(conditions), prediction_value=rule.prediction_value, is_negated=not rule.is_negated, global_coverage=1-rule.global_coverage)
	return negated_rule

class RuleEnsemble():
	"""Ensemble of binary decision rules

	This class implements an ensemble of decision rules that extracts rules from
	an ensemble of decision trees.

	Parameters
	----------
	tree_list: List or array of DecisionTreeClassifier or DecisionTreeRegressor
		Trees from which the rules are created

	feature_names: List of strings, optional (default=None)
		Names of the features

	Attributes
	----------
	rules: List of Rule
		The ensemble of rules extracted from the trees
	"""
	def __init__(self, tree_list=None, tree_dump=None, datapoints=None, feature_names=None):
		self.feature_names = feature_names
		self.datapoints_count = len(datapoints)
		
		if tree_dump:
			self.rules = list(unique_everseen((
				rule
				for tree_dump in map(json.loads, tree_dump)
				for rule in RuleEnsemble.extract_rules_from_tree_dump(tree_dump, self.datapoints_count, feature_names=self.feature_names)
			), key=hash))
		else:
			self.rules = list(unique_everseen((
				rule
				for tree in tree_list
				for rule in RuleEnsemble.extract_rules_from_tree_list(tree[0].tree_, self.datapoints_count, feature_names=self.feature_names)
			), key=hash))
		for rule in self.rules:
			rule.global_coverage = rule.get_coverage(datapoints)
		self.rules = list(filter(lambda x: x.global_coverage != 0, self.rules)) # Remove any rule not covered by real data
		# self.rules += list(map(negate_rule, self.rules))
		print("Rules found:", len(self.rules))

	@staticmethod
	def extract_rules_from_tree_list(tree, datapoints_count, feature_names=None):
		"""Helper to turn a tree into as set of rules
		"""
		rules = set()

		def traverse_nodes(node_id=0, operator=None, threshold=None, feature=None, conditions=[]):
			# Children
			left_child = tree.children_left[node_id]
			right_child = tree.children_right[node_id]
			
			if node_id != 0:
				if feature_names is not None:
					feature_name = feature_names[feature]
				else:
					feature_name = feature

				# Impurities
				parent_impurity = tree.impurity[node_id]
				left_impurity = tree.impurity[left_child]
				right_impurity = tree.impurity[right_child]

				# Number of samples
				parent_samples = tree.weighted_n_node_samples[node_id]
				left_samples = tree.weighted_n_node_samples[left_child]
				right_samples = tree.weighted_n_node_samples[right_child]

				# Weighted impurity of children
				weighted_impurity = (
					(left_samples * left_impurity + right_samples * right_impurity) / parent_samples
				)
				
				# Gain
				gain = parent_impurity - weighted_impurity

				rule_condition = RuleCondition(feature_index=feature,
											   threshold=threshold,
											   operator=operator,
											   support = tree.n_node_samples[node_id] / datapoints_count,
											   gain=gain,
											   feature_name=feature_name)
				new_conditions = conditions + [rule_condition]
			else:
				new_conditions = []
			## if not terminal node
			if left_child != right_child: 
				feature = tree.feature[node_id]
				threshold = tree.threshold[node_id]
				traverse_nodes(left_child, "<=", threshold, feature, new_conditions)
				traverse_nodes(right_child, ">", threshold, feature, new_conditions)
			else: # a leaf node
				if len(new_conditions)>0:
					new_rule = Rule(List(RuleEnsemble.filter_conditions(new_conditions)),tree.value[node_id][0][0])
					rules.update([new_rule])
				else:
					pass #tree only has a root node!
				return None

		traverse_nodes()
		
		return rules

	@staticmethod
	def extract_rules_from_tree_dump(tree, datapoints_count, feature_names=None):
		"""Recursively extract rules from a JSON XGBoost tree."""
		rules = []

		# print('tree:', json.dumps(tree, indent=4))

		def traverse_nodes(node, conditions=[]):
			# Base case: Check if it's a leaf node
			if 'leaf' in node:
				if conditions:
					new_rule = Rule(List(RuleEnsemble.filter_conditions(conditions)), node['leaf'])
					rules.append(new_rule)
				return

			# Extract split details
			feature = node['split']
			feature_index = int(feature[1:]) if feature.startswith('f') else int(feature)
			feature_name = feature_names[feature_index] if feature_names else feature
			threshold = node['split_condition']
			support = node['cover'] / datapoints_count
			gain = node['gain']
			yes_child = node['yes']
			no_child = node['no']
			missing_child = node['missing']

			# Condition for the left child (<= threshold)
			left_condition = RuleCondition(feature_index=feature_index,
											 threshold=threshold,
											 operator="<=",
											 support=support,
											 gain=gain,
											 feature_name=feature_name)

			# Condition for the right child (> threshold)
			right_condition = RuleCondition(feature_index=feature_index,
											threshold=threshold,
											operator=">",
											support=support,
											gain=gain,
											feature_name=feature_name)

			# Recurse for children with updated conditions
			for child in node.get('children', []):
				if child['nodeid'] == yes_child:  # Left child
					traverse_nodes(child, conditions + [left_condition])
				elif child['nodeid'] == no_child:  # Right child
					traverse_nodes(child, conditions + [right_condition])

		traverse_nodes(tree)
		return rules

	@staticmethod
	def filter_conditions(conditions):
		"""
		Filters RuleCondition objects:
		- Keeps only the highest threshold for operator '>='
		- Keeps only the lowest threshold for operator '<='

		Parameters:
		-----------
		conditions : list[RuleCondition]
			List of RuleCondition objects to filter.

		Returns:
		--------
		filtered_conditions : list[RuleCondition]
			Filtered list of RuleCondition objects.
		"""

		grouped_conditions = defaultdict(list)

		# Group conditions by feature_index and operator
		for condition in conditions:
			key = (condition.feature_index, condition.operator.strip('='))
			grouped_conditions[key].append(condition)

		filtered_conditions = []

		# Apply filtering logic
		for (feature_index, operator), group in grouped_conditions.items():
			if operator == '>':
				# Keep condition with the highest threshold
				best_condition = max(group, key=lambda x: (x.threshold, 0 if x.operator.endswith('=') else 1))
			elif operator == '<':
				# Keep condition with the lowest threshold
				best_condition = min(group, key=lambda x: (x.threshold, 1 if x.operator.endswith('=') else 0))
			
			filtered_conditions.append(best_condition)

		return filtered_conditions

	def transform(self, X, shap_weights=None):
		"""Transform dataset.

		Parameters
		----------
		X:      array-like matrix, shape=(n_samples, n_features)
		Returns
		-------
		X_transformed: array-like matrix, shape=(n_samples, n_out)
			Transformed dataset. Each column represents one rule.
		"""
		transformed_shap_weights = np.array([rule.transform_weight(shap_weights) for rule in self.rules]).T if shap_weights is not None else None ### My edit
		return np.array([rule.transform(X) for rule in self.rules]).T, transformed_shap_weights

	def __str__(self):
		return ' '.join(map(str, self.rules))

class RuleSHAP(BaseEstimator, TransformerMixin):

	def __init__(self, gboost_config_dict=None, model_type='rl', rfmode='regress', lin_trim_quantile=0.025, lin_standardise=True, Cs=None, cv=3, random_state=None, max_rules=4000, tree_size=10):
		if gboost_config_dict is None:
			gboost_config_dict = {
				'n_estimators': 100, # Number of boosting rounds. Each round is a tree
				'max_depth': 10, # max rule lenght
				'subsample': 0.8, # Subsample ratio of the training instance.
				# 'max_leaves': 50, # Maximum number of terminal nodes (leaves)
				# 'learning_rate': 0.01,
			}
		gboost_config_dict['random_state'] = random_state
		self.gboost_config_dict = gboost_config_dict
		self.model_type = model_type
		self.rfmode = rfmode
		self.lin_trim_quantile=lin_trim_quantile
		self.lin_standardise=lin_standardise
		self.winsorizer=Winsorizer(trim_quantile=lin_trim_quantile)
		self.friedscale=FriedScale(self.winsorizer)
		self.Cs = Cs
		self.cv = cv
		self.random_state = random_state
		self.max_rules = max_rules
		self.tree_size = tree_size
	
	def fit(self, X, y=None, feature_names=None, sample_weight=1, shap_weights=None, use_shap_in_xgb=True, use_shap_in_lasso=True):
		print('Calling RuleSHAP')

		if isinstance(shap_weights, (list, tuple)): ### My edit
			shap_weights = np.array(shap_weights)

		if isinstance(shap_weights, np.ndarray): ### My edit
			assert len(shap_weights) == X.shape[1], "Feature weights must match the number of features!"
			assert np.all(shap_weights >= 0), "Feature weights must be non-negative!"
			assert np.all(np.isfinite(shap_weights)), "Feature weights must be finite numbers!"
			shap_weights = shap_weights/np.sum(shap_weights) # SHAP weights normalized in (0,1]

		N=X.shape[0]
		if feature_names is None:
			self.feature_names = ['feature_' + str(x) for x in range(0, X.shape[1])]
		else:
			self.feature_names=feature_names

		if 'r' in self.model_type:
			#############################################################################
			# initialise tree generator
			if use_shap_in_xgb:
				feature_weights = shap_weights # Weight for each feature, defines the probability of each feature being selected when colsample is being used. All values must be greater than 0, otherwise a ValueError is thrown.
				self.gboost_config_dict['tree_method'] = 'exact'
				# self.gboost_config_dict['colsample_bytree'] = 0.5 # Fraction of features considered for each tree. Limits the number of features, simplifying the rules. Use values around 0.5–0.8.
				self.gboost_config_dict['colsample_bylevel'] = 1/len(self.feature_names) # Fraction of features considered for each level. Limits the number of features, simplifying the rules.
			else:
				feature_weights = None
			########
			if self.rfmode=='regress':
				# XGBRegressor with feature_weights
				self.tree_generator = XGBRegressor(**self.gboost_config_dict)
			else:
				self.tree_generator = XGBClassifier(**self.gboost_config_dict)
			########
			self.tree_generator.fit(X, y, feature_weights=feature_weights)
			tree_dump = self.tree_generator.get_booster().get_dump(with_stats=True, dump_format='json') # Access the Booster and get individual trees as strings. Dump the individual trees into a list
			self.rule_ensemble = RuleEnsemble(datapoints=X, tree_dump=tree_dump, feature_names=self.feature_names) ## extract rules
			#############################################################################

			# ############################################################################
			# # initialise tree generator
			# if self.rfmode=='regress':
			# 	self.tree_generator = GradientBoostingRegressor(
			# 		**self.gboost_config_dict,
			# 	)
			# else:
			# 	self.tree_generator = GradientBoostingClassifier(
			# 		**self.gboost_config_dict,
			# 	)
			# np.random.seed(self.random_state)
			# tree_sizes=np.random.exponential(scale=self.tree_size-2,size=int(np.ceil(self.max_rules*2/self.tree_size)))
			# tree_sizes=np.asarray([2+np.floor(tree_sizes[i_]) for i_ in np.arange(len(tree_sizes))],dtype=int)
			# i=int(len(tree_sizes)/4)
			# while np.sum(tree_sizes[0:i])<self.max_rules:
			# 	i=i+1
			# tree_sizes=tree_sizes[0:i]
			# self.tree_generator.set_params(warm_start=True) 
			# curr_est_=0
			# for i_size in np.arange(len(tree_sizes)):
			# 	size=tree_sizes[i_size]
			# 	self.tree_generator.set_params(n_estimators=curr_est_+1)
			# 	self.tree_generator.set_params(max_leaf_nodes=size)
			# 	random_state_add = self.random_state if self.random_state else 0
			# 	self.tree_generator.set_params(random_state=i_size+random_state_add) # warm_state=True seems to reset random_state, such that the trees are highly correlated, unless we manually change the random_sate here.
			# 	self.tree_generator.get_params()['n_estimators']
			# 	self.tree_generator.fit(np.copy(X, order='C'), np.copy(y, order='C'))
			# 	curr_est_=curr_est_+1
			# self.tree_generator.set_params(warm_start=False) 
			# tree_list = self.tree_generator.estimators_
			# if isinstance(self.tree_generator, RandomForestRegressor) or isinstance(self.tree_generator, RandomForestClassifier):
			# 	tree_list = [[x] for x in self.tree_generator.estimators_]
			# self.rule_ensemble = RuleEnsemble(tree_list=tree_list, datapoints_count=N, feature_names=self.feature_names) ## extract rules
			# ############################################################################

			## concatenate original features and rules
			X_rules, rules_shap_weights = self.rule_ensemble.transform(X, shap_weights=shap_weights) ### My edit
			# rules_shap_weights = np.log(rules_shap_weights)
		
		## standardise linear variables if requested (for regression model only)
		if 'l' in self.model_type: 

			## standard deviation and mean of winsorized features
			self.winsorizer.train(X)
			winsorized_X = self.winsorizer.trim(X)
			self.stddev = np.std(winsorized_X, axis = 0)
			self.mean = np.mean(winsorized_X, axis = 0)

			if self.lin_standardise:
				self.friedscale.train(X)
				X_regn=self.friedscale.scale(X)
			else:
				X_regn=X.copy() 
			regn_shap_weights = shap_weights ### My edit
			# regn_shap_weights = np.log(regn_shap_weights)
		
		## Compile Training data
		X_concat=np.zeros([X.shape[0],0])
		concat_shap_weights = np.array([])  # Initialize as a 1D array ### My edit
		if 'l' in self.model_type:
			X_concat = np.concatenate((X_concat,X_regn), axis=1)
			if shap_weights is not None:
				concat_shap_weights = np.concatenate((concat_shap_weights,regn_shap_weights)) ### My edit
		if 'r' in self.model_type:
			if X_rules.shape[0] >0:
				X_concat = np.concatenate((X_concat, X_rules), axis=1)
				if shap_weights is not None:
					concat_shap_weights = np.concatenate((concat_shap_weights,rules_shap_weights)) ### My edit

		## fit Lasso
		if self.rfmode=='regress':
			if self.Cs is None: # use defaultshasattr(self.Cs, "__len__"):
				n_alphas= 100
				alphas=None
			elif hasattr(self.Cs, "__len__"):
				n_alphas= None
				alphas=1./self.Cs
			else:
				n_alphas= self.Cs
				alphas=None
			# Derive feature weights from SHAP values by using the mean absolute SHAP value per feature as its weight. This makes the shap_weights a probability distribution
			# concat_shap_weights = concat_shap_weights/np.max(concat_shap_weights)
			self.lscv = SHAPLassoCV(
				n_alphas=n_alphas, 
				alphas=alphas, 
				cv=self.cv, 
				random_state=self.random_state, 
				shap_weights=None if not use_shap_in_lasso or concat_shap_weights.size == 0 else concat_shap_weights
			) ### My edit
			self.lscv.fit(X_concat, y, sample_weight=sample_weight)
			self.coef_=self.lscv.coef_
			self.intercept_=self.lscv.intercept_
		else:
			# Cs=10 if self.Cs is None else self.Cs
			# self.lscv=LogisticRegressionCV(Cs=Cs,cv=self.cv,penalty='l1',random_state=self.random_state,solver='liblinear')
			# self.lscv.fit(X_concat, y)
			# self.coef_=self.lscv.coef_[0]
			# self.intercept_=self.lscv.intercept_[0]
			raise ValueError("Not implemented; rfmode", self.rfmode)
		
		return self

	def predict(self, X):
		"""Predict outcome for X

		"""
		X_concat=np.zeros([X.shape[0],0])
		if 'l' in self.model_type:
			if self.lin_standardise:
				X_concat = np.concatenate((X_concat,self.friedscale.scale(X)), axis=1)
			else:
				X_concat = np.concatenate((X_concat,X), axis=1)
		if 'r' in self.model_type:
			rule_coefs=self.coef_[-len(self.rule_ensemble.rules):] 
			if len(rule_coefs)>0:
				X_rules,_ = self.rule_ensemble.transform(X)
				if X_rules.shape[0] >0:
					X_concat = np.concatenate((X_concat, X_rules), axis=1)
		return self.lscv.predict(X_concat)

	def transform(self, X=None, shap_weights=None):
		"""Transform dataset.

		Parameters
		----------
		X : array-like matrix, shape=(n_samples, n_features)
		  Input data to be transformed. Use ``dtype=np.float32`` for maximum
		  efficiency.

		Returns
		-------
		X_transformed: matrix, shape=(n_samples, n_out)
		  Transformed data set
		"""
		return self.rule_ensemble.transform(X, shap_weights=shap_weights)

	def get_rules(self):
		"""Return the estimated rules

		Parameters
		----------
		subregion: If None (default) returns global importances (FP 2004 eq. 28/29), else returns importance over 
						 subregion of inputs (FP 2004 eq. 30/31/32).

		Returns
		-------
		rules: pandas.DataFrame with the rules. Column 'rule' describes the rule, 'coef' holds
			 the coefficients and 'support' the support of the rule in the training
			 data set (X)
		"""

		output_rules = []

		## Add coefficients for linear effects
		n_features= len(self.coef_) - len(self.rule_ensemble.rules)
		for i in range(0, n_features):
			coef = self.coef_[i]
			if self.lin_standardise:
				coef *= self.friedscale.scale_multipliers[i]
			if not coef:
				continue
			importance = abs(coef)#*self.stddev[i]
			impact_dir = 'negative' if coef < 0 else 'positive'
			output_rules.append((self.feature_names[i], 'linear', None, impact_dir, round(importance,2), None, None, None, None))
		####

		rule_list = list(self.rule_ensemble.rules)
		coefficient_list = self.coef_[n_features:]
		rule_coefficient_iter = zip(rule_list, coefficient_list)
		rule_coefficient_iter = filter(lambda x: x[-1], rule_coefficient_iter)
		# rule_coefficient_iter = filter_redundant_rules(list(rule_coefficient_iter))
		# rule_coefficient_iter = unique_everseen( # Remove redundant negated rules
		# 	sorted(rule_coefficient_iter, key=lambda x: abs(x[-1]), reverse=True), 
		# 	lambda x: hash(x[0])+hash(negate_rule(x[0]))
		# )
		
		## Add rules
		for rule, coef in rule_coefficient_iter:
			importance = abs(coef)#*(rule.support * (1-rule.support))**(1/2)
			impact_dir = 'negative' if coef < 0 else 'positive'
			weighted_importance = (1+importance)*np.maximum(1,np.log(np.abs(rule.max_gain)))
			output_rules.append((str(rule), 'rule', round(rule.global_coverage,2), impact_dir, round(importance,2), round(rule.min_rule_coverage,2), round(rule.total_gain,2), round(rule.max_gain,2), round(weighted_importance,2)))
		
		return pd.DataFrame(output_rules, columns=["rule", "type", "conditions_coverage", "impact_direction", "importance", "min_rule_coverage", "total_gain", "max_gain", "weighted_importance"])

@jit(nopython=True, fastmath=True)
def filter_redundant_rules(rule_coefficient_list):
	if len(rule_coefficient_list) <= 1:
		return rule_coefficient_list
	# Sort by min_rule_coverage in descending order
	rule_coefficient_list.sort(key=lambda x: abs(x[-1]), reverse=True)

	# Filter out redundant rules
	filtered_rule_coefficient_list = [rule_coefficient_list[0]]
	for i in range(1, len(rule_coefficient_list)):
		this_rule, this_coef = rule_coefficient_list[i]
		this_coef_sign = np.sign(this_coef)
		this_rule_is_negated = this_rule.is_negated
		better_rule_index = None
		j = 0
		while j < i and j < len(filtered_rule_coefficient_list) and better_rule_index is None:
			other_rule, other_coef = filtered_rule_coefficient_list[j]
			other_coef_sign = np.sign(other_coef)
			other_rule_is_negated = other_rule.is_negated
			if this_coef_sign == other_coef_sign and this_rule_is_negated == other_rule_is_negated:
				if this_rule < other_rule:
					better_rule_index = j
			j += 1

		if better_rule_index is None:
			filtered_rule_coefficient_list.append(rule_coefficient_list[i])

	return filtered_rule_coefficient_list

###############
### Test cases
###############
assert Rule(List([
	RuleCondition(0, 0.5, '>'),
	RuleCondition(1, 0.9, '<='),
	RuleCondition(2, 0.7, '>'),
])) > Rule(List([
	RuleCondition(0, 0.5, '>'),
	RuleCondition(1, 0.9, '<='),
	RuleCondition(2, 0.7, '>'),
	RuleCondition(3, 0.7, '>'),
])), "Failed test case 1"
###############
assert not Rule(List([
	RuleCondition(0, 0.5, '>'),
	RuleCondition(1, 0.6, '<='),
	RuleCondition(2, 0.7, '>'),
])) > Rule(List([
	RuleCondition(0, 0.5, '>'),
	RuleCondition(1, 0.9, '<='),
	RuleCondition(2, 0.7, '>'),
	RuleCondition(3, 0.7, '>'),
])), "Failed test case 2"
###############
assert Rule(List([
	RuleCondition(0, 0.5, '>'),
	RuleCondition(1, 0.9, '<='),
	RuleCondition(2, 0.7, '>'),
	RuleCondition(3, 0.7, '>'),
])) < Rule(List([
	RuleCondition(0, 0.5, '>'),
	RuleCondition(1, 0.9, '<='),
	RuleCondition(2, 0.7, '>'),
])), "Failed test case 3"