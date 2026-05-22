import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree


class GELPE:
    """
    GELPE-inspired CART surrogate for the structured abstraction benchmark.

    Important: this is an adaptation, not a line-by-line reimplementation of the
    original GELPE pipeline from SKE_NLP. The original code:
    - aggregates local token/lemma relevance scores across examples,
    - keeps the top-k valuable lemmas / skipgrams,
    - binarizes their presence in each input, and
    - trains a CART surrogate over that binary dataframe.

    This repository does not operate on token-level local explanations. Inputs are
    already structured abstractions, so the closest analogue is to:
    - aggregate SHAP relevance over abstractions,
    - keep the top-k abstractions, and
    - fit a CART surrogate on that restricted feature set.

    The resulting leaf paths are exported as global rule expressions. This makes
    the baseline GELPE-inspired and review-defensible, but not a faithful port of
    the original text-classification implementation.
    """

    def __init__(
        self,
        top_k_features=6,
        max_depth=5,
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=42,
        rfmode='regress',
    ):
        self.top_k_features = top_k_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.rfmode = rfmode
        self.model_ = None
        self.feature_names_ = None
        self.selected_feature_indices_ = None
        self.selected_feature_names_ = None
        self.shap_weights_ = None
        self.rules_ = None
        self.base_prediction_ = None

    def _resolve_feature_names(self, X, feature_names=None):
        if feature_names is None:
            return [f'feature_{i}' for i in range(X.shape[1])]
        return list(feature_names)

    def _resolve_shap_weights(self, X, shap_weights=None):
        if shap_weights is None:
            return np.ones(X.shape[1], dtype=float)
        weights = np.asarray(shap_weights, dtype=float).reshape(-1)
        if weights.shape[0] != X.shape[1]:
            raise ValueError(
                f'shap_weights length {weights.shape[0]} does not match n_features={X.shape[1]}'
            )
        weights = np.where(np.isfinite(weights), weights, 0.0)
        if np.all(weights <= 0):
            weights = np.ones(X.shape[1], dtype=float)
        return weights

    def _select_feature_indices(self, X, shap_weights):
        n_features = X.shape[1]
        if self.top_k_features is None:
            k = min(n_features, max(1, int(np.ceil(np.sqrt(n_features))) + 1))
        else:
            k = int(self.top_k_features)
            k = max(1, min(n_features, k))
        ranked = np.argsort(-shap_weights)
        return np.sort(ranked[:k])

    def fit(self, X, y, feature_names=None, shap_weights=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).reshape(-1)
        if X.ndim != 2:
            raise ValueError(f'Expected 2D X, got shape={X.shape}')
        if len(y) != X.shape[0]:
            raise ValueError(f'Expected y with {X.shape[0]} rows, got {len(y)}')

        self.feature_names_ = self._resolve_feature_names(X, feature_names)
        self.shap_weights_ = self._resolve_shap_weights(X, shap_weights)
        self.selected_feature_indices_ = self._select_feature_indices(X, self.shap_weights_)
        self.selected_feature_names_ = [self.feature_names_[i] for i in self.selected_feature_indices_]

        X_sel = X[:, self.selected_feature_indices_]
        if self.rfmode == 'classify':
            self.model_ = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )
        else:
            self.model_ = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )
        self.model_.fit(X_sel, y)
        self.base_prediction_ = float(np.mean(y)) if len(y) else 0.0
        self.rules_ = self._extract_rules()
        return self

    def _format_condition(self, feat_name, op, threshold):
        return f'{feat_name} {op} {threshold}'

    def _leaf_prediction(self, tree_, node):
        value = tree_.value[node]
        if self.rfmode == 'classify':
            class_idx = int(np.argmax(value[0]))
            return class_idx
        return float(value[0][0])

    def _extract_rules(self):
        if self.model_ is None:
            raise RuntimeError('Call fit() before get_rules().')

        tree_ = self.model_.tree_
        total_samples = max(1, int(tree_.n_node_samples[0]))
        rules = []

        def recurse(node, conds):
            feature_idx = tree_.feature[node]
            if feature_idx != _tree.TREE_UNDEFINED:
                feat_name = self.selected_feature_names_[feature_idx]
                threshold = float(tree_.threshold[node])
                recurse(
                    tree_.children_left[node],
                    conds + [self._format_condition(feat_name, '<=', threshold)],
                )
                recurse(
                    tree_.children_right[node],
                    conds + [self._format_condition(feat_name, '>', threshold)],
                )
                return

            if not conds:
                return

            prediction = self._leaf_prediction(tree_, node)
            support_count = int(tree_.n_node_samples[node])
            support_ratio = support_count / total_samples
            deviation = abs(float(prediction) - float(self.base_prediction_))
            importance = support_ratio * max(deviation, 1e-12)
            rules.append({
                'rule': ' & '.join(conds),
                'importance': float(importance),
                'weighted_importance': float(importance),
                'support': float(support_ratio),
                'support_count': support_count,
                'prediction': float(prediction),
                'depth': int(len(conds)),
                'component_type': 'rule',
            })

        recurse(0, [])
        if not rules:
            return pd.DataFrame(columns=[
                'rule', 'importance', 'weighted_importance', 'support',
                'support_count', 'prediction', 'depth', 'component_type'
            ])
        df = pd.DataFrame(rules)
        return df.sort_values(
            ['importance', 'support', 'depth'], ascending=[False, False, True]
        ).reset_index(drop=True)

    def get_rules(self):
        if self.rules_ is None:
            raise RuntimeError('Call fit() before get_rules().')
        return self.rules_.copy()

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError('Call fit() before predict().')
        X = np.asarray(X, dtype=np.float32)
        return self.model_.predict(X[:, self.selected_feature_indices_])
