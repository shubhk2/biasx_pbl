"""Model training and fairness evaluation pipeline for backend /analyze requests."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from aif360.datasets import BinaryLabelDataset

from bias_calculator import BiasDetector


# Small adapter to ensure y is passed to scikit-learn estimators as a 1-D array.
class _RavelingEstimator(BaseEstimator, ClassifierMixin):
    """Wrap an sklearn estimator and ravel y before calling fit.

    This prevents DataConversionWarning when callers provide a column-vector
    shaped (n_samples, 1) for y (common inside aif360 datasets).
    """
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, np.ravel(y), **kwargs)
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        if hasattr(self.estimator_, 'predict_proba'):
            return self.estimator_.predict_proba(X)
        raise AttributeError('Underlying estimator has no predict_proba')

    def __getattr__(self, name):
        if name in ["estimator_", "classes_"]:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        if hasattr(self, "estimator_"):
            return getattr(self.estimator_, name)
        return getattr(self.estimator, name)


SUPPORTED_MODEL_TYPES = {
    "logistic_regression",
    "neural_network",
    "prejudice_remover",
    "adversarial_debiasing",
    "exponentiated_gradient_reduction",
}


def _as_binary(series: pd.Series, positive_value: Any = 1) -> pd.Series:
    """Map an arbitrary binary-like series to 0/1."""
    values = series.dropna().unique().tolist()
    if len(values) <= 2:
        if set(values).issubset({0, 1, True, False}):
            return series.astype(int)
        if positive_value in values:
            return (series == positive_value).astype(int)
        # Fallback: treat the second sorted unique value as positive.
        sorted_vals = sorted(values, key=lambda x: str(x))
        positive = sorted_vals[-1]
        return (series == positive).astype(int)
    raise ValueError("Label/protected column must be binary for model fairness analysis")


def _encode_features(
    df: pd.DataFrame,
    protected_attr: str,
    label_column: str,
    privileged_value: Any,
    unprivileged_value: Any,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Encode features for sklearn/AIF360 models while preserving binary protected attr."""
    if protected_attr not in df.columns:
        raise ValueError(f"Protected attribute '{protected_attr}' not found")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found")

    work_df = df.copy()
    y = _as_binary(work_df[label_column], positive_value=1)
    work_df[label_column] = y

    # Keep protected attribute binary with explicit mapping: privileged=1, unprivileged=0.
    protected_raw = work_df[protected_attr]
    mapped = pd.Series(np.nan, index=protected_raw.index)
    mapped[protected_raw == privileged_value] = 1
    mapped[protected_raw == unprivileged_value] = 0
    if mapped.isna().any():
        # Fallback for dtype mismatch between payload and dataframe (e.g., "1" vs 1).
        protected_str = protected_raw.astype(str)
        mapped = pd.Series(np.nan, index=protected_raw.index)
        mapped[protected_str == str(privileged_value)] = 1
        mapped[protected_str == str(unprivileged_value)] = 0
    if mapped.isna().any():
        raise ValueError(
            f"Protected attribute '{protected_attr}' contains values outside configured groups "
            f"({privileged_value}, {unprivileged_value})"
        )
    work_df[protected_attr] = mapped.astype(int)

    feature_df = work_df.drop(columns=[label_column]).copy()
    for col in feature_df.columns:
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())
        else:
            codes, _ = pd.factorize(feature_df[col].astype(str), sort=True)
            feature_df[col] = codes

    return feature_df, y


def _evaluate_predictions(
    df_with_id: pd.DataFrame,
    config: Dict[str, Any],
    test_indices: np.ndarray,
    y_pred: np.ndarray,
    selected_metrics: List[str],
) -> Dict[str, Any]:
    pred_df = pd.DataFrame({
        "__row_id__": test_indices.astype(int),
        "y_pred": y_pred.astype(int),
    })
    detector = BiasDetector(
        dataset=df_with_id,
        protected_attr=config["protected_attr"],
        label_column=config["label_column"],
        privileged_value=config["privileged_value"],
        unprivileged_value=config["unprivileged_value"],
        dataset_pred=pred_df,
        detection_type="Model Bias Detection",
        id_column="__row_id__",
        pred_label_col="y_pred",
    )
    return detector.calculate_metrics(selected_metrics)


def _train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    model = LogisticRegression(max_iter=1000, random_state=42)
    # Ensure y is 1-D when fitting sklearn estimators
    model.fit(X_train, np.ravel(y_train))
    return model.predict(X_test)


def _train_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42)
    # Ensure y is 1-D when fitting sklearn estimators
    model.fit(X_train, np.ravel(y_train))
    return model.predict(X_test)


def _run_prejudice_remover(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> np.ndarray:
    try:
        from aif360.algorithms.inprocessing import PrejudiceRemover
    except Exception as exc:
        raise RuntimeError("PrejudiceRemover is not available in this aif360 installation") from exc

    train_dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=train_df,
        label_names=[config["label_column"]],
        protected_attribute_names=[config["protected_attr"]],
    )
    test_dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=test_df,
        label_names=[config["label_column"]],
        protected_attribute_names=[config["protected_attr"]],
    )

    eta = float(config.get("eta", 25.0))
    model = PrejudiceRemover(sensitive_attr=config["protected_attr"], eta=eta)
    model.fit(train_dataset)
    pred = model.predict(test_dataset)
    return pred.labels.ravel().astype(int)


def _run_adversarial_debiasing(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> np.ndarray:
    try:
        from aif360.algorithms.inprocessing import AdversarialDebiasing
    except Exception as exc:
        raise RuntimeError("AdversarialDebiasing is not available in this aif360 installation") from exc

    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError("TensorFlow is required for AdversarialDebiasing") from exc

    tf.compat.v1.disable_eager_execution()
    train_dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=train_df,
        label_names=[config["label_column"]],
        protected_attribute_names=[config["protected_attr"]],
    )
    test_dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=test_df,
        label_names=[config["label_column"]],
        protected_attribute_names=[config["protected_attr"]],
    )

    sess = tf.compat.v1.Session()
    try:
        model = AdversarialDebiasing(
            privileged_groups=({config["protected_attr"]: 1},),
            unprivileged_groups=({config["protected_attr"]: 0},),
            scope_name="adversarial_debiasing",
            debias=True,
            sess=sess,
        )
        model.fit(train_dataset)
        pred = model.predict(test_dataset)
        return pred.labels.ravel().astype(int)
    finally:
        sess.close()


def _run_exponentiated_gradient_reduction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> np.ndarray:
    """Run Exponentiated Gradient Reduction using a LogisticRegression estimator.

    Uses aif360.algorithms.inprocessing.ExponentiatedGradientReduction with a
    sklearn LogisticRegression() estimator. The `constraints` can be provided
    via config (default: "DemographicParity").
    """
    try:
        from aif360.algorithms.inprocessing import ExponentiatedGradientReduction
    except Exception as exc:
        raise RuntimeError("ExponentiatedGradientReduction is not available in this aif360 installation") from exc

    try:
        from sklearn.linear_model import LogisticRegression as SklearnLR
    except Exception:
        # sklearn should already be available; raise similar error if not
        raise RuntimeError("scikit-learn is required for ExponentiatedGradientReduction")

    train_dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=train_df,
        label_names=[config["label_column"]],
        protected_attribute_names=[config["protected_attr"]],
    )
    test_dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=test_df,
        label_names=[config["label_column"]],
        protected_attribute_names=[config["protected_attr"]],
    )

    constraints = config.get("constraints") or config.get("eg_constraints") or "DemographicParity"
    # Instantiate the ExponentiatedGradientReduction with a logistic regression estimator
    # Use a LogisticRegression estimator with matching solver settings to the baseline
    # Ravel the AIF360 dataset labels here to prevent fairlearn from producing DataConversionWarning inside
    base_est = SklearnLR(max_iter=1000, random_state=42)
    model = ExponentiatedGradientReduction(
        estimator=_RavelingEstimator(base_est),
        constraints=constraints,
    )
    
    # Fix AIF360 2D labels issue by unravelling before fit
    train_dataset.labels = train_dataset.labels.ravel()
    
    model.fit(train_dataset)
    pred = model.predict(test_dataset)
    return pred.labels.ravel().astype(int)


def _build_comparison_table(model_results: Dict[str, Dict[str, Any]], selected_metrics: List[str]) -> Dict[str, Any]:
    columns = ["model", "bias_detected"] + selected_metrics
    rows = []
    for model_name, result in model_results.items():
        row = {
            "model": model_name,
            "bias_detected": bool(result.get("summary", {}).get("bias_detected", False)),
        }
        metric_map = result.get("metrics", {})
        for metric in selected_metrics:
            row[metric] = metric_map.get(metric, {}).get("value")
        rows.append(row)
    return {"columns": columns, "rows": rows}


def run_model_pipeline(df: pd.DataFrame, config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """Train/evaluate selected model type and return fairness results + comparison table.

    For fairness in-processing methods, paired baseline is also evaluated:
    - prejudice_remover => logistic_regression + prejudice_remover
    - adversarial_debiasing => neural_network + adversarial_debiasing
    - exponentiated_gradient_reduction => logistic_regression + exponentiated_gradient_reduction
    """
    model_type = (model_type or "logistic_regression").strip().lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Supported: {sorted(SUPPORTED_MODEL_TYPES)}"
        )

    selected_metrics = config.get("selected_metrics", [])
    test_size = float(config.get("test_size", 0.3))
    random_state = int(config.get("random_state", 42))

    # Use only the uploaded dataset and add a stable row id for post-split alignment.
    df_with_id = df.copy().reset_index(drop=True)
    df_with_id["__row_id__"] = np.arange(len(df_with_id), dtype=int)

    X, y = _encode_features(
        df_with_id,
        config["protected_attr"],
        config["label_column"],
        config["privileged_value"],
        config["unprivileged_value"],
    )

    # Shared split for all candidate models so comparisons are on the same test data.
    all_indices = np.arange(len(df_with_id))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y.values,
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)

    prepared_train_df = X_train.copy()
    prepared_train_df[config["label_column"]] = y_train.values

    prepared_test_df = X_test.copy()
    prepared_test_df[config["label_column"]] = y.iloc[test_idx].reset_index(drop=True).values

    run_order: List[str]
    predictions: Dict[str, np.ndarray] = {}

    if model_type == "logistic_regression":
        run_order = ["logistic_regression"]
        predictions["logistic_regression"] = _train_logistic_regression(X_train, y_train, X_test)
    elif model_type == "neural_network":
        run_order = ["neural_network"]
        predictions["neural_network"] = _train_neural_network(X_train, y_train, X_test)
    elif model_type == "prejudice_remover":
        run_order = ["logistic_regression", "prejudice_remover"]
        predictions["logistic_regression"] = _train_logistic_regression(X_train, y_train, X_test)
        predictions["prejudice_remover"] = _run_prejudice_remover(prepared_train_df, prepared_test_df, config)
    elif model_type == "adversarial_debiasing":
        run_order = ["neural_network", "adversarial_debiasing"]
        predictions["neural_network"] = _train_neural_network(X_train, y_train, X_test)
        predictions["adversarial_debiasing"] = _run_adversarial_debiasing(prepared_train_df, prepared_test_df, config)
    elif model_type == "exponentiated_gradient_reduction":
        run_order = ["logistic_regression", "exponentiated_gradient_reduction"]
        predictions["logistic_regression"] = _train_logistic_regression(X_train, y_train, X_test)
        predictions["exponentiated_gradient_reduction"] = _run_exponentiated_gradient_reduction(prepared_train_df, prepared_test_df, config)
    else:
        # Fallback: if unknown type slipped through, default to logistic regression baseline
        run_order = ["logistic_regression"]
        predictions["logistic_regression"] = _train_logistic_regression(X_train, y_train, X_test)

    model_results: Dict[str, Dict[str, Any]] = {}
    for name in run_order:
        model_results[name] = _evaluate_predictions(
            df_with_id,
            config,
            test_idx,
            predictions[name],
            selected_metrics,
        )

    primary_result = model_results[model_type]
    comparison_table = _build_comparison_table(model_results, selected_metrics)

    return {
        "primary_model": model_type,
        "results": primary_result,
        "comparison_table": comparison_table,
        "all_model_results": model_results,
    }

