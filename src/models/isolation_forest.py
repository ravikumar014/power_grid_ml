"""
Isolation Forest baseline for power-grid anomaly detection.

Experimental protocol
---------------------
1. Load temporally split feature matrices.
2. Fit Isolation Forest on TRAIN only.
3. Generate anomaly scores on TRAIN and VALIDATION.
4. Select an operating threshold using VALIDATION.
5. Evaluate the fixed detector on TEST.
6. Save scores, predictions, model, and evaluation metadata.

This module deliberately keeps metadata separate from the feature matrix.
"""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


                                                                        
               
                                                                        

FEATURE_DIR = Path(
    "data/features/final"
)

RESULT_DIR = Path(
    "data/results/baseline"
)

MODEL_DIR = Path(
    "models"
)


class IsolationForestBaseline:
    """
    Isolation Forest anomaly detector.

    The model is fitted only on the training feature matrix.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        contamination: str | float = "auto",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:

        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.is_fitted_ = False
        self.threshold_ = None

                                                                        
         
                                                                        

    def fit(
        self,
        X_train: pd.DataFrame,
    ) -> "IsolationForestBaseline":
        """
        Fit Isolation Forest on training data only.
        """

        if X_train.empty:
            raise ValueError(
                "Training feature matrix is empty."
            )

        X = self._validate_features(
            X_train
        )

        self.model.fit(X)

        self.is_fitted_ = True

        return self

                                                                        
           
                                                                        

    def score(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate anomaly scores.

        sklearn's decision_function assigns lower values to
        anomalous observations.

        We invert the sign so that:

            larger score = more anomalous
        """

        self._check_fitted()

        X = self._validate_features(
            X
        )

        scores = -self.model.decision_function(
            X
        )

        return scores

                                                                        
               
                                                                        

    def select_threshold(
        self,
        validation_scores: np.ndarray,
        quantile: float = 0.95,
    ) -> float:
        """
        Select anomaly threshold using validation scores.

        By default, observations above the 95th percentile of the
        validation anomaly-score distribution are treated as anomalous.

        IMPORTANT:
        This threshold is selected from validation data only and is
        subsequently frozen before evaluating the test set.
        """

        if validation_scores.size == 0:
            raise ValueError(
                "Validation score array is empty."
            )

        if not (
            0.0 < quantile < 1.0
        ):
            raise ValueError(
                "Threshold quantile must be between 0 and 1."
            )

        self.threshold_ = float(
            np.quantile(
                validation_scores,
                quantile,
            )
        )

        return self.threshold_

                                                                        
             
                                                                        

    def predict(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Convert anomaly scores into binary predictions.

        1 = anomaly
        0 = normal
        """

        if self.threshold_ is None:
            raise RuntimeError(
                "Threshold has not been selected. "
                "Call select_threshold() first."
            )

        return (
            scores >= self.threshold_
        ).astype(int)

                                                                        
                
                                                                        

    @staticmethod
    def _validate_features(
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Validate and convert feature dataframe to numpy array.
        """

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "Expected pandas DataFrame."
            )

        if X.empty:
            raise ValueError(
                "Feature dataframe is empty."
            )

        values = X.to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():

            raise ValueError(
                "Feature matrix contains NaN "
                "or infinite values."
            )

        return values

    def _check_fitted(self) -> None:

        if not self.is_fitted_:

            raise RuntimeError(
                "Isolation Forest has not been fitted."
            )

                                                                        
          
                                                                        

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save the fitted detector.
        """

        self._check_fitted()

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self,
            path,
        )

                                                                        
          
                                                                        

    @staticmethod
    def load(
        path: str | Path,
    ) -> "IsolationForestBaseline":

        detector = joblib.load(
            path
        )

        if not isinstance(
            detector,
            IsolationForestBaseline,
        ):
            raise TypeError(
                "Saved object is not an "
                "IsolationForestBaseline."
            )

        return detector


                                                                        
              
                                                                        


def load_split_data(
    entity: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load train/validation/test features and metadata.

    Returns
    -------
    X_train
    X_validation
    X_test
    metadata_train
    metadata_validation
    metadata_test
    """

    if entity == "bus":

        feature_prefix = "bus"
        metadata_prefix = "bus"

    elif entity == "line":

        feature_prefix = "line"
        metadata_prefix = "line"

    else:

        raise ValueError(
            "entity must be 'bus' or 'line'."
        )

    X_train = pd.read_csv(
        FEATURE_DIR
        / f"{feature_prefix}_train_features.csv"
    )

    X_validation = pd.read_csv(
        FEATURE_DIR
        / f"{feature_prefix}_validation_features.csv"
    )

    X_test = pd.read_csv(
        FEATURE_DIR
        / f"{feature_prefix}_test_features.csv"
    )

    metadata_train = pd.read_csv(
        FEATURE_DIR
        / f"{metadata_prefix}_train_metadata.csv"
    )

    metadata_validation = pd.read_csv(
        FEATURE_DIR
        / f"{metadata_prefix}_validation_metadata.csv"
    )

    metadata_test = pd.read_csv(
        FEATURE_DIR
        / f"{metadata_prefix}_test_metadata.csv"
    )

    return (
        X_train,
        X_validation,
        X_test,
        metadata_train,
        metadata_validation,
        metadata_test,
    )


                                                                        
                 
                                                                        


def create_results_dataframe(
    metadata: pd.DataFrame,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Combine metadata, anomaly scores and predictions.
    """

    if len(metadata) != len(scores):

        raise ValueError(
            "Metadata and score lengths do not match."
        )

    results = metadata.copy()

    results[
        "anomaly_score"
    ] = scores

    results[
        "is_anomaly"
    ] = predictions

    return results


                                                                        
                    
                                                                        


def save_summary(
    entity: str,
    detector: IsolationForestBaseline,
    train_scores: np.ndarray,
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    test_predictions: np.ndarray,
) -> None:
    """
    Save experiment summary.
    """

    summary = {

        "entity": entity,

        "n_estimators": detector.n_estimators,

        "contamination": detector.contamination,

        "random_state": detector.random_state,

        "threshold": detector.threshold_,

        "train_samples": int(
            len(train_scores)
        ),

        "validation_samples": int(
            len(validation_scores)
        ),

        "test_samples": int(
            len(test_scores)
        ),

        "train_score_mean": float(
            np.mean(train_scores)
        ),

        "validation_score_mean": float(
            np.mean(validation_scores)
        ),

        "test_score_mean": float(
            np.mean(test_scores)
        ),

        "test_anomalies": int(
            test_predictions.sum()
        ),

        "test_anomaly_fraction": float(
            test_predictions.mean()
        ),
    }

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULT_DIR
        / f"isolation_forest_{entity}_summary.json",
        "w",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )


                                                                        
                     
                                                                        


def run_experiment(
    entity: str,
) -> None:

    print()
    print("=" * 70)
    print(
        f"ISOLATION FOREST BASELINE — "
        f"{entity.upper()}"
    )
    print("=" * 70)

                                                                        
               
                                                                        

    (
        X_train,
        X_validation,
        X_test,
        metadata_train,
        metadata_validation,
        metadata_test,
    ) = load_split_data(
        entity
    )

    print()
    print("Dataset sizes:")
    print(
        f"Train       : {X_train.shape}"
    )
    print(
        f"Validation  : {X_validation.shape}"
    )
    print(
        f"Test        : {X_test.shape}"
    )

                                                                        
         
                                                                        

    detector = IsolationForestBaseline(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    print()
    print(
        "Fitting Isolation Forest on TRAIN only..."
    )

    detector.fit(
        X_train
    )

                                                                        
            
                                                                        

    train_scores = detector.score(
        X_train
    )

    validation_scores = detector.score(
        X_validation
    )

    test_scores = detector.score(
        X_test
    )

                                                                        
                         
                                                                        

    threshold = detector.select_threshold(
        validation_scores,
        quantile=0.95,
    )

    print()
    print(
        f"Validation threshold: "
        f"{threshold:.6f}"
    )

                                                                        
                 
                                                                        

    train_predictions = detector.predict(
        train_scores
    )

    validation_predictions = detector.predict(
        validation_scores
    )

    test_predictions = detector.predict(
        test_scores
    )

                                                                        
             
                                                                        

    train_results = create_results_dataframe(
        metadata_train,
        train_scores,
        train_predictions,
    )

    validation_results = create_results_dataframe(
        metadata_validation,
        validation_scores,
        validation_predictions,
    )

    test_results = create_results_dataframe(
        metadata_test,
        test_scores,
        test_predictions,
    )

                                                                        
          
                                                                        

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_results.to_csv(
        RESULT_DIR
        / f"isolation_forest_{entity}_train.csv",
        index=False,
    )

    validation_results.to_csv(
        RESULT_DIR
        / f"isolation_forest_{entity}_validation.csv",
        index=False,
    )

    test_results.to_csv(
        RESULT_DIR
        / f"isolation_forest_{entity}_test.csv",
        index=False,
    )

    model_path = (
        MODEL_DIR
        / f"isolation_forest_{entity}.joblib"
    )

    detector.save(
        model_path
    )

    save_summary(
        entity,
        detector,
        train_scores,
        validation_scores,
        test_scores,
        test_predictions,
    )

                                                                        
            
                                                                        

    print()
    print(
        "-" * 70
    )

    print(
        "ANOMALY COUNTS"
    )

    print(
        f"Train      : "
        f"{train_predictions.sum()} / "
        f"{len(train_predictions)} "
        f"({train_predictions.mean():.4f})"
    )

    print(
        f"Validation : "
        f"{validation_predictions.sum()} / "
        f"{len(validation_predictions)} "
        f"({validation_predictions.mean():.4f})"
    )

    print(
        f"Test       : "
        f"{test_predictions.sum()} / "
        f"{len(test_predictions)} "
        f"({test_predictions.mean():.4f})"
    )

    print()
    print(
        "Score statistics"
    )

    print(
        f"Train mean      : "
        f"{train_scores.mean():.6f}"
    )

    print(
        f"Validation mean : "
        f"{validation_scores.mean():.6f}"
    )

    print(
        f"Test mean       : "
        f"{test_scores.mean():.6f}"
    )

    print()
    print(
        f"Saved model to:\n"
        f"{model_path}"
    )

    print(
        f"Saved results to:\n"
        f"{RESULT_DIR}"
    )


                                                                        
      
                                                                        


if __name__ == "__main__":

    run_experiment(
        "bus"
    )

    run_experiment(
        "line"
    )

    print()
    print("=" * 70)
    print(
        "ISOLATION FOREST BASELINE COMPLETE"
    )
    print("=" * 70)