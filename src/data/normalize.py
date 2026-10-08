"""
Normalization utilities for the power-grid ML pipeline.

The scaler is fitted ONLY on the training dataset and then reused
unchanged for validation and test data.

This prevents information leakage from future observations.

Supported methods
-----------------
standard:
    z = (x - mean) / std

minmax:
    x' = (x - min) / (max - min)

robust:
    x' = (x - median) / IQR

The default method is standard scaling.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class NormalizationReport:
    """
    Records normalization information.
    """

    method: str

    fitted: bool = False

    feature_columns: list[str] = field(
        default_factory=list
    )

    train_rows: int = 0
    validation_rows: int = 0
    test_rows: int = 0

    def summary(self) -> str:
        """
        Return a human-readable normalization report.
        """

        lines = [
            "",
            "=" * 70,
            "POWER GRID NORMALIZATION REPORT",
            "=" * 70,
            f"Method             : {self.method}",
            f"Fitted             : {self.fitted}",
            f"Number of features : {len(self.feature_columns)}",
            f"Train rows         : {self.train_rows}",
            f"Validation rows    : {self.validation_rows}",
            f"Test rows          : {self.test_rows}",
            "",
            "Features:",
        ]

        for feature in self.feature_columns:
            lines.append(
                f"  - {feature}"
            )

        lines.append("=" * 70)

        return "\n".join(lines)


class PowerGridNormalizer:
    """
    Leakage-safe normalizer for power-grid measurements.

    Parameters
    ----------
    method:
        One of:
            "standard"
            "minmax"
            "robust"

    exclude_columns:
        Columns that should never be normalized.

        Typical examples:
            timestamp
            bus_id
            line_id
            anomaly labels
    """

    SUPPORTED_METHODS = {
        "standard",
        "minmax",
        "robust",
    }

    DEFAULT_EXCLUDED_COLUMNS = {
        "timestamp",
        "bus_id",
        "line_id",
        "hour",
        "day_of_week",
        "day_of_year",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    }

    def __init__(
        self,
        method: str = "standard",
        exclude_columns: set[str] | None = None,
    ) -> None:

        method = method.lower()

        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported normalization method "
                f"'{method}'. Supported methods: "
                f"{sorted(self.SUPPORTED_METHODS)}"
            )

        self.method = method

        if exclude_columns is None:
            self.exclude_columns = (
                self.DEFAULT_EXCLUDED_COLUMNS.copy()
            )
        else:
            self.exclude_columns = set(
                exclude_columns
            )

        self.feature_columns: list[str] = []

        self.parameters: dict[str, dict[str, float]] = {}

        self.fitted = False

                                                                        
         
                                                                        

    def fit(
        self,
        train_data: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> "PowerGridNormalizer":
        """
        Learn normalization parameters from TRAINING data only.

        Parameters
        ----------
        train_data:
            Training dataframe.

        feature_columns:
            Explicit columns to normalize.

            If None, all numerical columns not present in
            exclude_columns are used.
        """

        if train_data.empty:
            raise ValueError(
                "Cannot fit normalizer on empty training data."
            )

        data = train_data.copy()

                                                                        
                                    
                                                                        

        if feature_columns is None:

            feature_columns = [
                column
                for column in data.columns
                if (
                    pd.api.types.is_numeric_dtype(
                        data[column]
                    )
                    and column not in self.exclude_columns
                )
            ]

        if not feature_columns:
            raise ValueError(
                "No numerical feature columns available "
                "for normalization."
            )

                                                                        
                                   
                                                                        

        missing = set(feature_columns) - set(
            data.columns
        )

        if missing:
            raise ValueError(
                "Feature columns not found in training "
                f"data: {sorted(missing)}"
            )

        self.feature_columns = list(
            feature_columns
        )

                                                                        
                           
                                                                        

        self.parameters = {}

        for column in self.feature_columns:

            values = pd.to_numeric(
                data[column],
                errors="coerce",
            )

            if values.isna().any():
                raise ValueError(
                    f"Feature '{column}' contains missing "
                    "or non-numeric values. "
                    "Preprocess the data first."
                )

            values_array = values.to_numpy(
                dtype=float
            )

            if not np.isfinite(
                values_array
            ).all():

                raise ValueError(
                    f"Feature '{column}' contains "
                    "infinite values."
                )

            if self.method == "standard":

                mean = float(
                    np.mean(values_array)
                )

                std = float(
                    np.std(
                        values_array,
                        ddof=0,
                    )
                )

                                                                 
                if std < 1e-12:
                    std = 1.0

                self.parameters[column] = {
                    "mean": mean,
                    "scale": std,
                }

            elif self.method == "minmax":

                minimum = float(
                    np.min(values_array)
                )

                maximum = float(
                    np.max(values_array)
                )

                scale = maximum - minimum

                if scale < 1e-12:
                    scale = 1.0

                self.parameters[column] = {
                    "min": minimum,
                    "max": maximum,
                    "scale": scale,
                }

            elif self.method == "robust":

                median = float(
                    np.median(values_array)
                )

                q1 = float(
                    np.percentile(
                        values_array,
                        25,
                    )
                )

                q3 = float(
                    np.percentile(
                        values_array,
                        75,
                    )
                )

                iqr = q3 - q1

                if iqr < 1e-12:
                    iqr = 1.0

                self.parameters[column] = {
                    "median": median,
                    "scale": iqr,
                }

        self.fitted = True

        return self

                                                                        
               
                                                                        

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform data using parameters learned during fit().
        """

        if not self.fitted:
            raise RuntimeError(
                "Normalizer has not been fitted. "
                "Call fit() using training data first."
            )

        data = dataframe.copy()

        missing = (
            set(self.feature_columns)
            - set(data.columns)
        )

        if missing:
            raise ValueError(
                "Input data is missing feature columns: "
                f"{sorted(missing)}"
            )

        for column in self.feature_columns:

            values = pd.to_numeric(
                data[column],
                errors="coerce",
            )

            if values.isna().any():
                raise ValueError(
                    f"Feature '{column}' contains missing "
                    "or non-numeric values."
                )

            params = self.parameters[column]

            if self.method == "standard":

                data[column] = (
                    values - params["mean"]
                ) / params["scale"]

            elif self.method == "minmax":

                data[column] = (
                    values - params["min"]
                ) / params["scale"]

            elif self.method == "robust":

                data[column] = (
                    values - params["median"]
                ) / params["scale"]

        return data

                                                                        
                     
                                                                        

    def fit_transform(
        self,
        train_data: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit the normalizer on training data and transform it.
        """

        self.fit(
            train_data,
            feature_columns=feature_columns,
        )

        return self.transform(
            train_data
        )

                                                                        
                       
                                                                        

    def inverse_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert normalized features back to their original scale.
        """

        if not self.fitted:
            raise RuntimeError(
                "Normalizer has not been fitted."
            )

        data = dataframe.copy()

        for column in self.feature_columns:

            params = self.parameters[column]

            if self.method == "standard":

                data[column] = (
                    data[column]
                    * params["scale"]
                    + params["mean"]
                )

            elif self.method == "minmax":

                data[column] = (
                    data[column]
                    * params["scale"]
                    + params["min"]
                )

            elif self.method == "robust":

                data[column] = (
                    data[column]
                    * params["scale"]
                    + params["median"]
                )

        return data

                                                                        
          
                                                                        

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save the fitted normalizer.
        """

        if not self.fitted:
            raise RuntimeError(
                "Cannot save an unfitted normalizer."
            )

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "wb"
        ) as file:

            pickle.dump(
                self,
                file,
            )

                                                                        
          
                                                                        

    @staticmethod
    def load(
        path: str | Path,
    ) -> "PowerGridNormalizer":
        """
        Load a previously fitted normalizer.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Normalizer file not found: {path}"
            )

        with path.open(
            "rb"
        ) as file:

            normalizer = pickle.load(
                file
            )

        if not isinstance(
            normalizer,
            PowerGridNormalizer,
        ):
            raise TypeError(
                "Saved object is not a "
                "PowerGridNormalizer."
            )

        return normalizer


def normalize_saved_splits(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    output_directory: str | Path = (
        "data/processed/normalized"
    ),
    method: str = "standard",
) -> NormalizationReport:
    """
    Normalize train/validation/test datasets.

    CRITICAL:
        The normalizer is fitted ONLY on train_data.
    """

    train_path = Path(train_path)
    validation_path = Path(validation_path)
    test_path = Path(test_path)

    for path in [
        train_path,
        validation_path,
        test_path,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

    train = pd.read_csv(
        train_path,
        parse_dates=["timestamp"],
    )

    validation = pd.read_csv(
        validation_path,
        parse_dates=["timestamp"],
    )

    test = pd.read_csv(
        test_path,
        parse_dates=["timestamp"],
    )

                                                                    
                                                 
                                                                    

    normalizer = PowerGridNormalizer(
        method=method
    )

    normalizer.fit(
        train
    )

                                                                    
                                                             
                                                                    

    train_normalized = normalizer.transform(
        train
    )

    validation_normalized = normalizer.transform(
        validation
    )

    test_normalized = normalizer.transform(
        test
    )

                                                                    
                    
                                                                    

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_normalized.to_csv(
        output_directory / "train.csv",
        index=False,
    )

    validation_normalized.to_csv(
        output_directory / "validation.csv",
        index=False,
    )

    test_normalized.to_csv(
        output_directory / "test.csv",
        index=False,
    )

                                                                    
                             
                                                                    

    normalizer.save(
        output_directory / "normalizer.pkl"
    )

                                                                    
             
                                                                    

    report = NormalizationReport(
        method=method,
        fitted=True,
        feature_columns=normalizer.feature_columns,
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
    )

    return report


if __name__ == "__main__":

    report = normalize_saved_splits(
        train_path=(
            "data/processed/train.csv"
        ),
        validation_path=(
            "data/processed/validation.csv"
        ),
        test_path=(
            "data/processed/test.csv"
        ),
        output_directory=(
            "data/processed/normalized"
        ),
        method="standard",
    )

    print(
        report.summary()
    )