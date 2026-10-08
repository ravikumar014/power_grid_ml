"""
Validation utilities for power-grid measurement datasets.

This module performs structural and basic physical validation before
data enter the preprocessing and ML pipeline.

Validation categories
---------------------
1. Schema validation
2. Timestamp validation
3. Missing-value validation
4. Duplicate-record validation
5. Numerical validation
6. Basic electrical plausibility checks

This module does NOT modify the input data.
It only reports validation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ValidationIssue:
    """
    Represents one validation problem.
    """

    category: str
    message: str
    count: int = 1
    severity: str = "error"


@dataclass
class ValidationReport:
    """
    Complete validation report.
    """

    dataset_name: str

    total_rows: int = 0
    total_columns: int = 0

    issues: list[ValidationIssue] = field(
        default_factory=list
    )

    passed: bool = True

    def add_issue(
        self,
        category: str,
        message: str,
        count: int = 1,
        severity: str = "error",
    ) -> None:

        self.issues.append(
            ValidationIssue(
                category=category,
                message=message,
                count=count,
                severity=severity,
            )
        )

        if severity == "error":
            self.passed = False

    def summary(self) -> str:
        """
        Return a human-readable validation summary.
        """

        lines = [
            "",
            "=" * 70,
            "POWER GRID DATA VALIDATION REPORT",
            "=" * 70,
            f"Dataset       : {self.dataset_name}",
            f"Rows          : {self.total_rows}",
            f"Columns       : {self.total_columns}",
            f"Validation    : {'PASSED' if self.passed else 'FAILED'}",
            "",
        ]

        if not self.issues:

            lines.append(
                "No validation issues detected."
            )

        else:

            lines.append(
                f"Issues detected: {len(self.issues)}"
            )

            lines.append("")

            for issue in self.issues:

                lines.append(
                    f"[{issue.severity.upper()}] "
                    f"{issue.category}: "
                    f"{issue.message} "
                    f"(count={issue.count})"
                )

        lines.append("=" * 70)

        return "\n".join(lines)


class MeasurementValidator:
    """
    Validator for bus and line measurement datasets.
    """

    BUS_REQUIRED_COLUMNS = {
        "timestamp",
        "bus_id",
        "voltage_pu",
        "angle_deg",
        "active_power_mw",
        "reactive_power_mvar",
    }

    LINE_REQUIRED_COLUMNS = {
        "timestamp",
        "line_id",
        "current_from_ka",
        "current_to_ka",
        "loading_percent",
    }

    def __init__(
        self,
        voltage_min: float = 0.0,
        voltage_max: float = 2.0,
        frequency_min: float = 45.0,
        frequency_max: float = 55.0,
        loading_max: float = 1000.0,
    ) -> None:

        self.voltage_min = voltage_min
        self.voltage_max = voltage_max

        self.frequency_min = frequency_min
        self.frequency_max = frequency_max

        self.loading_max = loading_max

                                                                        
                        
                                                                        

    def validate_bus_measurements(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str = "bus_measurements",
    ) -> ValidationReport:
        """
        Validate bus-level measurements.
        """

        report = ValidationReport(
            dataset_name=dataset_name,
            total_rows=len(dataframe),
            total_columns=len(dataframe.columns),
        )

                                                                        
                       
                                                                        

        if dataframe.empty:

            report.add_issue(
                category="structure",
                message="Dataset is empty.",
            )

            return report

                                                                        
                          
                                                                        

        missing_columns = (
            self.BUS_REQUIRED_COLUMNS
            - set(dataframe.columns)
        )

        if missing_columns:

            report.add_issue(
                category="schema",
                message=(
                    "Missing required columns: "
                    f"{sorted(missing_columns)}"
                ),
                count=len(missing_columns),
            )

                                                       
            return report

                                                                        
                              
                                                                        

        self._validate_timestamps(
            dataframe,
            report,
        )

                                                                        
                       
                                                                        

        self._validate_ids(
            dataframe,
            id_column="bus_id",
            report=report,
        )

                                                                        
                        
                                                                        

        self._validate_missing_values(
            dataframe,
            report,
        )

                                                                        
                           
                                                                        

        self._validate_duplicates(
            dataframe,
            keys=[
                "timestamp",
                "bus_id",
            ],
            report=report,
        )

                                                                        
                          
                                                                        

        numerical_columns = [
            "voltage_pu",
            "angle_deg",
            "active_power_mw",
            "reactive_power_mvar",
        ]

        self._validate_numeric_columns(
            dataframe,
            numerical_columns,
            report,
        )

                                                                        
                              
                                                                        

        self._validate_range(
            dataframe,
            column="voltage_pu",
            minimum=self.voltage_min,
            maximum=self.voltage_max,
            report=report,
            category="electrical",
        )

        return report

                                                                        
                     
                                                                        

    def validate_line_measurements(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str = "line_measurements",
    ) -> ValidationReport:
        """
        Validate line-level measurements.
        """

        report = ValidationReport(
            dataset_name=dataset_name,
            total_rows=len(dataframe),
            total_columns=len(dataframe.columns),
        )

        if dataframe.empty:

            report.add_issue(
                category="structure",
                message="Dataset is empty.",
            )

            return report

                                                                        
                          
                                                                        

        missing_columns = (
            self.LINE_REQUIRED_COLUMNS
            - set(dataframe.columns)
        )

        if missing_columns:

            report.add_issue(
                category="schema",
                message=(
                    "Missing required columns: "
                    f"{sorted(missing_columns)}"
                ),
                count=len(missing_columns),
            )

            return report

                                                                        
                              
                                                                        

        self._validate_timestamps(
            dataframe,
            report,
        )

                                                                        
                       
                                                                        

        self._validate_ids(
            dataframe,
            id_column="line_id",
            report=report,
        )

                                                                        
                        
                                                                        

        self._validate_missing_values(
            dataframe,
            report,
        )

                                                                        
                           
                                                                        

        self._validate_duplicates(
            dataframe,
            keys=[
                "timestamp",
                "line_id",
            ],
            report=report,
        )

                                                                        
                          
                                                                        

        numerical_columns = [
            "current_from_ka",
            "current_to_ka",
            "loading_percent",
        ]

        self._validate_numeric_columns(
            dataframe,
            numerical_columns,
            report,
        )

                                                                        
                      
                                                                        

        self._validate_range(
            dataframe,
            column="loading_percent",
            minimum=0.0,
            maximum=self.loading_max,
            report=report,
            category="electrical",
        )

                                                                        
                        
                                                                        

        self._validate_range(
            dataframe,
            column="current_from_ka",
            minimum=0.0,
            maximum=np.inf,
            report=report,
            category="electrical",
        )

        self._validate_range(
            dataframe,
            column="current_to_ka",
            minimum=0.0,
            maximum=np.inf,
            report=report,
            category="electrical",
        )

        return report

                                                                        
                          
                                                                        

    @staticmethod
    def _validate_timestamps(
        dataframe: pd.DataFrame,
        report: ValidationReport,
    ) -> None:

        timestamps = pd.to_datetime(
            dataframe["timestamp"],
            errors="coerce",
        )

        invalid = timestamps.isna()

        if invalid.any():

            report.add_issue(
                category="timestamp",
                message="Invalid timestamp values detected.",
                count=int(invalid.sum()),
            )

            return

        if not timestamps.is_monotonic_increasing:

            report.add_issue(
                category="timestamp",
                message=(
                    "Timestamps are not monotonically increasing."
                ),
            )

                                                                        
                   
                                                                        

    @staticmethod
    def _validate_ids(
        dataframe: pd.DataFrame,
        id_column: str,
        report: ValidationReport,
    ) -> None:

        ids = dataframe[id_column]

        invalid = ids.isna()

        if invalid.any():

            report.add_issue(
                category="identifier",
                message=(
                    f"Missing values detected in '{id_column}'."
                ),
                count=int(invalid.sum()),
            )

        if not pd.api.types.is_numeric_dtype(ids):

            report.add_issue(
                category="identifier",
                message=(
                    f"'{id_column}' is not numeric."
                ),
                severity="warning",
            )

                                                                        
                    
                                                                        

    @staticmethod
    def _validate_missing_values(
        dataframe: pd.DataFrame,
        report: ValidationReport,
    ) -> None:

        missing_counts = (
            dataframe.isna()
            .sum()
        )

        missing_counts = missing_counts[
            missing_counts > 0
        ]

        for column, count in (
            missing_counts.items()
        ):

            report.add_issue(
                category="missing_values",
                message=(
                    f"Column '{column}' contains "
                    f"{count} missing values."
                ),
                count=int(count),
                severity="warning",
            )

                                                                        
                          
                                                                        

    @staticmethod
    def _validate_duplicates(
        dataframe: pd.DataFrame,
        keys: list[str],
        report: ValidationReport,
    ) -> None:

        duplicates = dataframe.duplicated(
            subset=keys,
            keep=False,
        )

        count = int(
            duplicates.sum()
        )

        if count > 0:

            report.add_issue(
                category="duplicates",
                message=(
                    "Duplicate measurement records "
                    f"detected using keys {keys}."
                ),
                count=count,
            )

                                                                        
                          
                                                                        

    @staticmethod
    def _validate_numeric_columns(
        dataframe: pd.DataFrame,
        columns: list[str],
        report: ValidationReport,
    ) -> None:

        for column in columns:

            values = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            invalid = values.isna()

            if invalid.any():

                report.add_issue(
                    category="numerical",
                    message=(
                        f"Column '{column}' contains "
                        "non-numeric or invalid values."
                    ),
                    count=int(invalid.sum()),
                )

            infinite = np.isinf(
                values.to_numpy(
                    dtype=float
                )
            )

            if infinite.any():

                report.add_issue(
                    category="numerical",
                    message=(
                        f"Column '{column}' contains "
                        "infinite values."
                    ),
                    count=int(infinite.sum()),
                )

                                                                        
                      
                                                                        

    @staticmethod
    def _validate_range(
        dataframe: pd.DataFrame,
        column: str,
        minimum: float,
        maximum: float,
        report: ValidationReport,
        category: str,
    ) -> None:

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        invalid = (
            (values < minimum)
            | (values > maximum)
        )

        invalid = invalid.fillna(False)

        count = int(
            invalid.sum()
        )

        if count > 0:

            report.add_issue(
                category=category,
                message=(
                    f"Column '{column}' contains "
                    f"{count} values outside the "
                    f"allowed range "
                    f"[{minimum}, {maximum}]."
                ),
                count=count,
            )


def validate_saved_dataset(
    bus_path: str | Path,
    line_path: str | Path,
) -> tuple[
    ValidationReport,
    ValidationReport,
]:
    """
    Convenience function for validating saved measurement files.
    """

    bus_path = Path(bus_path)
    line_path = Path(line_path)

    if not bus_path.exists():
        raise FileNotFoundError(
            f"Bus measurement file not found: {bus_path}"
        )

    if not line_path.exists():
        raise FileNotFoundError(
            f"Line measurement file not found: {line_path}"
        )

    bus_data = pd.read_csv(
        bus_path,
        parse_dates=["timestamp"],
    )

    line_data = pd.read_csv(
        line_path,
        parse_dates=["timestamp"],
    )

    validator = MeasurementValidator()

    bus_report = validator.validate_bus_measurements(
        bus_data
    )

    line_report = validator.validate_line_measurements(
        line_data
    )

    return bus_report, line_report


if __name__ == "__main__":

    bus_path = Path(
        "data/simulated/measurements/"
        "bus_measurements.csv"
    )

    line_path = Path(
        "data/simulated/measurements/"
        "line_measurements.csv"
    )

    bus_report, line_report = validate_saved_dataset(
        bus_path,
        line_path,
    )

    print(
        bus_report.summary()
    )

    print(
        line_report.summary()
    )