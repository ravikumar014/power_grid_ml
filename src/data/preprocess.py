"""
Preprocessing utilities for power-grid measurement data.

Responsibilities
----------------
1. Convert timestamps to a consistent representation.
2. Sort measurements chronologically.
3. Remove duplicate records.
4. Handle missing values.
5. Preserve the electrical meaning of measurements.
6. Return a clean dataframe ready for normalization.

Important
---------
This module does not perform normalization and does not split the
dataset into train/validation/test sets. Those operations are kept
separate to prevent data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PreprocessingReport:
    """
    Records what happened during preprocessing.
    """

    initial_rows: int = 0
    final_rows: int = 0

    duplicates_removed: int = 0
    missing_values_before: int = 0
    missing_values_after: int = 0

    interpolated_values: int = 0
    forward_filled_values: int = 0

    columns_added: list[str] | None = None

    def __post_init__(self) -> None:
        if self.columns_added is None:
            self.columns_added = []

    def summary(self) -> str:
        """
        Return a human-readable preprocessing summary.
        """

        lines = [
            "",
            "=" * 70,
            "POWER GRID PREPROCESSING REPORT",
            "=" * 70,
            f"Initial rows            : {self.initial_rows}",
            f"Final rows              : {self.final_rows}",
            f"Duplicates removed     : {self.duplicates_removed}",
            f"Missing before         : {self.missing_values_before}",
            f"Missing after          : {self.missing_values_after}",
            f"Interpolated values    : {self.interpolated_values}",
            f"Forward-filled values  : {self.forward_filled_values}",
            f"Columns added          : {self.columns_added}",
            "=" * 70,
        ]

        return "\n".join(lines)


class PowerGridPreprocessor:
    """
    Preprocessor for bus and line measurement data.

    Parameters
    ----------
    missing_strategy:
        Strategy used for missing numerical measurements.

        Supported:
            "interpolate"
            "forward_fill"
            "none"

    add_time_features:
        Whether to add hour/minute/day-related features.
    """

    def __init__(
        self,
        missing_strategy: str = "interpolate",
        add_time_features: bool = True,
    ) -> None:

        supported_strategies = {
            "interpolate",
            "forward_fill",
            "none",
        }

        if missing_strategy not in supported_strategies:
            raise ValueError(
                f"Unsupported missing strategy: "
                f"{missing_strategy}. "
                f"Choose from {supported_strategies}."
            )

        self.missing_strategy = missing_strategy
        self.add_time_features = add_time_features

                                                                        
                      
                                                                        

    def process(
        self,
        dataframe: pd.DataFrame,
        dataset_type: str = "bus",
    ) -> tuple[pd.DataFrame, PreprocessingReport]:
        """
        Preprocess a measurement dataframe.

        Parameters
        ----------
        dataframe:
            Input measurement dataframe.

        dataset_type:
            "bus" or "line".

        Returns
        -------
        processed_dataframe
        preprocessing_report
        """

        if dataset_type not in {"bus", "line"}:
            raise ValueError(
                "dataset_type must be either 'bus' or 'line'."
            )

        if dataframe.empty:
            raise ValueError(
                "Cannot preprocess an empty dataframe."
            )

        data = dataframe.copy()

        report = PreprocessingReport(
            initial_rows=len(data)
        )

                                                                        
                                 
                                                                        

        data = self._process_timestamps(
            data
        )

                                                                        
                                    
                                                                        

        before = len(data)

        data = data.drop_duplicates()

        report.duplicates_removed = (
            before - len(data)
        )

                                                                        
                                              
         
                                                                  
                                                                   
                                                                        

        id_column = (
            "bus_id"
            if dataset_type == "bus"
            else "line_id"
        )

        before = len(data)

        data = data.drop_duplicates(
            subset=[
                "timestamp",
                id_column,
            ],
            keep="first",
        )

        report.duplicates_removed += (
            before - len(data)
        )

                                                                        
                                 
                                                                        

        data = data.sort_values(
            by=[
                id_column,
                "timestamp",
            ]
        ).reset_index(drop=True)

                                                                        
                                                  
                                                                        

        report.missing_values_before = int(
            data.isna().sum().sum()
        )

                                                                        
                                      
                                                                        

        numerical_columns = self._get_numerical_columns(
            data,
            dataset_type,
        )

        for column in numerical_columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

                                                                        
                                  
                                                                        

        (
            data,
            interpolated,
            forward_filled,
        ) = self._handle_missing_values(
            data,
            numerical_columns,
            id_column,
        )

        report.interpolated_values = interpolated
        report.forward_filled_values = forward_filled

                                                                        
                                  
                                                                        

        if self.add_time_features:

            before_columns = set(
                data.columns
            )

            data = self._add_time_features(
                data
            )

            report.columns_added = sorted(
                set(data.columns)
                - before_columns
            )

                                                                        
                           
                                                                        

        data = data.sort_values(
            by=[
                id_column,
                "timestamp",
            ]
        ).reset_index(drop=True)

                                                                        
                                       
                                                                        

        report.missing_values_after = int(
            data.isna().sum().sum()
        )

        report.final_rows = len(data)

        return data, report

                                                                        
                          
                                                                        

    @staticmethod
    def _process_timestamps(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert timestamp column to pandas datetime.
        """

        data = dataframe.copy()

        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce",
        )

        if data["timestamp"].isna().any():
            raise ValueError(
                "Invalid timestamps detected during preprocessing."
            )

        return data

                                                                        
                       
                                                                        

    @staticmethod
    def _get_numerical_columns(
        dataframe: pd.DataFrame,
        dataset_type: str,
    ) -> list[str]:
        """
        Return numerical measurement columns.
        """

        if dataset_type == "bus":

            candidates = [
                "voltage_pu",
                "angle_deg",
                "active_power_mw",
                "reactive_power_mvar",
                "frequency_hz",
                "der_generation_mw",
            ]

        else:

            candidates = [
                "current_from_ka",
                "current_to_ka",
                "loading_percent",
            ]

        return [
            column
            for column in candidates
            if column in dataframe.columns
        ]

                                                                        
                    
                                                                        

    def _handle_missing_values(
        self,
        dataframe: pd.DataFrame,
        numerical_columns: list[str],
        id_column: str,
    ) -> tuple[
        pd.DataFrame,
        int,
        int,
    ]:
        """
        Handle missing numerical measurements independently for each
        bus/line.

        Returns
        -------
        dataframe
        interpolated_count
        forward_filled_count
        """

        data = dataframe.copy()

        interpolated_count = 0
        forward_filled_count = 0

        if self.missing_strategy == "none":

            return (
                data,
                0,
                0,
            )

                                                                        
                                                     
         
                                                                     
                                                                        

        for entity_id, group_index in data.groupby(
            id_column
        ).groups.items():

            group = data.loc[
                group_index
            ].sort_values(
                "timestamp"
            )

            for column in numerical_columns:

                before_missing = int(
                    group[column].isna().sum()
                )

                if before_missing == 0:
                    continue

                if self.missing_strategy == "interpolate":

                    group[column] = (
                        group[column]
                        .interpolate(
                            method="linear",
                            limit_direction="both",
                        )
                    )

                    after_missing = int(
                        group[column].isna().sum()
                    )

                    interpolated_count += (
                        before_missing
                        - after_missing
                    )

                elif self.missing_strategy == "forward_fill":

                    group[column] = (
                        group[column]
                        .ffill()
                    )

                    after_missing = int(
                        group[column].isna().sum()
                    )

                    forward_filled_count += (
                        before_missing
                        - after_missing
                    )

                data.loc[
                    group.index,
                    column,
                ] = group[column]

        return (
            data,
            interpolated_count,
            forward_filled_count,
        )

                                                                        
                       
                                                                        

    @staticmethod
    def _add_time_features(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add non-electrical temporal features.

        These are useful later for distinguishing normal daily
        operating patterns from genuinely anomalous behaviour.
        """

        data = dataframe.copy()

        timestamp = data["timestamp"]

        data["hour"] = (
            timestamp.dt.hour
            + timestamp.dt.minute / 60.0
        )

        data["day_of_week"] = (
            timestamp.dt.dayofweek
        )

        data["day_of_year"] = (
            timestamp.dt.dayofyear
        )

                                                                        
                         
         
                                                      
                                                                        

        data["hour_sin"] = np.sin(
            2.0
            * np.pi
            * data["hour"]
            / 24.0
        )

        data["hour_cos"] = np.cos(
            2.0
            * np.pi
            * data["hour"]
            / 24.0
        )

        data["day_of_week_sin"] = np.sin(
            2.0
            * np.pi
            * data["day_of_week"]
            / 7.0
        )

        data["day_of_week_cos"] = np.cos(
            2.0
            * np.pi
            * data["day_of_week"]
            / 7.0
        )

        return data

                                                                        
          
                                                                        

    @staticmethod
    def save(
        dataframe: pd.DataFrame,
        path: str | Path,
    ) -> None:
        """
        Save processed data as CSV.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(
            path,
            index=False,
        )


def preprocess_saved_dataset(
    input_path: str | Path,
    output_path: str | Path,
    dataset_type: str = "bus",
) -> PreprocessingReport:
    """
    Convenience function for preprocessing a saved CSV dataset.
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}"
        )

    data = pd.read_csv(
        input_path,
        parse_dates=["timestamp"],
    )

    preprocessor = PowerGridPreprocessor(
        missing_strategy="interpolate",
        add_time_features=True,
    )

    processed_data, report = (
        preprocessor.process(
            data,
            dataset_type=dataset_type,
        )
    )

    preprocessor.save(
        processed_data,
        output_path,
    )

    return report


if __name__ == "__main__":

    bus_input = Path(
        "data/simulated/measurements/"
        "bus_measurements.csv"
    )

    bus_output = Path(
        "data/processed/"
        "bus_measurements_processed.csv"
    )

    report = preprocess_saved_dataset(
        input_path=bus_input,
        output_path=bus_output,
        dataset_type="bus",
    )

    print(
        report.summary()
    )