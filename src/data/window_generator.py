"""
Temporal window generation for power-grid time-series ML.

This module converts normalized measurement data into fixed-length
temporal sequences.

Example
-------
Given:

    window_size = 60

the generated sample is:

    X[t] = [
        x[t-59],
        x[t-58],
        ...
        x[t]
    ]

Windows are generated independently for train, validation and test
datasets to prevent temporal leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class WindowDataset:
    """
    Container for generated temporal windows.
    """

    X: np.ndarray

    timestamps: np.ndarray

    entity_ids: np.ndarray

    feature_columns: list[str]

    window_size: int

    stride: int

    def __len__(self) -> int:
        return len(self.X)

    @property
    def shape(self) -> tuple:
        return self.X.shape


class TemporalWindowGenerator:
    """
    Generate fixed-length temporal windows.

    Parameters
    ----------
    window_size:
        Number of consecutive timesteps in each window.

    stride:
        Number of timesteps between consecutive windows.

    entity_column:
        Identifier for the physical entity.

        For bus-level data this is normally:
            bus_id

        For line-level data:
            line_id

    timestamp_column:
        Name of timestamp column.
    """

    def __init__(
        self,
        window_size: int = 60,
        stride: int = 1,
        entity_column: str = "bus_id",
        timestamp_column: str = "timestamp",
    ) -> None:

        if window_size <= 0:
            raise ValueError(
                "window_size must be greater than zero."
            )

        if stride <= 0:
            raise ValueError(
                "stride must be greater than zero."
            )

        self.window_size = window_size
        self.stride = stride
        self.entity_column = entity_column
        self.timestamp_column = timestamp_column

                                                                        
                            
                                                                        

    def generate(
        self,
        dataframe: pd.DataFrame,
        feature_columns: list[str],
    ) -> WindowDataset:
        """
        Generate temporal windows.

        Windows are created independently for each physical entity.

        Parameters
        ----------
        dataframe:
            Normalized dataframe.

        feature_columns:
            Numerical features to include in each window.

        Returns
        -------
        WindowDataset
        """

        self._validate_input(
            dataframe,
            feature_columns,
        )

        data = dataframe.copy()

        data[self.timestamp_column] = pd.to_datetime(
            data[self.timestamp_column]
        )

                                                                        
                                  
                                                                        

        data = data.sort_values(
            by=[
                self.entity_column,
                self.timestamp_column,
            ]
        ).reset_index(drop=True)

        windows = []
        timestamps = []
        entity_ids = []

                                                                        
                                                           
                                                                        

        for entity_id, group in data.groupby(
            self.entity_column,
            sort=True,
        ):

            group = group.sort_values(
                self.timestamp_column
            ).reset_index(drop=True)

            values = group[
                feature_columns
            ].to_numpy(
                dtype=np.float32
            )

            time_values = group[
                self.timestamp_column
            ].to_numpy()

                                                                        
                                                     
                                                                        

            if len(group) < self.window_size:
                continue

                                                                        
                             
                                                                        

            for start in range(
                0,
                len(group) - self.window_size + 1,
                self.stride,
            ):

                end = (
                    start
                    + self.window_size
                )

                window = values[
                    start:end
                ]

                windows.append(
                    window
                )

                                                       
                 
                                                                
                                                               
                timestamps.append(
                    time_values[end - 1]
                )

                entity_ids.append(
                    entity_id
                )

                                                                        
                                      
                                                                        

        if not windows:

            X = np.empty(
                (
                    0,
                    self.window_size,
                    len(feature_columns),
                ),
                dtype=np.float32,
            )

            timestamps_array = np.empty(
                (0,),
                dtype="datetime64[ns]",
            )

            entity_array = np.empty(
                (0,),
                dtype=np.int64,
            )

            return WindowDataset(
                X=X,
                timestamps=timestamps_array,
                entity_ids=entity_array,
                feature_columns=feature_columns,
                window_size=self.window_size,
                stride=self.stride,
            )

                                                                        
                                  
                                                                        

        X = np.stack(
            windows
        ).astype(
            np.float32
        )

        timestamps_array = np.asarray(
            timestamps
        )

        entity_array = np.asarray(
            entity_ids
        )

        return WindowDataset(
            X=X,
            timestamps=timestamps_array,
            entity_ids=entity_array,
            feature_columns=feature_columns,
            window_size=self.window_size,
            stride=self.stride,
        )

                                                                        
                
                                                                        

    def _validate_input(
        self,
        dataframe: pd.DataFrame,
        feature_columns: list[str],
    ) -> None:

        if dataframe.empty:
            raise ValueError(
                "Cannot generate windows from an empty dataframe."
            )

        required_columns = {
            self.timestamp_column,
            self.entity_column,
        }

        missing = (
            required_columns
            - set(dataframe.columns)
        )

        if missing:
            raise ValueError(
                "Missing required columns: "
                f"{sorted(missing)}"
            )

        if not feature_columns:
            raise ValueError(
                "feature_columns cannot be empty."
            )

        missing_features = (
            set(feature_columns)
            - set(dataframe.columns)
        )

        if missing_features:
            raise ValueError(
                "Feature columns not found: "
                f"{sorted(missing_features)}"
            )

        for column in feature_columns:

            if not pd.api.types.is_numeric_dtype(
                dataframe[column]
            ):

                raise ValueError(
                    f"Feature '{column}' must be numeric."
                )

        if dataframe[
            self.timestamp_column
        ].isna().any():

            raise ValueError(
                "Timestamp column contains missing values."
            )

        if dataframe[
            self.entity_column
        ].isna().any():

            raise ValueError(
                "Entity column contains missing values."
            )

                                                                        
          
                                                                        

    @staticmethod
    def save(
        dataset: WindowDataset,
        path: str | Path,
    ) -> None:
        """
        Save generated windows as a compressed NumPy archive.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            path,
            X=dataset.X,
            timestamps=dataset.timestamps,
            entity_ids=dataset.entity_ids,
            feature_columns=np.asarray(
                dataset.feature_columns,
                dtype=object,
            ),
            window_size=np.asarray(
                dataset.window_size
            ),
            stride=np.asarray(
                dataset.stride
            ),
        )

                                                                        
          
                                                                        

    @staticmethod
    def load(
        path: str | Path,
    ) -> WindowDataset:
        """
        Load a saved WindowDataset.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Window dataset not found: {path}"
            )

        archive = np.load(
            path,
            allow_pickle=True,
        )

        return WindowDataset(
            X=archive["X"],
            timestamps=archive["timestamps"],
            entity_ids=archive["entity_ids"],
            feature_columns=archive[
                "feature_columns"
            ].tolist(),
            window_size=int(
                archive["window_size"]
            ),
            stride=int(
                archive["stride"]
            ),
        )


def generate_saved_windows(
    input_path: str | Path,
    output_path: str | Path,
    entity_column: str = "bus_id",
    window_size: int = 60,
    stride: int = 1,
) -> WindowDataset:
    """
    Convenience function for generating windows from a saved CSV.
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

                                                                    
                                                
     
                                                          
                                                                    

    feature_columns = [
        "voltage_pu",
        "angle_deg",
        "active_power_mw",
        "reactive_power_mvar",
    ]

    generator = TemporalWindowGenerator(
        window_size=window_size,
        stride=stride,
        entity_column=entity_column,
    )

    dataset = generator.generate(
        data,
        feature_columns=feature_columns,
    )

    generator.save(
        dataset,
        output_path,
    )

    return dataset


if __name__ == "__main__":

    input_directory = Path(
        "data/processed/normalized"
    )

    output_directory = Path(
        "data/processed/windows"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

                                                                    
                   
                                                                    

    train_dataset = generate_saved_windows(
        input_path=(
            input_directory
            / "train.csv"
        ),
        output_path=(
            output_directory
            / "train.npz"
        ),
        entity_column="bus_id",
        window_size=60,
        stride=1,
    )

                                                                    
                        
                                                                    

    validation_dataset = generate_saved_windows(
        input_path=(
            input_directory
            / "validation.csv"
        ),
        output_path=(
            output_directory
            / "validation.npz"
        ),
        entity_column="bus_id",
        window_size=60,
        stride=1,
    )

                                                                    
                  
                                                                    

    test_dataset = generate_saved_windows(
        input_path=(
            input_directory
            / "test.csv"
        ),
        output_path=(
            output_directory
            / "test.npz"
        ),
        entity_column="bus_id",
        window_size=60,
        stride=1,
    )

    print(
        "\nTemporal windows generated successfully."
    )

    print(
        "\nTrain shape:"
        f" {train_dataset.shape}"
    )

    print(
        "Validation shape:"
        f" {validation_dataset.shape}"
    )

    print(
        "Test shape:"
        f" {test_dataset.shape}"
    )

    print(
        "\nFeature columns:"
    )

    for feature in train_dataset.feature_columns:
        print(
            f"  - {feature}"
        )