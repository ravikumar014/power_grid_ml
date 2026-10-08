"""
Statistical feature engineering for power-grid measurements.

This module converts raw electrical measurements into local statistical
descriptors suitable for downstream anomaly-detection models.

Feature families
----------------
For each selected signal x(t):

    Mean
    Standard deviation
    Variance
    RMS
    Skewness
    Kurtosis
    Minimum
    Maximum
    Range

The features are calculated over a rolling temporal window so that
the resulting feature matrix retains temporal information.

Example
-------
For a voltage signal:

    voltage_pu

we generate:

    voltage_pu_mean
    voltage_pu_std
    voltage_pu_variance
    voltage_pu_rms
    voltage_pu_skewness
    voltage_pu_kurtosis
    voltage_pu_min
    voltage_pu_max
    voltage_pu_range

These features will later be combined with electrical, frequency and
graph features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class StatisticalFeatureConfig:
    """
    Configuration for statistical feature generation.
    """

    window_size: int = 4

    min_periods: int = 2

    include_mean: bool = True
    include_std: bool = True
    include_variance: bool = True
    include_rms: bool = True
    include_skewness: bool = True
    include_kurtosis: bool = True
    include_minimum: bool = True
    include_maximum: bool = True
    include_range: bool = True


class StatisticalFeatureEngineer:
    """
    Generate rolling statistical features.

    The class is deliberately independent of any ML model.
    """

    def __init__(
        self,
        config: StatisticalFeatureConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else StatisticalFeatureConfig()
        )

        if self.config.window_size < 2:

            raise ValueError(
                "window_size must be >= 2."
            )

        if (
            self.config.min_periods < 1
            or self.config.min_periods
            > self.config.window_size
        ):

            raise ValueError(
                "min_periods must be between "
                "1 and window_size."
            )

                                                                        
                
                                                                        

    @staticmethod
    def _validate_columns(
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> None:

        missing = (
            set(columns)
            - set(dataframe.columns)
        )

        if missing:

            raise ValueError(
                "Required columns missing: "
                f"{sorted(missing)}"
            )

                                                                        
                        
                                                                        

    @staticmethod
    def _prepare_numeric(
        dataframe: pd.DataFrame,
        features: list[str],
    ) -> pd.DataFrame:
        """
        Convert selected columns to numeric values.
        """

        result = dataframe[
            features
        ].copy()

        for feature in features:

            result[feature] = pd.to_numeric(
                result[feature],
                errors="coerce",
            )

        result = result.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return result

                                                                        
         
                                                                        

    @staticmethod
    def _rolling_rms(
        series: pd.Series,
        window: int,
        min_periods: int,
    ) -> pd.Series:
        """
        Rolling root-mean-square:

            RMS = sqrt(mean(x^2))
        """

        return np.sqrt(
            series
            .pow(2)
            .rolling(
                window=window,
                min_periods=min_periods,
            )
            .mean()
        )

                                                                        
                                      
                                                                        

    def generate_signal_features(
        self,
        series: pd.Series,
        prefix: str | None = None,
    ) -> pd.DataFrame:
        """
        Generate statistical features for one signal.

        Parameters
        ----------
        series:
            Input time-series signal.

        prefix:
            Prefix used in output column names.

        Returns
        -------
        DataFrame
            Rolling statistical features.
        """

        if prefix is None:

            prefix = series.name

        if prefix is None:

            raise ValueError(
                "Signal must have a name or prefix."
            )

        series = pd.to_numeric(
            series,
            errors="coerce",
        )

        series = series.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        window = (
            self.config.window_size
        )

        min_periods = (
            self.config.min_periods
        )

        rolling = (
            series
            .rolling(
                window=window,
                min_periods=min_periods,
            )
        )

        features: dict[
            str,
            pd.Series,
        ] = {}

                                                                        
              
                                                                        

        if self.config.include_mean:

            features[
                f"{prefix}_mean"
            ] = rolling.mean()

                                                                        
                            
                                                                        

        if self.config.include_std:

            features[
                f"{prefix}_std"
            ] = rolling.std(
                ddof=1
            )

                                                                        
                  
                                                                        

        if self.config.include_variance:

            features[
                f"{prefix}_variance"
            ] = rolling.var(
                ddof=1
            )

                                                                        
             
                                                                        

        if self.config.include_rms:

            features[
                f"{prefix}_rms"
            ] = self._rolling_rms(
                series,
                window,
                min_periods,
            )

                                                                        
                  
                                                                        

        if self.config.include_skewness:

            features[
                f"{prefix}_skewness"
            ] = rolling.skew()

                                                                        
                  
                                                                        

        if self.config.include_kurtosis:

            features[
                f"{prefix}_kurtosis"
            ] = rolling.kurt()

                                                                        
                 
                                                                        

        if self.config.include_minimum:

            features[
                f"{prefix}_min"
            ] = rolling.min()

                                                                        
                 
                                                                        

        if self.config.include_maximum:

            features[
                f"{prefix}_max"
            ] = rolling.max()

                                                                        
               
                                                                        

        if self.config.include_range:

            features[
                f"{prefix}_range"
            ] = (
                rolling.max()
                - rolling.min()
            )

        return pd.DataFrame(
            features,
            index=series.index,
        )

                                                                        
                                            
                                                                        

    def generate(
        self,
        dataframe: pd.DataFrame,
        features: list[str],
    ) -> pd.DataFrame:
        """
        Generate statistical features for multiple signals.

        Original dataframe is not modified.
        """

        self._validate_columns(
            dataframe,
            features,
        )

        numeric = self._prepare_numeric(
            dataframe,
            features,
        )

        generated = []

        for feature in features:

            signal_features = (
                self.generate_signal_features(
                    numeric[feature],
                    prefix=feature,
                )
            )

            generated.append(
                signal_features
            )

        if not generated:

            return pd.DataFrame(
                index=dataframe.index
            )

        return pd.concat(
            generated,
            axis=1,
        )

                                                                        
                                        
                                                                        

    def transform(
        self,
        dataframe: pd.DataFrame,
        features: list[str],
        preserve_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Generate features while preserving selected metadata columns.

        Example
        -------
        preserve_columns:

            ["timestamp", "bus_id"]

        """

        if preserve_columns is None:

            preserve_columns = []

        self._validate_columns(
            dataframe,
            features
            + preserve_columns,
        )

        metadata = (
            dataframe[
                preserve_columns
            ].copy()
        )

        generated = self.generate(
            dataframe,
            features,
        )

        return pd.concat(
            [
                metadata.reset_index(
                    drop=True
                ),
                generated.reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )

                                                                        
                       
                                                                        

    @staticmethod
    def global_statistics(
        dataframe: pd.DataFrame,
        features: list[str],
    ) -> pd.DataFrame:
        """
        Calculate global descriptive statistics.

        This is mainly for reporting/inspection.

        The ML feature matrix uses rolling statistics instead.
        """

        missing = (
            set(features)
            - set(dataframe.columns)
        )

        if missing:

            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

        result = (
            dataframe[
                features
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .describe()
            .T
        )

                                      
        result["variance"] = (
            dataframe[
                features
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .var()
        )

        result["skewness"] = (
            dataframe[
                features
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .skew()
        )

        result["kurtosis"] = (
            dataframe[
                features
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .kurt()
        )

        return result

                                                                        
                         
                                                                        

    @staticmethod
    def save(
        dataframe: pd.DataFrame,
        path: str | Path,
    ) -> Path:
        """
        Save generated features.
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

        return path


                                                                        
      
                                                                        


if __name__ == "__main__":

    from src.eda.load_data import (
        EDADataLoader,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "STATISTICAL FEATURE ENGINEERING"
    )

    print(
        "=" * 70
    )

                                                                    
               
                                                                    

    loader = EDADataLoader()

    dataset = loader.load_all()

                                                                    
                   
     
                                             
     
                
     
                
     
                         
                                                                    

    config = StatisticalFeatureConfig(
        window_size=4,
        min_periods=2,
    )

    engineer = (
        StatisticalFeatureEngineer(
            config=config
        )
    )

    bus_features = [
        "voltage_pu",
        "angle_deg",
        "active_power_mw",
        "reactive_power_mvar",
    ]

                                                                    
                           
                                                                    

    print(
        "\nGenerating bus statistical features..."
    )

    bus_feature_matrix = (
        engineer.transform(
            dataset.bus,
            bus_features,
            preserve_columns=[
                "timestamp",
                "bus_id",
            ],
        )
    )

                                                                    
          
                                                                    

    bus_path = engineer.save(
        bus_feature_matrix,
        "data/features/bus/statistical_features.csv",
    )

    print(
        f"Saved bus features to:"
        f"\n{bus_path}"
    )

    print(
        f"\nBus feature matrix shape:"
        f" {bus_feature_matrix.shape}"
    )

                                                                    
                   
                                                                    

    line_features = [
        "current_from_ka",
        "current_to_ka",
        "loading_percent",
    ]

    print(
        "\nGenerating line statistical features..."
    )

    line_feature_matrix = (
        engineer.transform(
            dataset.line,
            line_features,
            preserve_columns=[
                "timestamp",
                "line_id",
            ],
        )
    )

    line_path = engineer.save(
        line_feature_matrix,
        "data/features/line/statistical_features.csv",
    )

    print(
        f"Saved line features to:"
        f"\n{line_path}"
    )

    print(
        f"\nLine feature matrix shape:"
        f" {line_feature_matrix.shape}"
    )

                                                                    
                       
                                                                    

    statistics = (
        engineer.global_statistics(
            dataset.bus,
            bus_features,
        )
    )

    statistics_path = engineer.save(
        statistics.reset_index(),
        "outputs/features/"
        "bus_global_statistics.csv",
    )

    print(
        f"\nGlobal statistics saved to:"
        f"\n{statistics_path}"
    )

                                                                    
             
                                                                    

    print()
    print(
        "Generated bus features:"
    )

    print(
        bus_feature_matrix.head(
            10
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "STATISTICAL FEATURE ENGINEERING COMPLETE"
    )

    print(
        "=" * 70
    )