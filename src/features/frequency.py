"""
Frequency-domain feature engineering for power-grid measurements.

This module converts temporal electrical signals into numerical
frequency-domain descriptors suitable for anomaly-detection models.

Feature families
----------------
1. Dominant frequency
2. Dominant period
3. Dominant amplitude
4. Spectral energy
5. Spectral centroid
6. Spectral bandwidth
7. Spectral entropy
8. Low-frequency energy
9. High-frequency energy
10. Spectral peak count

The implementation uses FFT.

Important
---------
Frequency-domain features are computed over a temporal window.

For the current development dataset:

    sampling interval = 15 minutes
    window size        = 96 samples

corresponds to approximately one day.

This is useful for development, but longer datasets are required
before interpreting daily/weekly periodicity scientifically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


                                                                        
               
                                                                        


@dataclass
class FrequencyFeatureConfig:
    """
    Configuration for FFT feature extraction.
    """

    window_size: int = 96

    min_periods: int = 32

                                                   
     
                                                 
                                                 
    low_frequency_max_cph: float = 1.0

    high_frequency_min_cph: float = 2.0

    epsilon: float = 1e-12

    include_dominant_frequency: bool = True

    include_dominant_period: bool = True

    include_dominant_amplitude: bool = True

    include_total_energy: bool = True

    include_centroid: bool = True

    include_bandwidth: bool = True

    include_entropy: bool = True

    include_low_frequency_energy: bool = True

    include_high_frequency_energy: bool = True

    include_peak_count: bool = True


                                                                        
                            
                                                                        


class FrequencyFeatureEngineer:
    """
    Generate rolling FFT-based features.

    The class operates independently of any anomaly-detection model.
    """

    def __init__(
        self,
        config: FrequencyFeatureConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else FrequencyFeatureConfig()
        )

        if self.config.window_size < 4:

            raise ValueError(
                "window_size must be >= 4."
            )

        if (
            self.config.min_periods
            < 4
        ):

            raise ValueError(
                "min_periods must be >= 4."
            )

        if (
            self.config.min_periods
            > self.config.window_size
        ):

            raise ValueError(
                "min_periods cannot exceed window_size."
            )

        if (
            self.config.low_frequency_max_cph
            < 0
        ):

            raise ValueError(
                "low_frequency_max_cph must be >= 0."
            )

        if (
            self.config.high_frequency_min_cph
            <= self.config.low_frequency_max_cph
        ):

            raise ValueError(
                "high_frequency_min_cph must be greater "
                "than low_frequency_max_cph."
            )

                                                                        
                       
                                                                        

    @staticmethod
    def infer_sampling_interval(
        timestamps: pd.Series | pd.DatetimeIndex,
    ) -> float:
        """
        Infer the sampling interval in seconds.

        Important:
        Timestamps may contain duplicates because multiple physical
        entities (buses/lines) are recorded at the same timestamp.

        Therefore duplicate timestamps are removed before calculating
        temporal differences.
        """

        timestamps = pd.DatetimeIndex(
            pd.to_datetime(timestamps)
        )

                                      
        timestamps = timestamps.drop_duplicates()

                               
        timestamps = timestamps.sort_values()

        if len(timestamps) < 2:

            raise ValueError(
                "At least two unique timestamps are required."
            )

        differences = (
            pd.Series(timestamps)
            .diff()
            .dt.total_seconds()
            .dropna()
        )

                                                                  
        differences = differences[
            differences > 0
        ]

        if differences.empty:

            raise ValueError(
                "Unable to infer a positive sampling interval."
            )

                                                              
        interval = float(
            differences.median()
        )

        if interval <= 0:

            raise ValueError(
                "Sampling interval must be positive."
            )

        return interval
    
                                                                            
                       
                                                                        

    def compute_spectrum(
        self,
        values: np.ndarray,
        sampling_interval_seconds: float,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        Compute a one-sided FFT spectrum for a single temporal window.

        Parameters
        ----------
        values:
            Time-series samples belonging to ONE physical entity.

        sampling_interval_seconds:
            Temporal spacing between consecutive samples.

        Returns
        -------
        frequencies_cph:
            Frequencies expressed in cycles/hour.

        power:
            Corresponding spectral power.

        Notes
        -----
        The mean is removed before FFT so that the DC component does
        not dominate the spectrum.

        A Hann window is applied to reduce spectral leakage.
        """

        values = np.asarray(
            values,
            dtype=np.float64,
        )

                                                                        
                                
                                                                        

        values = values[
            np.isfinite(values)
        ]

        if len(values) < 4:

            raise ValueError(
                "At least four finite samples are required "
                "for FFT."
            )

        if sampling_interval_seconds <= 0:

            raise ValueError(
                "Sampling interval must be positive."
            )

                                                                        
                                  
         
                                
                                                                        

        values = (
            values
            - np.mean(values)
        )

        n = len(values)

                                                                        
                      
         
                                                          
                                                                        

        window = np.hanning(
            n
        )

        windowed_values = (
            values
            * window
        )

                                                                        
                          
                                                                        

        fft_values = np.fft.rfft(
            windowed_values
        )

                                                                        
                               
         
                            
                                                                        

        frequencies_hz = (
            np.fft.rfftfreq(
                n,
                d=sampling_interval_seconds,
            )
        )

                                                                        
                                   
                                                                        

        frequencies_cph = (
            frequencies_hz
            * 3600.0
        )

                                                                        
                         
                                                                        

        power = (
            np.abs(
                fft_values
            ) ** 2
        )

                                                                        
                              
                                                                        

        frequencies_cph = (
            frequencies_cph[1:]
        )

        power = (
            power[1:]
        )

        return (
            frequencies_cph,
            power,
        )

                                                                        
                      
                                                                        

    def spectral_entropy(
        self,
        power: np.ndarray,
    ) -> float:
        """
        Calculate normalized spectral entropy.

        p_i = P_i / sum(P)

        H = -sum(p_i log(p_i))

        Normalized:

        H_norm = H / log(N)

        Values approximately lie in:

            [0, 1]

        Low entropy:
            energy concentrated in a few frequencies.

        High entropy:
            energy spread over many frequencies.
        """

        power = np.asarray(
            power,
            dtype=np.float64,
        )

        power = np.maximum(
            power,
            0.0,
        )

        total = float(
            power.sum()
        )

        if total <= self.config.epsilon:

            return 0.0

        probability = (
            power
            / total
        )

        probability = (
            probability[
                probability
                > self.config.epsilon
            ]
        )

        entropy = -np.sum(
            probability
            * np.log(
                probability
            )
        )

        if len(probability) <= 1:

            return 0.0

        return float(
            entropy
            / np.log(
                len(probability)
            )
        )

                                                                        
                          
                                                                        

    def extract_window_features(
        self,
        values: np.ndarray,
        sampling_interval_seconds: float,
    ) -> dict[str, float]:
        """
        Extract all frequency features from one temporal window.
        """

        (
            frequencies,
            power,
        ) = self.compute_spectrum(
            values,
            sampling_interval_seconds,
        )

        if len(power) == 0:

            return self._empty_features()

        total_power = float(
            power.sum()
        )

        if total_power <= self.config.epsilon:

            return self._empty_features()

                                                                        
                            
                                                                        

        dominant_index = int(
            np.argmax(power)
        )

        dominant_frequency = float(
            frequencies[
                dominant_index
            ]
        )

        dominant_power = float(
            power[
                dominant_index
            ]
        )

        dominant_period_hours = (
            1.0
            / dominant_frequency
            if dominant_frequency > 0
            else np.inf
        )

                                                                        
                           
         
                                      
                                                                        

        centroid = float(
            np.sum(
                frequencies
                * power
            )
            / total_power
        )

                                                                        
                            
         
               
                                         
           
                                                                        

        bandwidth = float(
            np.sqrt(
                np.sum(
                    power
                    * (
                        frequencies
                        - centroid
                    ) ** 2
                )
                / total_power
            )
        )

                                                                        
                          
                                                                        

        entropy = (
            self.spectral_entropy(
                power
            )
        )

                                                                        
                              
                                                                        

        low_mask = (
            frequencies
            <= self.config.low_frequency_max_cph
        )

        low_energy = float(
            power[
                low_mask
            ].sum()
        )

                                                                        
                               
                                                                        

        high_mask = (
            frequencies
            >= self.config.high_frequency_min_cph
        )

        high_energy = float(
            power[
                high_mask
            ].sum()
        )

                                                                        
                       
                                                                        

        low_energy_ratio = (
            low_energy
            / total_power
        )

        high_energy_ratio = (
            high_energy
            / total_power
        )

                                                                        
                    
         
                                                                        
                                                                        

        threshold = (
            0.10
            * float(
                power.max()
            )
        )

        if len(power) >= 3:

            peak_mask = (
                (power[1:-1] > power[:-2])
                & (power[1:-1] > power[2:])
                & (power[1:-1] >= threshold)
            )

            peak_count = int(
                peak_mask.sum()
            )

        else:

            peak_count = 0

        return {
            "dominant_frequency_cph": (
                dominant_frequency
            ),

            "dominant_period_hours": (
                dominant_period_hours
            ),

            "dominant_amplitude_power": (
                dominant_power
            ),

            "total_spectral_energy": (
                total_power
            ),

            "spectral_centroid_cph": (
                centroid
            ),

            "spectral_bandwidth_cph": (
                bandwidth
            ),

            "spectral_entropy": (
                entropy
            ),

            "low_frequency_energy": (
                low_energy
            ),

            "high_frequency_energy": (
                high_energy
            ),

            "low_frequency_energy_ratio": (
                low_energy_ratio
            ),

            "high_frequency_energy_ratio": (
                high_energy_ratio
            ),

            "spectral_peak_count": (
                peak_count
            ),
        }

                                                                        
                            
                                                                        

    @staticmethod
    def _empty_features() -> dict[str, float]:

        return {
            "dominant_frequency_cph": np.nan,
            "dominant_period_hours": np.nan,
            "dominant_amplitude_power": np.nan,
            "total_spectral_energy": np.nan,
            "spectral_centroid_cph": np.nan,
            "spectral_bandwidth_cph": np.nan,
            "spectral_entropy": np.nan,
            "low_frequency_energy": np.nan,
            "high_frequency_energy": np.nan,
            "low_frequency_energy_ratio": np.nan,
            "high_frequency_energy_ratio": np.nan,
            "spectral_peak_count": np.nan,
        }

                                                                        
                                              
                                                                        

    def generate_signal_features(
        self,
        series: pd.Series,
        timestamps: pd.Series | None = None,
        prefix: str | None = None,
    ) -> pd.DataFrame:
        """
        Generate frequency features over rolling temporal windows.

        The output has one row per original timestamp.

        Early rows may contain NaN because there are not yet enough
        samples to form a complete spectral window.
        """

        if prefix is None:

            prefix = series.name

        if prefix is None:

            raise ValueError(
                "Signal must have a name or prefix."
            )

        values = pd.to_numeric(
            series,
            errors="coerce",
        ).to_numpy(
            dtype=np.float64
        )

                                                                        
                                      
                                                                        

        if timestamps is not None:

            sampling_interval = (
                self.infer_sampling_interval(
                    timestamps
                )
            )

        else:

            raise ValueError(
                "timestamps are required for frequency features."
            )

        n = len(values)

        feature_names = [
            "dominant_frequency_cph",
            "dominant_period_hours",
            "dominant_amplitude_power",
            "total_spectral_energy",
            "spectral_centroid_cph",
            "spectral_bandwidth_cph",
            "spectral_entropy",
            "low_frequency_energy",
            "high_frequency_energy",
            "low_frequency_energy_ratio",
            "high_frequency_energy_ratio",
            "spectral_peak_count",
        ]

        result = pd.DataFrame(
            np.nan,
            index=series.index,
            columns=[
                f"{prefix}_{name}"
                for name in feature_names
            ],
        )

        window = (
            self.config.window_size
        )

        min_periods = (
            self.config.min_periods
        )

                                                                        
                      
                                                                        

        for end in range(n):

            start = max(
                0,
                end - window + 1,
            )

            window_values = (
                values[start : end + 1]
            )

            finite_mask = (
                np.isfinite(
                    window_values
                )
            )

            valid_values = (
                window_values[
                    finite_mask
                ]
            )

            if len(valid_values) < min_periods:

                continue

            features = (
                self.extract_window_features(
                    valid_values,
                    sampling_interval,
                )
            )

            for name, value in features.items():

                result.loc[
                    result.index[end],
                    f"{prefix}_{name}",
                ] = value

        return result

                                                                        
                      
                                                                        

    def generate(
        self,
        dataframe: pd.DataFrame,
        features: list[str],
        timestamp_column: str = "timestamp",
        entity_column: str | None = None,
    ) -> pd.DataFrame:
        """
        Generate frequency features independently for each physical
        entity.

        For example:

            bus_id = 0 → independent FFT sequence
            bus_id = 1 → independent FFT sequence
            bus_id = 2 → independent FFT sequence

        The original dataframe ordering is preserved.
        """

        if timestamp_column not in dataframe.columns:

            raise ValueError(
                f"Missing timestamp column: "
                f"{timestamp_column}"
            )

        missing = (
            set(features)
            - set(dataframe.columns)
        )

        if missing:

            raise ValueError(
                f"Missing features: "
                f"{sorted(missing)}"
            )

        if (
            entity_column is not None
            and entity_column not in dataframe.columns
        ):

            raise ValueError(
                f"Missing entity column: "
                f"{entity_column}"
            )

        data = dataframe.copy()

                                                                        
                                  
                                                                        

        data["_original_position"] = np.arange(
            len(data)
        )

                                                                        
                               
                                                                        

        data[timestamp_column] = pd.to_datetime(
            data[timestamp_column]
        )

                                                                        
                                            
                                                                        

        sort_columns = []

        if entity_column is not None:
            sort_columns.append(
                entity_column
            )

        sort_columns.append(
            timestamp_column
        )

        data = data.sort_values(
            sort_columns
        )

                                                                        
                                         
                                                                        

        feature_names = [
            "dominant_frequency_cph",
            "dominant_period_hours",
            "dominant_amplitude_power",
            "total_spectral_energy",
            "spectral_centroid_cph",
            "spectral_bandwidth_cph",
            "spectral_entropy",
            "low_frequency_energy",
            "high_frequency_energy",
            "low_frequency_energy_ratio",
            "high_frequency_energy_ratio",
            "spectral_peak_count",
        ]

        output_columns = [
            f"{feature}_{suffix}"
            for feature in features
            for suffix in feature_names
        ]

        generated = pd.DataFrame(
            np.nan,
            index=data.index,
            columns=output_columns,
        )

                                                                        
                        
                                                                        

        if entity_column is None:

            groups = [
                (None, data)
            ]

        else:

            groups = data.groupby(
                entity_column,
                sort=False,
                dropna=False,
            )

                                                                        
                                                     
                                                                        

        for _, group in groups:

            timestamps = group[
                timestamp_column
            ]

            for feature in features:

                signal_features = (
                    self.generate_signal_features(
                        group[feature],
                        timestamps=timestamps,
                        prefix=feature,
                    )
                )

                generated.loc[
                    group.index,
                    signal_features.columns,
                ] = signal_features.values

                                                                        
                                           
                                                                        

        generated["_original_position"] = (
            data["_original_position"]
        )

        generated = (
            generated
            .sort_values(
                "_original_position"
            )
            .drop(
                columns="_original_position"
            )
        )

                                                                        
                                                             
                                                                        

        generated = generated.reset_index(
            drop=True
        )

        return generated

                                                                        
                                         
                                                                        

    def transform(
        self,
        dataframe: pd.DataFrame,
        features: list[str],
        timestamp_column: str = "timestamp",
        preserve_columns: list[str] | None = None,
        entity_column: str | None = None,
    ) -> pd.DataFrame:
        """
        Generate frequency features while preserving metadata.

        entity_column should normally be:

            bus_id
            line_id

        so that temporal windows never cross physical entities.
        """

        if preserve_columns is None:

            preserve_columns = []

        required = (
            features
            + [timestamp_column]
            + preserve_columns
        )

        if entity_column is not None:

            required.append(
                entity_column
            )

        missing = (
            set(required)
            - set(dataframe.columns)
        )

        if missing:

            raise ValueError(
                f"Missing columns: "
                f"{sorted(missing)}"
            )

        metadata = (
            dataframe[
                preserve_columns
            ].copy()
        )

        generated = self.generate(
            dataframe,
            features,
            timestamp_column=timestamp_column,
            entity_column=entity_column,
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
    def save(
        dataframe: pd.DataFrame,
        path: str | Path,
    ) -> Path:
        """
        Save frequency features.
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
        "FREQUENCY FEATURE ENGINEERING"
    )

    print(
        "=" * 70
    )

                                                                    
               
                                                                    

    loader = EDADataLoader()

    dataset = loader.load_all()

                                                                    
                                  
     
                        
     
                           
     
                                                                 
                                                                    

    config = FrequencyFeatureConfig(
        window_size=96,
        min_periods=32,
        low_frequency_max_cph=1.0,
        high_frequency_min_cph=2.0,
    )

    engineer = (
        FrequencyFeatureEngineer(
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
        "\nGenerating bus frequency features..."
    )

    bus_frequency_features = (
        engineer.transform(
            dataset.bus,
            bus_features,
            timestamp_column="timestamp",
            preserve_columns=[
                "timestamp",
                "bus_id",
            ],
            entity_column="bus_id",
        )
    )

    bus_path = engineer.save(
        bus_frequency_features,
        "data/features/bus/"
        "frequency_features.csv",
    )

    print(
        f"Saved bus features to:"
        f"\n{bus_path}"
    )

    print(
        f"\nBus feature matrix shape:"
        f" {bus_frequency_features.shape}"
    )

    print(
        "\nBus frequency features:"
    )

    print(
        bus_frequency_features.head(
            10
        ).to_string(
            index=False
        )
    )

                                                                    
                   
                                                                    

    line_features = [
        "current_from_ka",
        "current_to_ka",
        "loading_percent",
    ]

    print(
        "\nGenerating line frequency features..."
    )

    line_frequency_features = (
        engineer.transform(
            dataset.line,
            line_features,
            timestamp_column="timestamp",
            preserve_columns=[
                "timestamp",
                "line_id",
            ],
            entity_column="line_id",
        )
    )

    line_path = engineer.save(
        line_frequency_features,
        "data/features/line/"
        "frequency_features.csv",
    )

    print(
        f"Saved line features to:"
        f"\n{line_path}"
    )

    print(
        f"\nLine feature matrix shape:"
        f" {line_frequency_features.shape}"
    )

    print()
    print(
        "=" * 70
    )

    print(
        "FREQUENCY FEATURE ENGINEERING COMPLETE"
    )

    print(
        "=" * 70
    )
