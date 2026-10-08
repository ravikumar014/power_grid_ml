"""
Frequency-domain analysis for power-grid measurements.

Uses the Fast Fourier Transform (FFT) to identify periodic
components and frequency-domain energy in electrical signals.

Analyses
--------
- Voltage magnitude
- Voltage angle
- Active power
- Reactive power
- Line current
- Line loading

Outputs
-------
- Frequency spectra
- Power spectral density-like magnitude plots
- Dominant frequency components
- Dominant periods
- Spectral energy statistics

Important
---------
FFT results are meaningful only when the signal has sufficient
temporal length. A single simulated day is enough for basic
pipeline verification, but longer simulations are required for
reliable seasonal interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda.load_data import EDADataLoader


@dataclass
class FrequencyResult:
    """Summary of the frequency spectrum."""

    feature: str

    sampling_interval_seconds: float

    dominant_frequency_hz: float

    dominant_period_seconds: float

    dominant_period_minutes: float

    dominant_period_hours: float

    dominant_amplitude: float

    total_spectral_energy: float


class FrequencyAnalyzer:
    """
    Analyze temporal signals in the frequency domain.
    """

    DEFAULT_BUS_FEATURES = [
        "voltage_pu",
        "angle_deg",
        "active_power_mw",
        "reactive_power_mvar",
    ]

    DEFAULT_LINE_FEATURES = [
        "current_from_ka",
        "current_to_ka",
        "loading_percent",
    ]

    def __init__(
        self,
        output_directory: str | Path = (
            "outputs/eda/frequency"
        ),
    ) -> None:

        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

                                                                        
                           
                                                                        

    @staticmethod
    def aggregate_system(
        dataframe: pd.DataFrame,
        feature: str,
    ) -> pd.Series:
        """
        Aggregate measurements across physical entities.

        The mean value at each timestamp is used to obtain a
        system-level signal.
        """

        required = {
            "timestamp",
            feature,
        }

        missing = (
            required
            - set(dataframe.columns)
        )

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

        data = dataframe.copy()

        data["timestamp"] = pd.to_datetime(
            data["timestamp"]
        )

        series = (
            data
            .groupby("timestamp")[feature]
            .mean()
            .sort_index()
        )

        return series.astype(float)

                                                                        
                       
                                                                        

    @staticmethod
    def get_sampling_interval(
        series: pd.Series,
    ) -> float:
        """
        Return the sampling interval in seconds.
        """

        if len(series.index) < 2:
            raise ValueError(
                "At least two timestamps are required."
            )

        timestamps = pd.DatetimeIndex(
            series.index
        )

        differences = (
            timestamps
            .to_series()
            .diff()
            .dt.total_seconds()
            .dropna()
        )

        if differences.empty:
            raise ValueError(
                "Unable to determine sampling interval."
            )

                                                            
                                                                         
        interval = float(
            differences.median()
        )

        if interval <= 0:
            raise ValueError(
                "Sampling interval must be positive."
            )

        return interval

                                                                        
         
                                                                        

    def compute_fft(
        self,
        series: pd.Series,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
    ]:
        """
        Compute the one-sided FFT spectrum.

        Returns
        -------
        frequencies:
            Frequencies in Hz.

        amplitudes:
            Single-sided amplitude spectrum.

        sampling_interval:
            Sampling interval in seconds.
        """

        series = series.dropna()

        if len(series) < 4:
            raise ValueError(
                "At least four samples are required for FFT."
            )

        sampling_interval = (
            self.get_sampling_interval(
                series
            )
        )

        values = series.to_numpy(
            dtype=np.float64
        )

                                                                        
                          
         
                                                            
                                                                        

        values = (
            values
            - np.mean(values)
        )

        n = len(values)

                                                                        
                   
                                                                        

        fft_values = np.fft.rfft(
            values
        )

        frequencies = np.fft.rfftfreq(
            n,
            d=sampling_interval,
        )

        amplitudes = (
            2.0
            / n
            * np.abs(fft_values)
        )

                                             
        if len(amplitudes) > 0:
            amplitudes[0] /= 2.0

                                                              
        if (
            n % 2 == 0
            and len(amplitudes) > 1
        ):
            amplitudes[-1] /= 2.0

        return (
            frequencies,
            amplitudes,
            sampling_interval,
        )

                                                                        
                        
                                                                        

    @staticmethod
    def find_dominant_frequency(
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
    ) -> tuple[float, float]:
        """
        Find the strongest non-zero frequency component.
        """

        if len(frequencies) != len(amplitudes):
            raise ValueError(
                "Frequency and amplitude arrays must "
                "have equal length."
            )

        if len(frequencies) < 2:
            raise ValueError(
                "At least two frequency bins are required."
            )

                    
        amplitudes_nonzero = amplitudes[1:]

        if len(amplitudes_nonzero) == 0:
            return 0.0, 0.0

        index = (
            np.argmax(
                amplitudes_nonzero
            )
            + 1
        )

        return (
            float(frequencies[index]),
            float(amplitudes[index]),
        )

                                                                        
                             
                                                                        

    def analyze(
        self,
        dataframe: pd.DataFrame,
        feature: str,
    ) -> FrequencyResult:
        """
        Perform complete frequency-domain analysis.
        """

        series = self.aggregate_system(
            dataframe,
            feature,
        )

        (
            frequencies,
            amplitudes,
            sampling_interval,
        ) = self.compute_fft(
            series
        )

        (
            dominant_frequency,
            dominant_amplitude,
        ) = self.find_dominant_frequency(
            frequencies,
            amplitudes,
        )

        if dominant_frequency > 0:

            dominant_period_seconds = (
                1.0
                / dominant_frequency
            )

        else:

            dominant_period_seconds = float(
                "inf"
            )

        total_spectral_energy = float(
            np.sum(
                amplitudes**2
            )
        )

        return FrequencyResult(
            feature=feature,

            sampling_interval_seconds=(
                sampling_interval
            ),

            dominant_frequency_hz=(
                dominant_frequency
            ),

            dominant_period_seconds=(
                dominant_period_seconds
            ),

            dominant_period_minutes=(
                dominant_period_seconds
                / 60.0
            ),

            dominant_period_hours=(
                dominant_period_seconds
                / 3600.0
            ),

            dominant_amplitude=(
                dominant_amplitude
            ),

            total_spectral_energy=(
                total_spectral_energy
            ),
        )

                                                                        
                        
                                                                        

    def plot_spectrum(
        self,
        dataframe: pd.DataFrame,
        feature: str,
        max_frequency_hz: float | None = None,
        show: bool = False,
    ) -> Path:
        """
        Plot the one-sided amplitude spectrum.
        """

        series = self.aggregate_system(
            dataframe,
            feature,
        )

        (
            frequencies,
            amplitudes,
            sampling_interval,
        ) = self.compute_fft(
            series
        )

        if max_frequency_hz is not None:

            mask = (
                frequencies
                <= max_frequency_hz
            )

            frequencies = (
                frequencies[mask]
            )

            amplitudes = (
                amplitudes[mask]
            )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            frequencies,
            amplitudes,
        )

        ax.set_title(
            f"FFT Spectrum: {feature}"
        )

        ax.set_xlabel(
            "Frequency (Hz)"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        path = (
            self.output_directory
            / f"{feature}_fft_spectrum.png"
        )

        fig.savefig(
            path,
            dpi=200,
            bbox_inches="tight",
        )

        if show:
            plt.show()

        plt.close(fig)

        return path

                                                                        
                             
                                                                        

    def plot_spectrum_cycles_per_hour(
        self,
        dataframe: pd.DataFrame,
        feature: str,
        max_cycles_per_hour: float = 8.0,
        show: bool = False,
    ) -> Path:
        """
        Plot spectrum using cycles/hour.

        This is easier to interpret for grid time-series data.

        Examples
        --------
        1 cycle/hour  -> 1 hour period
        1/24 cycle/h  -> 24 hour period
        1/6 cycle/h   -> 6 hour period
        """

        series = self.aggregate_system(
            dataframe,
            feature,
        )

        (
            frequencies,
            amplitudes,
            _,
        ) = self.compute_fft(
            series
        )

        cycles_per_hour = (
            frequencies * 3600.0
        )

        mask = (
            cycles_per_hour
            <= max_cycles_per_hour
        )

        cycles_per_hour = (
            cycles_per_hour[mask]
        )

        amplitudes = (
            amplitudes[mask]
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            cycles_per_hour,
            amplitudes,
        )

        ax.set_title(
            f"Frequency Spectrum: {feature}"
        )

        ax.set_xlabel(
            "Frequency (cycles/hour)"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        path = (
            self.output_directory
            / f"{feature}_cycles_per_hour.png"
        )

        fig.savefig(
            path,
            dpi=200,
            bbox_inches="tight",
        )

        if show:
            plt.show()

        plt.close(fig)

        return path

                                                                        
                     
                                                                        

    def plot_period_spectrum(
        self,
        dataframe: pd.DataFrame,
        feature: str,
        min_period_minutes: float = 15.0,
        max_period_hours: float = 24.0,
        show: bool = False,
    ) -> Path:
        """
        Plot amplitude against period.

        This representation is particularly intuitive for detecting
        daily and sub-daily periodic components.
        """

        series = self.aggregate_system(
            dataframe,
            feature,
        )

        (
            frequencies,
            amplitudes,
            _,
        ) = self.compute_fft(
            series
        )

                    
        frequencies = frequencies[1:]
        amplitudes = amplitudes[1:]

        valid = (
            frequencies > 0
        )

        frequencies = (
            frequencies[valid]
        )

        amplitudes = (
            amplitudes[valid]
        )

        periods_minutes = (
            1.0
            / frequencies
            / 60.0
        )

        periods_hours = (
            periods_minutes
            / 60.0
        )

        valid = (
            (periods_minutes >= min_period_minutes)
            & (
                periods_hours
                <= max_period_hours
            )
        )

        periods_hours = (
            periods_hours[valid]
        )

        amplitudes = (
            amplitudes[valid]
        )

        if len(periods_hours) == 0:

            raise ValueError(
                "No frequency components fall within "
                "the requested period range."
            )

                                             
        order = np.argsort(
            periods_hours
        )

        periods_hours = (
            periods_hours[order]
        )

        amplitudes = (
            amplitudes[order]
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            periods_hours,
            amplitudes,
        )

        ax.set_title(
            f"Period Spectrum: {feature}"
        )

        ax.set_xlabel(
            "Period (hours)"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        path = (
            self.output_directory
            / f"{feature}_period_spectrum.png"
        )

        fig.savefig(
            path,
            dpi=200,
            bbox_inches="tight",
        )

        if show:
            plt.show()

        plt.close(fig)

        return path

                                                                        
                 
                                                                        

    @staticmethod
    def result_to_dataframe(
        results: list[FrequencyResult],
    ) -> pd.DataFrame:
        """
        Convert frequency results to a dataframe.
        """

        return pd.DataFrame(
            [
                {
                    "feature": result.feature,
                    "sampling_interval_seconds": (
                        result.sampling_interval_seconds
                    ),
                    "dominant_frequency_hz": (
                        result.dominant_frequency_hz
                    ),
                    "dominant_period_seconds": (
                        result.dominant_period_seconds
                    ),
                    "dominant_period_minutes": (
                        result.dominant_period_minutes
                    ),
                    "dominant_period_hours": (
                        result.dominant_period_hours
                    ),
                    "dominant_amplitude": (
                        result.dominant_amplitude
                    ),
                    "total_spectral_energy": (
                        result.total_spectral_energy
                    ),
                }
                for result in results
            ]
        )

    def save_results(
        self,
        results: list[FrequencyResult],
        filename: str = (
            "frequency_statistics.csv"
        ),
    ) -> Path:
        """
        Save frequency-domain summary statistics.
        """

        dataframe = (
            self.result_to_dataframe(
                results
            )
        )

        path = (
            self.output_directory
            / filename
        )

        dataframe.to_csv(
            path,
            index=False,
        )

        return path


                                                                        
      
                                                                        


if __name__ == "__main__":

    loader = EDADataLoader()

    dataset = loader.load_all()

    analyzer = FrequencyAnalyzer()

    all_results: list[
        FrequencyResult
    ] = []

                                                                    
                  
                                                                    

    print()
    print(
        "=" * 70
    )

    print(
        "BUS FREQUENCY ANALYSIS"
    )

    print(
        "=" * 70
    )

    for feature in (
        analyzer.DEFAULT_BUS_FEATURES
    ):

        print(
            f"\nAnalyzing: {feature}"
        )

        result = analyzer.analyze(
            dataset.bus,
            feature,
        )

        all_results.append(
            result
        )

        print(
            f"  Sampling interval : "
            f"{result.sampling_interval_seconds:.2f} s"
        )

        print(
            f"  Dominant frequency: "
            f"{result.dominant_frequency_hz:.8f} Hz"
        )

        print(
            f"  Dominant period   : "
            f"{result.dominant_period_hours:.4f} hours"
        )

        print(
            f"  Dominant amplitude: "
            f"{result.dominant_amplitude:.6f}"
        )

        analyzer.plot_spectrum(
            dataset.bus,
            feature,
        )

        analyzer.plot_spectrum_cycles_per_hour(
            dataset.bus,
            feature,
        )

        analyzer.plot_period_spectrum(
            dataset.bus,
            feature,
        )

                                                                    
                   
                                                                    

    print()
    print(
        "=" * 70
    )

    print(
        "LINE FREQUENCY ANALYSIS"
    )

    print(
        "=" * 70
    )

    for feature in (
        analyzer.DEFAULT_LINE_FEATURES
    ):

        print(
            f"\nAnalyzing: {feature}"
        )

        result = analyzer.analyze(
            dataset.line,
            feature,
        )

        all_results.append(
            result
        )

        print(
            f"  Sampling interval : "
            f"{result.sampling_interval_seconds:.2f} s"
        )

        print(
            f"  Dominant frequency: "
            f"{result.dominant_frequency_hz:.8f} Hz"
        )

        print(
            f"  Dominant period   : "
            f"{result.dominant_period_hours:.4f} hours"
        )

        print(
            f"  Dominant amplitude: "
            f"{result.dominant_amplitude:.6f}"
        )

        analyzer.plot_spectrum(
            dataset.line,
            feature,
        )

        analyzer.plot_spectrum_cycles_per_hour(
            dataset.line,
            feature,
        )

        analyzer.plot_period_spectrum(
            dataset.line,
            feature,
        )

                                                                    
                      
                                                                    

    result_path = analyzer.save_results(
        all_results
    )

    print()
    print(
        "=" * 70
    )

    print(
        "FREQUENCY ANALYSIS COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nResults saved to:"
        f"\n{result_path}"
    )