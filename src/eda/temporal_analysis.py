"""
Temporal and seasonal analysis for power-grid measurements.

This module investigates how electrical measurements evolve over time.

Analyses
--------
1. Time-series profiles
2. Hourly aggregation
3. Daily profiles
4. Temporal variability
5. Ramp-rate analysis
6. Rolling statistics
7. Autocorrelation
8. Daily/weekly periodicity when sufficient data exist

Important
---------
This module does not classify anomalies.

It characterizes normal temporal structure that later models need
to distinguish from anomalous behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda.load_data import EDADataLoader


@dataclass
class TemporalStatistics:
    """
    Summary statistics describing temporal behavior.
    """

    feature: str

    mean: float
    std: float

    minimum: float
    maximum: float

    mean_absolute_change: float
    maximum_absolute_change: float

    mean_ramp_rate: float
    maximum_absolute_ramp_rate: float

    lag_1_autocorrelation: float
    lag_4_autocorrelation: float

    rolling_std_mean: float
    rolling_std_max: float


class TemporalAnalyzer:
    """
    Analyze temporal structure in power-grid measurements.
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
            "outputs/eda/temporal"
        ),
        rolling_window: int = 4,
    ) -> None:

        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if rolling_window < 2:
            raise ValueError(
                "rolling_window must be >= 2."
            )

        self.rolling_window = rolling_window

                                                                        
                         
                                                                        

    @staticmethod
    def aggregate_system(
        dataframe: pd.DataFrame,
        feature: str,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """
        Aggregate a measurement across buses/lines at each timestamp.

        Parameters
        ----------
        dataframe:
            Measurement dataframe.

        feature:
            Electrical measurement.

        aggregation:
            mean, median, min, max, std.
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

        if aggregation not in {
            "mean",
            "median",
            "min",
            "max",
            "std",
        }:
            raise ValueError(
                "Unsupported aggregation."
            )

        data = dataframe.copy()

        data["timestamp"] = pd.to_datetime(
            data["timestamp"]
        )

        series = (
            data
            .groupby("timestamp")[feature]
            .agg(aggregation)
            .sort_index()
        )

        return series.to_frame(
            name=feature
        )

                                                                        
                         
                                                                        

    def calculate_statistics(
        self,
        dataframe: pd.DataFrame,
        features: list[str],
    ) -> list[TemporalStatistics]:
        """
        Calculate temporal statistics for system-level signals.
        """

        results = []

        for feature in features:

            series_dataframe = (
                self.aggregate_system(
                    dataframe,
                    feature,
                    aggregation="mean",
                )
            )

            series = (
                series_dataframe[feature]
                .astype(float)
                .dropna()
            )

            if len(series) < 2:
                continue

                                                                        
                              
                                                                        

            difference = series.diff().dropna()

            absolute_difference = (
                difference.abs()
            )

                                                                        
                                                    
                                                                        

            timestamps = series.index

            if len(timestamps) >= 2:

                delta_seconds = (
                    timestamps[1]
                    - timestamps[0]
                ).total_seconds()

                if delta_seconds <= 0:
                    delta_seconds = 1.0

            else:

                delta_seconds = 1.0

            ramp_rate = (
                difference
                / delta_seconds
            )

                                                                        
                             
                                                                        

            lag_1 = self._autocorrelation(
                series,
                lag=1,
            )

            lag_4 = self._autocorrelation(
                series,
                lag=4,
            )

                                                                        
                                        
                                                                        

            rolling_std = (
                series
                .rolling(
                    self.rolling_window
                )
                .std()
                .dropna()
            )

            results.append(
                TemporalStatistics(
                    feature=feature,

                    mean=float(
                        series.mean()
                    ),

                    std=float(
                        series.std(
                            ddof=1
                        )
                    ),

                    minimum=float(
                        series.min()
                    ),

                    maximum=float(
                        series.max()
                    ),

                    mean_absolute_change=float(
                        absolute_difference.mean()
                    ),

                    maximum_absolute_change=float(
                        absolute_difference.max()
                    ),

                    mean_ramp_rate=float(
                        ramp_rate.abs().mean()
                    ),

                    maximum_absolute_ramp_rate=float(
                        ramp_rate.abs().max()
                    ),

                    lag_1_autocorrelation=lag_1,

                    lag_4_autocorrelation=lag_4,

                    rolling_std_mean=float(
                        rolling_std.mean()
                    ),

                    rolling_std_max=float(
                        rolling_std.max()
                    ),
                )
            )

        return results

                                                                        
                     
                                                                        

    @staticmethod
    def _autocorrelation(
        series: pd.Series,
        lag: int,
    ) -> float:
        """
        Calculate autocorrelation at a specified lag.
        """

        if len(series) <= lag:
            return float("nan")

        value = series.autocorr(
            lag=lag
        )

        if pd.isna(value):
            return float("nan")

        return float(value)

                                                                        
                    
                                                                        

    def hourly_profile(
        self,
        dataframe: pd.DataFrame,
        feature: str,
    ) -> pd.DataFrame:
        """
        Calculate mean/std/min/max for each hour of the day.

        For datasets spanning multiple days, this provides a daily
        operating profile.
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

        data["hour"] = (
            data["timestamp"].dt.hour
        )

        system = (
            data
            .groupby("timestamp")[feature]
            .mean()
            .to_frame(feature)
        )

        system["hour"] = (
            system.index.hour
        )

        profile = (
            system
            .groupby("hour")[feature]
            .agg(
                [
                    "mean",
                    "std",
                    "min",
                    "max",
                ]
            )
        )

        return profile

                                                                        
                        
                                                                        

    def plot_hourly_profile(
        self,
        dataframe: pd.DataFrame,
        feature: str,
        show: bool = False,
    ) -> Path:
        """
        Plot the average hourly profile.
        """

        profile = self.hourly_profile(
            dataframe,
            feature,
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        hours = profile.index

        ax.plot(
            hours,
            profile["mean"],
            marker="o",
            label="Mean",
        )

        if len(profile) > 1:

            lower = (
                profile["mean"]
                - profile["std"].fillna(0)
            )

            upper = (
                profile["mean"]
                + profile["std"].fillna(0)
            )

            ax.fill_between(
                hours,
                lower,
                upper,
                alpha=0.20,
                label="±1 std",
            )

        ax.set_title(
            f"Hourly Profile: {feature}"
        )

        ax.set_xlabel(
            "Hour of Day"
        )

        ax.set_ylabel(
            feature
        )

        ax.set_xticks(
            np.arange(0, 24, 2)
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        path = (
            self.output_directory
            / f"{feature}_hourly_profile.png"
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

                                                                        
                          
                                                                        

    def plot_time_series(
        self,
        dataframe: pd.DataFrame,
        feature: str,
        aggregation: str = "mean",
        show: bool = False,
    ) -> Path:
        """
        Plot system-level time series.
        """

        series = self.aggregate_system(
            dataframe,
            feature,
            aggregation,
        )

        fig, ax = plt.subplots(
            figsize=(11, 5)
        )

        ax.plot(
            series.index,
            series[feature],
        )

        ax.set_title(
            f"Temporal Profile: {feature}"
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            feature
        )

        ax.grid(
            alpha=0.25
        )

        fig.autofmt_xdate()

        fig.tight_layout()

        path = (
            self.output_directory
            / f"{feature}_time_series.png"
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

                                                                        
                        
                                                                        

    def plot_ramp_rate(
        self,
        dataframe: pd.DataFrame,
        feature: str,
        show: bool = False,
    ) -> Path:
        """
        Plot first-order temporal changes.

        Approximation:

            dx/dt ≈ (x_t - x_{t-1}) / Δt
        """

        series_dataframe = (
            self.aggregate_system(
                dataframe,
                feature,
                aggregation="mean",
            )
        )

        series = (
            series_dataframe[feature]
            .astype(float)
        )

        timestamps = series.index

        if len(series) < 2:
            raise ValueError(
                "At least two timestamps are required."
            )

        delta_seconds = (
            timestamps.to_series()
            .diff()
            .dt.total_seconds()
        )

        delta_seconds.iloc[0] = (
            delta_seconds.iloc[1]
        )

        ramp = (
            series.diff()
            / delta_seconds.values
        )

        ramp = ramp.iloc[1:]

        fig, ax = plt.subplots(
            figsize=(11, 5)
        )

        ax.plot(
            timestamps[1:],
            ramp,
        )

        ax.axhline(
            0.0,
            linestyle="--",
        )

        ax.set_title(
            f"Ramp Rate: {feature}"
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            f"d({feature}) / dt"
        )

        ax.grid(
            alpha=0.25
        )

        fig.autofmt_xdate()

        fig.tight_layout()

        path = (
            self.output_directory
            / f"{feature}_ramp_rate.png"
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

                                                                        
                        
                                                                        

    def calculate_rolling_statistics(
        self,
        dataframe: pd.DataFrame,
        feature: str,
    ) -> pd.DataFrame:
        """
        Calculate rolling mean and standard deviation.
        """

        series_dataframe = (
            self.aggregate_system(
                dataframe,
                feature,
                aggregation="mean",
            )
        )

        series = (
            series_dataframe[feature]
            .astype(float)
        )

        result = pd.DataFrame(
            index=series.index
        )

        result["value"] = series

        result["rolling_mean"] = (
            series
            .rolling(
                self.rolling_window
            )
            .mean()
        )

        result["rolling_std"] = (
            series
            .rolling(
                self.rolling_window
            )
            .std()
        )

        return result

                                                                        
                             
                                                                        

    def plot_rolling_statistics(
        self,
        dataframe: pd.DataFrame,
        feature: str,
        show: bool = False,
    ) -> Path:
        """
        Plot the signal together with rolling mean/std.
        """

        result = (
            self.calculate_rolling_statistics(
                dataframe,
                feature,
            )
        )

        fig, ax = plt.subplots(
            figsize=(11, 5)
        )

        ax.plot(
            result.index,
            result["value"],
            label="Value",
        )

        ax.plot(
            result.index,
            result["rolling_mean"],
            linestyle="--",
            label="Rolling mean",
        )

        ax.fill_between(
            result.index,
            result["rolling_mean"]
            - result["rolling_std"].fillna(0),
            result["rolling_mean"]
            + result["rolling_std"].fillna(0),
            alpha=0.20,
            label="± rolling std",
        )

        ax.set_title(
            f"Rolling Statistics: {feature}"
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            feature
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.autofmt_xdate()

        fig.tight_layout()

        path = (
            self.output_directory
            / f"{feature}_rolling_statistics.png"
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

                                                                        
                          
                                                                        

    def plot_autocorrelation(
        self,
        dataframe: pd.DataFrame,
        feature: str,
        max_lag: int = 24,
        show: bool = False,
    ) -> Path:
        """
        Plot autocorrelation up to max_lag.

        For 15-minute sampling:

            lag 4  = 1 hour
            lag 24 = 6 hours
            lag 96 = 24 hours
        """

        series_dataframe = (
            self.aggregate_system(
                dataframe,
                feature,
                aggregation="mean",
            )
        )

        series = (
            series_dataframe[feature]
            .astype(float)
            .dropna()
        )

        max_lag = min(
            max_lag,
            len(series) - 1,
        )

        lags = np.arange(
            0,
            max_lag + 1,
        )

        values = np.array(
            [
                series.autocorr(
                    lag=int(lag)
                )
                for lag in lags
            ]
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.stem(
            lags,
            values,
        )

        ax.axhline(
            0.0,
            linestyle="--",
        )

        ax.set_title(
            f"Autocorrelation: {feature}"
        )

        ax.set_xlabel(
            "Lag"
        )

        ax.set_ylabel(
            "Autocorrelation"
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        path = (
            self.output_directory
            / f"{feature}_autocorrelation.png"
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

                                                                        
                       
                                                                        

    def daily_profile(
        self,
        dataframe: pd.DataFrame,
        feature: str,
    ) -> pd.DataFrame:
        """
        Calculate mean behavior by day.

        Useful once the dataset spans multiple days.
        """

        data = dataframe.copy()

        data["timestamp"] = pd.to_datetime(
            data["timestamp"]
        )

        system = (
            data
            .groupby("timestamp")[feature]
            .mean()
            .to_frame(feature)
        )

        system["date"] = (
            system.index.date
        )

        system["time"] = (
            system.index.strftime("%H:%M")
        )

        profile = (
            system
            .pivot_table(
                index="time",
                columns="date",
                values=feature,
            )
        )

        return profile

                                                                        
                     
                                                                        

    @staticmethod
    def statistics_to_dataframe(
        statistics: list[TemporalStatistics],
    ) -> pd.DataFrame:
        """
        Convert temporal statistics to a dataframe.
        """

        return pd.DataFrame(
            [
                {
                    "feature": stat.feature,
                    "mean": stat.mean,
                    "std": stat.std,
                    "minimum": stat.minimum,
                    "maximum": stat.maximum,
                    "mean_absolute_change": (
                        stat.mean_absolute_change
                    ),
                    "maximum_absolute_change": (
                        stat.maximum_absolute_change
                    ),
                    "mean_ramp_rate": (
                        stat.mean_ramp_rate
                    ),
                    "maximum_absolute_ramp_rate": (
                        stat.maximum_absolute_ramp_rate
                    ),
                    "lag_1_autocorrelation": (
                        stat.lag_1_autocorrelation
                    ),
                    "lag_4_autocorrelation": (
                        stat.lag_4_autocorrelation
                    ),
                    "rolling_std_mean": (
                        stat.rolling_std_mean
                    ),
                    "rolling_std_max": (
                        stat.rolling_std_max
                    ),
                }
                for stat in statistics
            ]
        )

    def save_statistics(
        self,
        statistics: list[TemporalStatistics],
        filename: str = (
            "temporal_statistics.csv"
        ),
    ) -> Path:
        """
        Save temporal statistics.
        """

        dataframe = (
            self.statistics_to_dataframe(
                statistics
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

    analyzer = TemporalAnalyzer(
        rolling_window=4
    )

                                                                    
                           
                                                                    

    print()
    print(
        "=" * 70
    )

    print(
        "BUS TEMPORAL ANALYSIS"
    )

    print(
        "=" * 70
    )

    bus_statistics = (
        analyzer.calculate_statistics(
            dataset.bus,
            analyzer.DEFAULT_BUS_FEATURES,
        )
    )

    bus_statistics_dataframe = (
        analyzer.statistics_to_dataframe(
            bus_statistics
        )
    )

    print(
        bus_statistics_dataframe.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}",
        )
    )

    analyzer.save_statistics(
        bus_statistics,
        filename="bus_temporal_statistics.csv",
    )

                                                                    
                    
                                                                    

    for feature in (
        analyzer.DEFAULT_BUS_FEATURES
    ):

        print(
            f"\nAnalyzing {feature}..."
        )

        analyzer.plot_time_series(
            dataset.bus,
            feature,
        )

        analyzer.plot_hourly_profile(
            dataset.bus,
            feature,
        )

        analyzer.plot_ramp_rate(
            dataset.bus,
            feature,
        )

        analyzer.plot_rolling_statistics(
            dataset.bus,
            feature,
        )

        analyzer.plot_autocorrelation(
            dataset.bus,
            feature,
            max_lag=24,
        )

                                                                    
                            
                                                                    

    print()
    print(
        "=" * 70
    )

    print(
        "LINE TEMPORAL ANALYSIS"
    )

    print(
        "=" * 70
    )

    line_statistics = (
        analyzer.calculate_statistics(
            dataset.line,
            analyzer.DEFAULT_LINE_FEATURES,
        )
    )

    line_statistics_dataframe = (
        analyzer.statistics_to_dataframe(
            line_statistics
        )
    )

    print(
        line_statistics_dataframe.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}",
        )
    )

    analyzer.save_statistics(
        line_statistics,
        filename="line_temporal_statistics.csv",
    )

    for feature in (
        analyzer.DEFAULT_LINE_FEATURES
    ):

        print(
            f"\nAnalyzing {feature}..."
        )

        analyzer.plot_time_series(
            dataset.line,
            feature,
        )

        analyzer.plot_hourly_profile(
            dataset.line,
            feature,
        )

        analyzer.plot_ramp_rate(
            dataset.line,
            feature,
        )

        analyzer.plot_rolling_statistics(
            dataset.line,
            feature,
        )

        analyzer.plot_autocorrelation(
            dataset.line,
            feature,
            max_lag=24,
        )

    print()
    print(
        "=" * 70
    )

    print(
        "TEMPORAL ANALYSIS COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nResults saved to:"
        f"\n{analyzer.output_directory}"
    )