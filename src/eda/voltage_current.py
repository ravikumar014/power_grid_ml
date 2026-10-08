"""
Power-system-specific voltage and current analysis.

This module analyzes:

Bus-level
---------
- Voltage magnitude
- Voltage angle
- Voltage variation
- Per-bus voltage statistics
- Temporal voltage profiles

Line-level
----------
- Current from/to
- Line loading
- Per-line loading statistics
- Temporal loading profiles

The purpose is exploratory analysis, not anomaly labeling.

A statistically unusual voltage/current value is NOT automatically
classified as a grid anomaly.

Outputs
-------
CSV:
    Per-bus voltage statistics
    Per-line loading statistics

Plots:
    System voltage profile
    Selected bus voltage profiles
    Bus voltage variation
    System line loading
    Selected line loading profiles
    Line loading variation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda.load_data import EDADataLoader


                                                                        
                 
                                                                        


@dataclass
class VoltageAnalysisResult:
    """
    Summary of voltage behavior across buses.
    """

    bus_id: int

    mean_voltage: float
    std_voltage: float

    minimum_voltage: float
    maximum_voltage: float

    voltage_range: float

    mean_angle: float
    std_angle: float


@dataclass
class LineAnalysisResult:
    """
    Summary of line loading/current behavior.
    """

    line_id: int

    mean_loading: float
    std_loading: float

    minimum_loading: float
    maximum_loading: float

    loading_range: float

    mean_current_from: float
    mean_current_to: float


                                                                        
          
                                                                        


class VoltageCurrentAnalyzer:
    """
    Analyze voltage and current/loading behavior in the grid.
    """

    def __init__(
        self,
        output_directory: str | Path = (
            "outputs/eda/voltage_current"
        ),
        voltage_min: float = 0.95,
        voltage_max: float = 1.05,
        loading_limit: float = 100.0,
    ) -> None:

        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

                                                                        
                                              
         
                                               
         
                                                                  
                                       
                                                                        

        self.voltage_min = voltage_min
        self.voltage_max = voltage_max

        self.loading_limit = loading_limit

                                                                        
                        
                                                                        

    def analyze_voltage(
        self,
        dataframe: pd.DataFrame,
    ) -> list[VoltageAnalysisResult]:
        """
        Calculate per-bus voltage statistics.
        """

        self._validate_columns(
            dataframe,
            [
                "bus_id",
                "voltage_pu",
                "angle_deg",
            ],
        )

        results = []

        for bus_id, group in dataframe.groupby(
            "bus_id",
            sort=True,
        ):

            voltage = group[
                "voltage_pu"
            ].astype(float)

            angle = group[
                "angle_deg"
            ].astype(float)

            results.append(
                VoltageAnalysisResult(
                    bus_id=int(bus_id),

                    mean_voltage=float(
                        voltage.mean()
                    ),

                    std_voltage=float(
                        voltage.std(
                            ddof=1
                        )
                    ),

                    minimum_voltage=float(
                        voltage.min()
                    ),

                    maximum_voltage=float(
                        voltage.max()
                    ),

                    voltage_range=float(
                        voltage.max()
                        - voltage.min()
                    ),

                    mean_angle=float(
                        angle.mean()
                    ),

                    std_angle=float(
                        angle.std(
                            ddof=1
                        )
                    ),
                )
            )

        return results

                                                                        
                     
                                                                        

    def analyze_lines(
        self,
        dataframe: pd.DataFrame,
    ) -> list[LineAnalysisResult]:
        """
        Calculate per-line current/loading statistics.
        """

        self._validate_columns(
            dataframe,
            [
                "line_id",
                "current_from_ka",
                "current_to_ka",
                "loading_percent",
            ],
        )

        results = []

        for line_id, group in dataframe.groupby(
            "line_id",
            sort=True,
        ):

            loading = group[
                "loading_percent"
            ].astype(float)

            current_from = group[
                "current_from_ka"
            ].astype(float)

            current_to = group[
                "current_to_ka"
            ].astype(float)

            results.append(
                LineAnalysisResult(
                    line_id=int(line_id),

                    mean_loading=float(
                        loading.mean()
                    ),

                    std_loading=float(
                        loading.std(
                            ddof=1
                        )
                    ),

                    minimum_loading=float(
                        loading.min()
                    ),

                    maximum_loading=float(
                        loading.max()
                    ),

                    loading_range=float(
                        loading.max()
                        - loading.min()
                    ),

                    mean_current_from=float(
                        current_from.mean()
                    ),

                    mean_current_to=float(
                        current_to.mean()
                    ),
                )
            )

        return results

                                                                        
                                  
                                                                        

    @staticmethod
    def voltage_results_dataframe(
        results: list[VoltageAnalysisResult],
    ) -> pd.DataFrame:

        return pd.DataFrame(
            [
                {
                    "bus_id": result.bus_id,
                    "mean_voltage_pu": result.mean_voltage,
                    "std_voltage_pu": result.std_voltage,
                    "minimum_voltage_pu": result.minimum_voltage,
                    "maximum_voltage_pu": result.maximum_voltage,
                    "voltage_range_pu": result.voltage_range,
                    "mean_angle_deg": result.mean_angle,
                    "std_angle_deg": result.std_angle,
                }
                for result in results
            ]
        )

                                                                        
                               
                                                                        

    @staticmethod
    def line_results_dataframe(
        results: list[LineAnalysisResult],
    ) -> pd.DataFrame:

        return pd.DataFrame(
            [
                {
                    "line_id": result.line_id,
                    "mean_loading_percent": result.mean_loading,
                    "std_loading_percent": result.std_loading,
                    "minimum_loading_percent": result.minimum_loading,
                    "maximum_loading_percent": result.maximum_loading,
                    "loading_range_percent": result.loading_range,
                    "mean_current_from_ka": result.mean_current_from,
                    "mean_current_to_ka": result.mean_current_to,
                }
                for result in results
            ]
        )

                                                                        
                                 
                                                                        

    def plot_system_voltage(
        self,
        dataframe: pd.DataFrame,
        show: bool = False,
    ) -> Path:
        """
        Plot the voltage envelope across all buses.

        Shows:
            minimum bus voltage
            mean bus voltage
            maximum bus voltage
        """

        self._validate_columns(
            dataframe,
            [
                "timestamp",
                "voltage_pu",
            ],
        )

        grouped = (
            dataframe
            .groupby("timestamp")["voltage_pu"]
            .agg(
                [
                    "min",
                    "mean",
                    "max",
                ]
            )
            .reset_index()
        )

        fig, ax = plt.subplots(
            figsize=(11, 5)
        )

        ax.plot(
            grouped["timestamp"],
            grouped["mean"],
            label="Mean voltage",
        )

        ax.fill_between(
            grouped["timestamp"],
            grouped["min"],
            grouped["max"],
            alpha=0.20,
            label="Bus voltage range",
        )

        ax.axhline(
            self.voltage_min,
            linestyle="--",
            label="Lower reference limit",
        )

        ax.axhline(
            self.voltage_max,
            linestyle="--",
            label="Upper reference limit",
        )

        ax.set_title(
            "System-Wide Voltage Profile"
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "Voltage (p.u.)"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.autofmt_xdate()

        fig.tight_layout()

        path = (
            self.output_directory
            / "system_voltage_profile.png"
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

                                                                        
                                   
                                                                        

    def plot_bus_voltage_profiles(
        self,
        dataframe: pd.DataFrame,
        bus_ids: list[int] | None = None,
        show: bool = False,
    ) -> Path:
        """
        Plot voltage profiles for selected buses.

        If bus_ids is None, the buses with the largest voltage
        variation are selected automatically.
        """

        self._validate_columns(
            dataframe,
            [
                "timestamp",
                "bus_id",
                "voltage_pu",
            ],
        )

        if bus_ids is None:

            variation = (
                dataframe
                .groupby("bus_id")["voltage_pu"]
                .agg(
                    lambda x:
                    x.max() - x.min()
                )
                .sort_values(
                    ascending=False
                )
            )

            bus_ids = (
                variation
                .head(5)
                .index
                .tolist()
            )

        fig, ax = plt.subplots(
            figsize=(11, 5)
        )

        for bus_id in bus_ids:

            bus_data = (
                dataframe[
                    dataframe["bus_id"]
                    == bus_id
                ]
                .sort_values("timestamp")
            )

            if bus_data.empty:
                continue

            ax.plot(
                bus_data["timestamp"],
                bus_data["voltage_pu"],
                label=f"Bus {bus_id}",
            )

        ax.axhline(
            self.voltage_min,
            linestyle="--",
            label="Lower reference",
        )

        ax.axhline(
            self.voltage_max,
            linestyle="--",
            label="Upper reference",
        )

        ax.set_title(
            "Selected Bus Voltage Profiles"
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "Voltage (p.u.)"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.autofmt_xdate()

        fig.tight_layout()

        path = (
            self.output_directory
            / "selected_bus_voltage_profiles.png"
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

                                                                        
                       
                                                                        

    def plot_voltage_variation(
        self,
        results: list[VoltageAnalysisResult],
        show: bool = False,
    ) -> Path:
        """
        Plot voltage range for every bus.
        """

        dataframe = (
            self.voltage_results_dataframe(
                results
            )
        )

        dataframe = dataframe.sort_values(
            "voltage_range_pu",
            ascending=False,
        )

        fig, ax = plt.subplots(
            figsize=(11, 5)
        )

        ax.bar(
            dataframe["bus_id"].astype(str),
            dataframe["voltage_range_pu"],
        )

        ax.set_title(
            "Voltage Variation by Bus"
        )

        ax.set_xlabel(
            "Bus ID"
        )

        ax.set_ylabel(
            "Voltage Range (p.u.)"
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        path = (
            self.output_directory
            / "voltage_variation_by_bus.png"
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

                                                                        
                          
                                                                        

    def plot_system_loading(
        self,
        dataframe: pd.DataFrame,
        show: bool = False,
    ) -> Path:
        """
        Plot system-wide line loading envelope.
        """

        self._validate_columns(
            dataframe,
            [
                "timestamp",
                "loading_percent",
            ],
        )

        grouped = (
            dataframe
            .groupby("timestamp")[
                "loading_percent"
            ]
            .agg(
                [
                    "min",
                    "mean",
                    "max",
                ]
            )
            .reset_index()
        )

        fig, ax = plt.subplots(
            figsize=(11, 5)
        )

        ax.plot(
            grouped["timestamp"],
            grouped["mean"],
            label="Mean line loading",
        )

        ax.fill_between(
            grouped["timestamp"],
            grouped["min"],
            grouped["max"],
            alpha=0.20,
            label="Line loading range",
        )

        ax.axhline(
            self.loading_limit,
            linestyle="--",
            label="Reference thermal limit",
        )

        ax.set_title(
            "System-Wide Line Loading"
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "Loading (%)"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.autofmt_xdate()

        fig.tight_layout()

        path = (
            self.output_directory
            / "system_line_loading.png"
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

                                                                        
                            
                                                                        

    def plot_line_loading_profiles(
        self,
        dataframe: pd.DataFrame,
        line_ids: list[int] | None = None,
        show: bool = False,
    ) -> Path:
        """
        Plot loading profiles for selected lines.
        """

        self._validate_columns(
            dataframe,
            [
                "timestamp",
                "line_id",
                "loading_percent",
            ],
        )

        if line_ids is None:

            variation = (
                dataframe
                .groupby("line_id")[
                    "loading_percent"
                ]
                .agg(
                    lambda x:
                    x.max() - x.min()
                )
                .sort_values(
                    ascending=False
                )
            )

            line_ids = (
                variation
                .head(5)
                .index
                .tolist()
            )

        fig, ax = plt.subplots(
            figsize=(11, 5)
        )

        for line_id in line_ids:

            line_data = (
                dataframe[
                    dataframe["line_id"]
                    == line_id
                ]
                .sort_values("timestamp")
            )

            if line_data.empty:
                continue

            ax.plot(
                line_data["timestamp"],
                line_data["loading_percent"],
                label=f"Line {line_id}",
            )

        ax.axhline(
            self.loading_limit,
            linestyle="--",
            label="Reference thermal limit",
        )

        ax.set_title(
            "Selected Line Loading Profiles"
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "Loading (%)"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.autofmt_xdate()

        fig.tight_layout()

        path = (
            self.output_directory
            / "selected_line_loading_profiles.png"
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

                                                                        
                       
                                                                        

    def plot_loading_variation(
        self,
        results: list[LineAnalysisResult],
        show: bool = False,
    ) -> Path:
        """
        Plot maximum loading for every line.
        """

        dataframe = (
            self.line_results_dataframe(
                results
            )
        )

        dataframe = dataframe.sort_values(
            "maximum_loading_percent",
            ascending=False,
        )

        fig, ax = plt.subplots(
            figsize=(11, 5)
        )

        ax.bar(
            dataframe["line_id"].astype(str),
            dataframe[
                "maximum_loading_percent"
            ],
        )

        ax.axhline(
            self.loading_limit,
            linestyle="--",
            label="Reference thermal limit",
        )

        ax.set_title(
            "Maximum Line Loading"
        )

        ax.set_xlabel(
            "Line ID"
        )

        ax.set_ylabel(
            "Maximum Loading (%)"
        )

        ax.legend()

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        path = (
            self.output_directory
            / "maximum_line_loading.png"
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

                                                                        
                      
                                                                        

    def calculate_limit_statistics(
        self,
        bus_data: pd.DataFrame,
        line_data: pd.DataFrame,
    ) -> dict[str, float | int]:
        """
        Calculate simple voltage/loading limit statistics.

        These are exploratory statistics only.
        """

        voltage = bus_data[
            "voltage_pu"
        ].astype(float)

        loading = line_data[
            "loading_percent"
        ].astype(float)

        undervoltage = (
            voltage < self.voltage_min
        )

        overvoltage = (
            voltage > self.voltage_max
        )

        overloaded = (
            loading > self.loading_limit
        )

        return {
            "undervoltage_count": int(
                undervoltage.sum()
            ),

            "overvoltage_count": int(
                overvoltage.sum()
            ),

            "thermal_overload_count": int(
                overloaded.sum()
            ),

            "undervoltage_fraction": float(
                undervoltage.mean()
            ),

            "overvoltage_fraction": float(
                overvoltage.mean()
            ),

            "thermal_overload_fraction": float(
                overloaded.mean()
            ),
        }

                                                                        
                     
                                                                        

    def save_statistics(
        self,
        voltage_results: list[VoltageAnalysisResult],
        line_results: list[LineAnalysisResult],
    ) -> tuple[Path, Path]:

        voltage_dataframe = (
            self.voltage_results_dataframe(
                voltage_results
            )
        )

        line_dataframe = (
            self.line_results_dataframe(
                line_results
            )
        )

        voltage_path = (
            self.output_directory
            / "per_bus_voltage_statistics.csv"
        )

        line_path = (
            self.output_directory
            / "per_line_loading_statistics.csv"
        )

        voltage_dataframe.to_csv(
            voltage_path,
            index=False,
        )

        line_dataframe.to_csv(
            line_path,
            index=False,
        )

        return (
            voltage_path,
            line_path,
        )

                                                                        
                
                                                                        

    @staticmethod
    def _validate_columns(
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> None:

        if dataframe.empty:
            raise ValueError(
                "Cannot analyze an empty dataframe."
            )

        missing = (
            set(columns)
            - set(dataframe.columns)
        )

        if missing:
            raise ValueError(
                "Required columns missing: "
                f"{sorted(missing)}"
            )


                                                                        
      
                                                                        


if __name__ == "__main__":

    loader = EDADataLoader()

    dataset = loader.load_all()

    analyzer = VoltageCurrentAnalyzer()

                                                                    
             
                                                                    

    print(
        "\nRunning voltage analysis..."
    )

    voltage_results = (
        analyzer.analyze_voltage(
            dataset.bus
        )
    )

                                                                    
                          
                                                                    

    print(
        "Running line current/loading analysis..."
    )

    line_results = (
        analyzer.analyze_lines(
            dataset.line
        )
    )

                                                                    
                               
                                                                    

    (
        voltage_path,
        line_path,
    ) = analyzer.save_statistics(
        voltage_results,
        line_results,
    )

                                                                    
                    
                                                                    

    analyzer.plot_system_voltage(
        dataset.bus
    )

    analyzer.plot_bus_voltage_profiles(
        dataset.bus
    )

    analyzer.plot_voltage_variation(
        voltage_results
    )

    analyzer.plot_system_loading(
        dataset.line
    )

    analyzer.plot_line_loading_profiles(
        dataset.line
    )

    analyzer.plot_loading_variation(
        line_results
    )

                                                                    
                      
                                                                    

    limit_statistics = (
        analyzer.calculate_limit_statistics(
            dataset.bus,
            dataset.line,
        )
    )

    print()

    print(
        "=" * 70
    )

    print(
        "VOLTAGE / CURRENT ANALYSIS SUMMARY"
    )

    print(
        "=" * 70
    )

    for key, value in (
        limit_statistics.items()
    ):

        print(
            f"{key:35s}: {value}"
        )

    print(
        "=" * 70
    )

    print(
        f"\nStatistics saved to:"
        f"\n{voltage_path}"
        f"\n{line_path}"
    )