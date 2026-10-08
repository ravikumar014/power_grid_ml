"""
Correlation analysis for power-grid measurements.

This module investigates statistical relationships between electrical
measurements.

Primary relationships
---------------------
Bus:
    voltage_pu
    angle_deg
    active_power_mw
    reactive_power_mvar

Line:
    current_from_ka
    current_to_ka
    loading_percent

Correlation methods
-------------------
Pearson:
    Measures linear correlation.

Spearman:
    Measures monotonic correlation and is more robust to non-Gaussian
    relationships and outliers.

Outputs
-------
- Pearson correlation matrix
- Spearman correlation matrix
- Heatmaps
- Strongest feature relationships
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda.load_data import EDADataLoader


@dataclass
class CorrelationPair:
    """
    Represents one pair of correlated features.
    """

    feature_1: str
    feature_2: str
    correlation: float
    absolute_correlation: float


class CorrelationAnalyzer:
    """
    Analyze relationships between power-grid measurements.
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
            "outputs/eda/correlation"
        ),
    ) -> None:

        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

                                                                        
                        
                                                                        

    def calculate(
        self,
        dataframe: pd.DataFrame,
        features: list[str],
        method: str = "pearson",
    ) -> pd.DataFrame:
        """
        Calculate a correlation matrix.

        Parameters
        ----------
        dataframe:
            Input dataframe.

        features:
            Numerical features to analyze.

        method:
            "pearson" or "spearman".
        """

        if method not in {
            "pearson",
            "spearman",
        }:

            raise ValueError(
                "method must be 'pearson' or 'spearman'."
            )

        self._validate_features(
            dataframe,
            features,
        )

        data = dataframe[
            features
        ].copy()

        for feature in features:

            data[feature] = pd.to_numeric(
                data[feature],
                errors="coerce",
            )

        return data.corr(
            method=method
        )

                                                                        
                        
                                                                        

    def calculate_both(
        self,
        dataframe: pd.DataFrame,
        features: list[str],
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Calculate both Pearson and Spearman matrices.
        """

        pearson = self.calculate(
            dataframe,
            features,
            method="pearson",
        )

        spearman = self.calculate(
            dataframe,
            features,
            method="spearman",
        )

        return pearson, spearman

                                                                        
             
                                                                        

    def plot_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str,
        filename: str,
        show: bool = False,
    ) -> Path:
        """
        Plot a correlation heatmap.

        No fixed colour palette is specified so matplotlib's default
        rendering is used.
        """

        size = max(
            7,
            len(correlation_matrix.columns) * 1.2,
        )

        fig, ax = plt.subplots(
            figsize=(size, size),
        )

        image = ax.imshow(
            correlation_matrix.values,
            vmin=-1,
            vmax=1,
            aspect="auto",
        )

        ax.set_xticks(
            np.arange(
                len(
                    correlation_matrix.columns
                )
            )
        )

        ax.set_yticks(
            np.arange(
                len(
                    correlation_matrix.index
                )
            )
        )

        ax.set_xticklabels(
            correlation_matrix.columns,
            rotation=45,
            ha="right",
        )

        ax.set_yticklabels(
            correlation_matrix.index,
        )

                                                                        
                                        
                                                                        

        for i in range(
            len(correlation_matrix.index)
        ):

            for j in range(
                len(correlation_matrix.columns)
            ):

                value = (
                    correlation_matrix.iloc[
                        i,
                        j
                    ]
                )

                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        fig.colorbar(
            image,
            ax=ax,
            label="Correlation",
        )

        ax.set_title(
            title
        )

        fig.tight_layout()

        path = (
            self.output_directory
            / filename
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

                                                                        
                             
                                                                        

    def strongest_pairs(
        self,
        correlation_matrix: pd.DataFrame,
        top_k: int = 10,
    ) -> list[CorrelationPair]:
        """
        Extract the strongest unique feature relationships.

        Self-correlations are excluded.
        """

        pairs = []

        features = list(
            correlation_matrix.columns
        )

        for i in range(
            len(features)
        ):

            for j in range(
                i + 1,
                len(features),
            ):

                feature_1 = features[i]
                feature_2 = features[j]

                correlation = float(
                    correlation_matrix.loc[
                        feature_1,
                        feature_2,
                    ]
                )

                pairs.append(
                    CorrelationPair(
                        feature_1=feature_1,
                        feature_2=feature_2,
                        correlation=correlation,
                        absolute_correlation=abs(
                            correlation
                        ),
                    )
                )

        pairs.sort(
            key=lambda pair:
            pair.absolute_correlation,
            reverse=True,
        )

        return pairs[:top_k]

                                                                        
                                
                                                                        

    @staticmethod
    def pairs_to_dataframe(
        pairs: list[CorrelationPair],
    ) -> pd.DataFrame:
        """
        Convert correlation pairs to a dataframe.
        """

        return pd.DataFrame(
            [
                {
                    "feature_1": pair.feature_1,
                    "feature_2": pair.feature_2,
                    "correlation": pair.correlation,
                    "absolute_correlation": (
                        pair.absolute_correlation
                    ),
                }
                for pair in pairs
            ]
        )

                                                                        
                 
                                                                        

    def save_matrix(
        self,
        matrix: pd.DataFrame,
        filename: str,
    ) -> Path:
        """
        Save a correlation matrix.
        """

        path = (
            self.output_directory
            / filename
        )

        matrix.to_csv(
            path
        )

        return path

                                                                        
                                  
                                                                        

    def save_pairs(
        self,
        pairs: list[CorrelationPair],
        filename: str,
    ) -> Path:
        """
        Save strongest correlations.
        """

        dataframe = (
            self.pairs_to_dataframe(
                pairs
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

                                                                        
                         
                                                                        

    def calculate_per_bus(
        self,
        dataframe: pd.DataFrame,
        features: list[str],
    ) -> dict[int, pd.DataFrame]:
        """
        Calculate a separate Pearson correlation matrix for each bus.

        This helps determine whether relationships are consistent
        throughout the network.
        """

        self._validate_features(
            dataframe,
            ["bus_id"] + features,
        )

        matrices = {}

        for bus_id, group in dataframe.groupby(
            "bus_id",
            sort=True,
        ):

            matrices[int(bus_id)] = (
                self.calculate(
                    group,
                    features,
                    method="pearson",
                )
            )

        return matrices

                                                                        
                                   
                                                                        

    def calculate_bus_voltage_correlation(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate correlation between voltage trajectories of buses.

        Rows:
            timestamps

        Columns:
            bus IDs

        Values:
            voltage magnitude

        This is different from ordinary feature correlation.

        It asks:

            "Do the voltage trajectories of two buses vary together?"
        """

        self._validate_features(
            dataframe,
            [
                "timestamp",
                "bus_id",
                "voltage_pu",
            ],
        )

        voltage_matrix = (
            dataframe
            .pivot_table(
                index="timestamp",
                columns="bus_id",
                values="voltage_pu",
                aggfunc="mean",
            )
            .sort_index()
        )

        return voltage_matrix.corr(
            method="pearson"
        )

                                                                        
                               
                                                                        

    def plot_bus_voltage_correlation(
        self,
        dataframe: pd.DataFrame,
        show: bool = False,
    ) -> Path:
        """
        Plot the bus-to-bus voltage correlation matrix.
        """

        matrix = (
            self.calculate_bus_voltage_correlation(
                dataframe
            )
        )

        return self.plot_heatmap(
            matrix,
            title=(
                "Bus-to-Bus Voltage Correlation"
            ),
            filename=(
                "bus_voltage_correlation.png"
            ),
            show=show,
        )

                                                                        
                
                                                                        

    @staticmethod
    def _validate_features(
        dataframe: pd.DataFrame,
        features: list[str],
    ) -> None:

        if dataframe.empty:

            raise ValueError(
                "Cannot calculate correlation "
                "on an empty dataframe."
            )

        missing = (
            set(features)
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

    analyzer = CorrelationAnalyzer()

                                                                        
                     
                                                                        

    print()
    print(
        "=" * 70
    )

    print(
        "BUS FEATURE CORRELATION"
    )

    print(
        "=" * 70
    )

    bus_pearson, bus_spearman = (
        analyzer.calculate_both(
            dataset.bus,
            analyzer.DEFAULT_BUS_FEATURES,
        )
    )

    print(
        "\nPearson correlation:"
    )

    print(
        bus_pearson.to_string(
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    print(
        "\nSpearman correlation:"
    )

    print(
        bus_spearman.to_string(
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

                                                                        
                   
                                                                        

    analyzer.save_matrix(
        bus_pearson,
        "bus_pearson_correlation.csv",
    )

    analyzer.save_matrix(
        bus_spearman,
        "bus_spearman_correlation.csv",
    )

                                                                        
              
                                                                        

    analyzer.plot_heatmap(
        bus_pearson,
        title="Bus Feature Pearson Correlation",
        filename="bus_pearson_heatmap.png",
    )

    analyzer.plot_heatmap(
        bus_spearman,
        title="Bus Feature Spearman Correlation",
        filename="bus_spearman_heatmap.png",
    )

                                                                        
                             
                                                                        

    strongest_bus = (
        analyzer.strongest_pairs(
            bus_pearson,
            top_k=10,
        )
    )

    print(
        "\nStrongest Pearson relationships:"
    )

    print(
        analyzer
        .pairs_to_dataframe(
            strongest_bus
        )
        .to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    analyzer.save_pairs(
        strongest_bus,
        "strongest_bus_correlations.csv",
    )

                                                                        
                      
                                                                        

    print()
    print(
        "=" * 70
    )

    print(
        "LINE FEATURE CORRELATION"
    )

    print(
        "=" * 70
    )

    line_pearson, line_spearman = (
        analyzer.calculate_both(
            dataset.line,
            analyzer.DEFAULT_LINE_FEATURES,
        )
    )

    print(
        "\nPearson correlation:"
    )

    print(
        line_pearson.to_string(
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    print(
        "\nSpearman correlation:"
    )

    print(
        line_spearman.to_string(
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    analyzer.save_matrix(
        line_pearson,
        "line_pearson_correlation.csv",
    )

    analyzer.save_matrix(
        line_spearman,
        "line_spearman_correlation.csv",
    )

    analyzer.plot_heatmap(
        line_pearson,
        title="Line Feature Pearson Correlation",
        filename="line_pearson_heatmap.png",
    )

    analyzer.plot_heatmap(
        line_spearman,
        title="Line Feature Spearman Correlation",
        filename="line_spearman_heatmap.png",
    )

    strongest_line = (
        analyzer.strongest_pairs(
            line_pearson,
            top_k=10,
        )
    )

    print(
        "\nStrongest line relationships:"
    )

    print(
        analyzer
        .pairs_to_dataframe(
            strongest_line
        )
        .to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    analyzer.save_pairs(
        strongest_line,
        "strongest_line_correlations.csv",
    )

                                                                        
                                    
                                                                        

    print()
    print(
        "=" * 70
    )

    print(
        "BUS-TO-BUS VOLTAGE CORRELATION"
    )

    print(
        "=" * 70
    )

    bus_voltage_matrix = (
        analyzer.calculate_bus_voltage_correlation(
            dataset.bus
        )
    )

    analyzer.save_matrix(
        bus_voltage_matrix,
        "bus_voltage_correlation.csv",
    )

    analyzer.plot_bus_voltage_correlation(
        dataset.bus
    )

    print(
        "\nBus-to-bus voltage correlation:"
    )

    print(
        bus_voltage_matrix.to_string(
            float_format=lambda x:
            f"{x:.3f}"
        )
    )

    print(
        "\nCorrelation analysis completed."
    )

    print(
        f"\nResults saved to:"
        f"\n{analyzer.output_directory}"
    )