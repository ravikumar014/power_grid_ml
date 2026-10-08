"""
Distribution analysis for power-grid measurements.

This module analyzes the statistical distributions of electrical
measurements before feature engineering or anomaly detection.

Analyzed variables
------------------
Bus:
    - voltage_pu
    - angle_deg
    - active_power_mw
    - reactive_power_mvar

Line:
    - current_from_ka
    - current_to_ka
    - loading_percent

Outputs
-------
1. Descriptive statistics
2. Skewness
3. Kurtosis
4. Quantiles
5. Outlier statistics
6. Histograms
7. Box plots

The analysis operates on physical-unit data, not normalized data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda.load_data import EDADataLoader


@dataclass
class DistributionResult:
    """
    Statistical summary for one measurement variable.
    """

    feature: str

    count: int
    mean: float
    median: float
    std: float

    minimum: float
    q1: float
    q3: float
    maximum: float

    skewness: float
    kurtosis: float

    outlier_count: int
    outlier_fraction: float


class DistributionAnalyzer:
    """
    Analyze distributions of power-grid measurements.
    """

    BUS_FEATURES = [
        "voltage_pu",
        "angle_deg",
        "active_power_mw",
        "reactive_power_mvar",
    ]

    LINE_FEATURES = [
        "current_from_ka",
        "current_to_ka",
        "loading_percent",
    ]

    def __init__(
        self,
        output_directory: str | Path = (
            "outputs/eda/distributions"
        ),
    ) -> None:

        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

                                                                        
                
                                                                        

    def analyze(
        self,
        dataframe: pd.DataFrame,
        features: list[str],
    ) -> list[DistributionResult]:
        """
        Calculate distribution statistics.

        Parameters
        ----------
        dataframe:
            Input physical-unit measurement dataframe.

        features:
            Columns to analyze.

        Returns
        -------
        list[DistributionResult]
        """

        self._validate_features(
            dataframe,
            features,
        )

        results = []

        for feature in features:

            values = (
                pd.to_numeric(
                    dataframe[feature],
                    errors="coerce",
                )
                .dropna()
            )

            if values.empty:
                continue

                                                                        
                                         
             
                          
             
                            
             
                          
             
                            
                                                                        

            q1 = float(
                values.quantile(0.25)
            )

            q3 = float(
                values.quantile(0.75)
            )

            iqr = q3 - q1

            lower_bound = (
                q1 - 1.5 * iqr
            )

            upper_bound = (
                q3 + 1.5 * iqr
            )

            outliers = (
                (values < lower_bound)
                | (values > upper_bound)
            )

            outlier_count = int(
                outliers.sum()
            )

            result = DistributionResult(
                feature=feature,

                count=int(
                    values.count()
                ),

                mean=float(
                    values.mean()
                ),

                median=float(
                    values.median()
                ),

                std=float(
                    values.std(
                        ddof=1
                    )
                ),

                minimum=float(
                    values.min()
                ),

                q1=q1,

                q3=q3,

                maximum=float(
                    values.max()
                ),

                skewness=float(
                    values.skew()
                ),

                kurtosis=float(
                    values.kurt()
                ),

                outlier_count=outlier_count,

                outlier_fraction=(
                    outlier_count
                    / len(values)
                ),
            )

            results.append(
                result
            )

        return results

                                                                        
                                  
                                                                        

    @staticmethod
    def results_to_dataframe(
        results: list[DistributionResult],
    ) -> pd.DataFrame:
        """
        Convert statistical results into a dataframe.
        """

        return pd.DataFrame(
            [
                {
                    "feature": result.feature,
                    "count": result.count,
                    "mean": result.mean,
                    "median": result.median,
                    "std": result.std,
                    "minimum": result.minimum,
                    "q1": result.q1,
                    "q3": result.q3,
                    "maximum": result.maximum,
                    "skewness": result.skewness,
                    "kurtosis": result.kurtosis,
                    "outlier_count": result.outlier_count,
                    "outlier_fraction": result.outlier_fraction,
                }
                for result in results
            ]
        )

                                                                        
               
                                                                        

    def plot_histogram(
        self,
        dataframe: pd.DataFrame,
        feature: str,
        bins: int = 40,
        show: bool = False,
    ) -> Path:
        """
        Generate a histogram for one feature.
        """

        self._validate_features(
            dataframe,
            [feature],
        )

        values = pd.to_numeric(
            dataframe[feature],
            errors="coerce",
        ).dropna()

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.75,
        )

        ax.set_title(
            f"Distribution of {feature}"
        )

        ax.set_xlabel(
            feature
        )

        ax.set_ylabel(
            "Density"
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        path = (
            self.output_directory
            / f"{feature}_histogram.png"
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

                                                                        
              
                                                                        

    def plot_boxplot(
        self,
        dataframe: pd.DataFrame,
        feature: str,
        show: bool = False,
    ) -> Path:
        """
        Generate a box plot for one feature.
        """

        self._validate_features(
            dataframe,
            [feature],
        )

        values = pd.to_numeric(
            dataframe[feature],
            errors="coerce",
        ).dropna()

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.boxplot(
            values,
            vert=False,
        )

        ax.set_title(
            f"Box Plot of {feature}"
        )

        ax.set_xlabel(
            feature
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        path = (
            self.output_directory
            / f"{feature}_boxplot.png"
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

                                                                        
                    
                                                                        

    def plot_all_distributions(
        self,
        dataframe: pd.DataFrame,
        features: list[str],
    ) -> None:
        """
        Generate histogram and box plot for every feature.
        """

        for feature in features:

            print(
                f"Generating distribution plots: "
                f"{feature}"
            )

            self.plot_histogram(
                dataframe,
                feature,
            )

            self.plot_boxplot(
                dataframe,
                feature,
            )

                                                                        
                        
                                                                        

    def save_statistics(
        self,
        results: list[DistributionResult],
        filename: str = "distribution_statistics.csv",
    ) -> Path:
        """
        Save statistical analysis results.
        """

        dataframe = (
            self.results_to_dataframe(
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

                                                                        
                
                                                                        

    @staticmethod
    def _validate_features(
        dataframe: pd.DataFrame,
        features: list[str],
    ) -> None:

        if dataframe.empty:
            raise ValueError(
                "Cannot analyze an empty dataframe."
            )

        missing = (
            set(features)
            - set(dataframe.columns)
        )

        if missing:
            raise ValueError(
                "Requested features not found in "
                f"dataset: {sorted(missing)}"
            )

                                                                        
                    
                                                                        

    def print_report(
        self,
        results: list[DistributionResult],
    ) -> None:
        """
        Print statistical results in a readable format.
        """

        dataframe = (
            self.results_to_dataframe(
                results
            )
        )

        print()

        print(
            "=" * 100
        )

        print(
            "POWER GRID DISTRIBUTION ANALYSIS"
        )

        print(
            "=" * 100
        )

        print(
            dataframe.to_string(
                index=False,
                float_format=lambda x: f"{x:.5f}",
            )
        )

        print(
            "=" * 100
        )


                                                                        
      
                                                                        

if __name__ == "__main__":

    loader = EDADataLoader()

    dataset = loader.load_all()

    analyzer = DistributionAnalyzer()

                                                                    
                  
                                                                    

    print(
        "\nAnalyzing bus measurements..."
    )

    bus_results = analyzer.analyze(
        dataset.bus,
        analyzer.BUS_FEATURES,
    )

    analyzer.print_report(
        bus_results
    )

    analyzer.save_statistics(
        bus_results,
        filename="bus_distribution_statistics.csv",
    )

    analyzer.plot_all_distributions(
        dataset.bus,
        analyzer.BUS_FEATURES,
    )

                                                                    
                   
                                                                    

    print(
        "\nAnalyzing line measurements..."
    )

    line_results = analyzer.analyze(
        dataset.line,
        analyzer.LINE_FEATURES,
    )

    analyzer.print_report(
        line_results
    )

    analyzer.save_statistics(
        line_results,
        filename="line_distribution_statistics.csv",
    )

    analyzer.plot_all_distributions(
        dataset.line,
        analyzer.LINE_FEATURES,
    )

    print(
        "\nDistribution analysis completed."
    )

    print(
        f"\nResults saved to:"
        f"\n{analyzer.output_directory}"
    )