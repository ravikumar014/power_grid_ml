"""
Master EDA pipeline for the power-grid project.

This module orchestrates all Milestone-2 exploratory analyses:

1. Dataset loading
2. Distribution analysis
3. Voltage/current analysis
4. Correlation analysis
5. Temporal analysis
6. Frequency-domain analysis
7. PCA
8. t-SNE
9. Consolidated EDA report

Usage
-----
From project root:

    python -m src.eda.run_eda

The script does not modify the underlying dataset.
All outputs are written to:

    outputs/eda/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.eda.load_data import EDADataLoader

from src.eda.distributions import (
    DistributionAnalyzer,
)

from src.eda.voltage_current import (
    VoltageCurrentAnalyzer,
)

from src.eda.correlation import (
    CorrelationAnalyzer,
)

from src.eda.temporal_analysis import (
    TemporalAnalyzer,
)

from src.eda.frequency_analysis import (
    FrequencyAnalyzer,
)

from src.eda.dimensionality import (
    DimensionalityAnalyzer,
)


                                                                        
               
                                                                        


@dataclass
class EDAConfig:
    """
    Configuration for the complete EDA pipeline.
    """

    processed_root: Path = Path(
        "data/processed"
    )

    simulated_root: Path = Path(
        "data/simulated"
    )

    output_root: Path = Path(
        "outputs/eda"
    )

    random_state: int = 42

    rolling_window: int = 4

    tsne_perplexity: float = 30.0

    tsne_max_samples: int = 3000

    autocorrelation_max_lag: int = 24


                                                                        
        
                                                                        


@dataclass
class EDAStageResult:
    """
    Record execution information for one EDA stage.
    """

    name: str

    duration_seconds: float

    status: str


class EDARunner:
    """
    Execute the complete EDA pipeline.
    """

    def __init__(
        self,
        config: EDAConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else EDAConfig()
        )

        self.config.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.stage_results: list[
            EDAStageResult
        ] = []

                                                                        
             
                                                                        

    def _stage_header(
        self,
        number: int,
        title: str,
    ) -> None:

        print()
        print(
            "=" * 76
        )

        print(
            f"STAGE {number}: {title}"
        )

        print(
            "=" * 76
        )

    def _run_stage(
        self,
        name: str,
        function,
    ) -> None:
        """
        Execute one stage and record its runtime.
        """

        start = perf_counter()

        try:

            function()

            duration = (
                perf_counter()
                - start
            )

            self.stage_results.append(
                EDAStageResult(
                    name=name,
                    duration_seconds=duration,
                    status="SUCCESS",
                )
            )

        except Exception:

            duration = (
                perf_counter()
                - start
            )

            self.stage_results.append(
                EDAStageResult(
                    name=name,
                    duration_seconds=duration,
                    status="FAILED",
                )
            )

            raise

                                                                        
                         
                                                                        

    def load_data(self) -> None:

        self._stage_header(
            1,
            "LOADING DATA",
        )

        loader = EDADataLoader(
            processed_root=(
                self.config.processed_root
            ),
            simulated_root=(
                self.config.simulated_root
            ),
        )

        dataset = loader.load_all()

        self.bus_data = dataset.bus
        self.line_data = dataset.line

        print(
            f"Bus rows : "
            f"{len(self.bus_data)}"
        )

        print(
            f"Line rows: "
            f"{len(self.line_data)}"
        )

        print(
            f"Bus features: "
            f"{len(self.bus_data.columns)}"
        )

        print(
            f"Line features: "
            f"{len(self.line_data.columns)}"
        )

                                                                        
                                     
                                                                        

    def distribution_analysis(self) -> None:

        self._stage_header(
            2,
            "DISTRIBUTION ANALYSIS",
        )

        output_directory = (
            self.config.output_root
            / "distributions"
        )

        analyzer = DistributionAnalyzer(
            output_directory=output_directory
        )

                                                                        
             
                                                                        

        bus_results = analyzer.analyze(
            self.bus_data,
            analyzer.BUS_FEATURES,
        )

        analyzer.save_statistics(
            bus_results,
            filename=(
                "bus_distribution_statistics.csv"
            ),
        )

        analyzer.plot_all_distributions(
            self.bus_data,
            analyzer.BUS_FEATURES,
        )

                                                                        
              
                                                                        

        line_results = analyzer.analyze(
            self.line_data,
            analyzer.LINE_FEATURES,
        )

        analyzer.save_statistics(
            line_results,
            filename=(
                "line_distribution_statistics.csv"
            ),
        )

        analyzer.plot_all_distributions(
            self.line_data,
            analyzer.LINE_FEATURES,
        )

        print(
            "Distribution analysis completed."
        )

                                                                        
                               
                                                                        

    def voltage_current_analysis(self) -> None:

        self._stage_header(
            3,
            "VOLTAGE / CURRENT ANALYSIS",
        )

        output_directory = (
            self.config.output_root
            / "voltage_current"
        )

        analyzer = VoltageCurrentAnalyzer(
            output_directory=output_directory
        )

        voltage_results = (
            analyzer.analyze_voltage(
                self.bus_data
            )
        )

        line_results = (
            analyzer.analyze_lines(
                self.line_data
            )
        )

        analyzer.save_statistics(
            voltage_results,
            line_results,
        )

        analyzer.plot_system_voltage(
            self.bus_data
        )

        analyzer.plot_bus_voltage_profiles(
            self.bus_data
        )

        analyzer.plot_voltage_variation(
            voltage_results
        )

        analyzer.plot_system_loading(
            self.line_data
        )

        analyzer.plot_line_loading_profiles(
            self.line_data
        )

        analyzer.plot_loading_variation(
            line_results
        )

        limit_statistics = (
            analyzer.calculate_limit_statistics(
                self.bus_data,
                self.line_data,
            )
        )

        limit_path = (
            output_directory
            / "limit_statistics.csv"
        )

        pd.DataFrame(
            [limit_statistics]
        ).to_csv(
            limit_path,
            index=False,
        )

        print(
            "Voltage/current analysis completed."
        )

                                                                        
                           
                                                                        

    def correlation_analysis(self) -> None:

        self._stage_header(
            4,
            "CORRELATION ANALYSIS",
        )

        output_directory = (
            self.config.output_root
            / "correlation"
        )

        analyzer = CorrelationAnalyzer(
            output_directory=output_directory
        )

                                                                        
                                 
                                                                        

        (
            bus_pearson,
            bus_spearman,
        ) = analyzer.calculate_both(
            self.bus_data,
            analyzer.DEFAULT_BUS_FEATURES,
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
            title=(
                "Bus Feature Pearson Correlation"
            ),
            filename=(
                "bus_pearson_heatmap.png"
            ),
        )

        analyzer.plot_heatmap(
            bus_spearman,
            title=(
                "Bus Feature Spearman Correlation"
            ),
            filename=(
                "bus_spearman_heatmap.png"
            ),
        )

        strongest_bus = (
            analyzer.strongest_pairs(
                bus_pearson,
                top_k=10,
            )
        )

        analyzer.save_pairs(
            strongest_bus,
            "strongest_bus_correlations.csv",
        )

                                                                        
                                  
                                                                        

        (
            line_pearson,
            line_spearman,
        ) = analyzer.calculate_both(
            self.line_data,
            analyzer.DEFAULT_LINE_FEATURES,
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
            title=(
                "Line Feature Pearson Correlation"
            ),
            filename=(
                "line_pearson_heatmap.png"
            ),
        )

        analyzer.plot_heatmap(
            line_spearman,
            title=(
                "Line Feature Spearman Correlation"
            ),
            filename=(
                "line_spearman_heatmap.png"
            ),
        )

        strongest_line = (
            analyzer.strongest_pairs(
                line_pearson,
                top_k=10,
            )
        )

        analyzer.save_pairs(
            strongest_line,
            "strongest_line_correlations.csv",
        )

                                                                        
                                        
                                                                        

        bus_voltage_correlation = (
            analyzer.calculate_bus_voltage_correlation(
                self.bus_data
            )
        )

        analyzer.save_matrix(
            bus_voltage_correlation,
            "bus_voltage_correlation.csv",
        )

        analyzer.plot_bus_voltage_correlation(
            self.bus_data
        )

        print(
            "Correlation analysis completed."
        )

                                                                        
                                 
                                                                        

    def temporal_analysis(self) -> None:

        self._stage_header(
            5,
            "TEMPORAL / SEASONAL ANALYSIS",
        )

                                                                        
                                                                    
                                                         
                                                            
                                                                        

        bus_output = (
            self.config.output_root
            / "temporal"
            / "bus"
        )

        line_output = (
            self.config.output_root
            / "temporal"
            / "line"
        )

        bus_analyzer = TemporalAnalyzer(
            output_directory=bus_output,
            rolling_window=(
                self.config.rolling_window
            ),
        )

        line_analyzer = TemporalAnalyzer(
            output_directory=line_output,
            rolling_window=(
                self.config.rolling_window
            ),
        )

                                                                        
             
                                                                        

        bus_statistics = (
            bus_analyzer.calculate_statistics(
                self.bus_data,
                bus_analyzer.DEFAULT_BUS_FEATURES,
            )
        )

        bus_analyzer.save_statistics(
            bus_statistics,
            filename=(
                "bus_temporal_statistics.csv"
            ),
        )

        for feature in (
            bus_analyzer.DEFAULT_BUS_FEATURES
        ):

            bus_analyzer.plot_time_series(
                self.bus_data,
                feature,
            )

            bus_analyzer.plot_hourly_profile(
                self.bus_data,
                feature,
            )

            bus_analyzer.plot_ramp_rate(
                self.bus_data,
                feature,
            )

            bus_analyzer.plot_rolling_statistics(
                self.bus_data,
                feature,
            )

            bus_analyzer.plot_autocorrelation(
                self.bus_data,
                feature,
                max_lag=(
                    self.config.autocorrelation_max_lag
                ),
            )

                                                                        
              
                                                                        

        line_statistics = (
            line_analyzer.calculate_statistics(
                self.line_data,
                line_analyzer.DEFAULT_LINE_FEATURES,
            )
        )

        line_analyzer.save_statistics(
            line_statistics,
            filename=(
                "line_temporal_statistics.csv"
            ),
        )

        for feature in (
            line_analyzer.DEFAULT_LINE_FEATURES
        ):

            line_analyzer.plot_time_series(
                self.line_data,
                feature,
            )

            line_analyzer.plot_hourly_profile(
                self.line_data,
                feature,
            )

            line_analyzer.plot_ramp_rate(
                self.line_data,
                feature,
            )

            line_analyzer.plot_rolling_statistics(
                self.line_data,
                feature,
            )

            line_analyzer.plot_autocorrelation(
                self.line_data,
                feature,
                max_lag=(
                    self.config.autocorrelation_max_lag
                ),
            )

        print(
            "Temporal analysis completed."
        )

                                                                        
                   
                                                                        

    def frequency_analysis(self) -> None:

        self._stage_header(
            6,
            "FREQUENCY-DOMAIN ANALYSIS",
        )

        bus_output = (
            self.config.output_root
            / "frequency"
            / "bus"
        )

        line_output = (
            self.config.output_root
            / "frequency"
            / "line"
        )

        bus_analyzer = FrequencyAnalyzer(
            output_directory=bus_output
        )

        line_analyzer = FrequencyAnalyzer(
            output_directory=line_output
        )

        bus_results = []

        for feature in (
            bus_analyzer.DEFAULT_BUS_FEATURES
        ):

            result = (
                bus_analyzer.analyze(
                    self.bus_data,
                    feature,
                )
            )

            bus_results.append(
                result
            )

            bus_analyzer.plot_spectrum(
                self.bus_data,
                feature,
            )

            bus_analyzer.plot_spectrum_cycles_per_hour(
                self.bus_data,
                feature,
            )

            bus_analyzer.plot_period_spectrum(
                self.bus_data,
                feature,
            )

        bus_analyzer.save_results(
            bus_results,
            filename=(
                "bus_frequency_statistics.csv"
            ),
        )

        line_results = []

        for feature in (
            line_analyzer.DEFAULT_LINE_FEATURES
        ):

            result = (
                line_analyzer.analyze(
                    self.line_data,
                    feature,
                )
            )

            line_results.append(
                result
            )

            line_analyzer.plot_spectrum(
                self.line_data,
                feature,
            )

            line_analyzer.plot_spectrum_cycles_per_hour(
                self.line_data,
                feature,
            )

            line_analyzer.plot_period_spectrum(
                self.line_data,
                feature,
            )

        line_analyzer.save_results(
            line_results,
            filename=(
                "line_frequency_statistics.csv"
            ),
        )

        print(
            "Frequency analysis completed."
        )

                                                                        
                           
                                                                        

    def dimensionality_analysis(self) -> None:

        self._stage_header(
            7,
            "PCA / t-SNE",
        )

        bus_output = (
            self.config.output_root
            / "dimensionality"
            / "bus"
        )

        line_output = (
            self.config.output_root
            / "dimensionality"
            / "line"
        )

        bus_analyzer = DimensionalityAnalyzer(
            output_directory=bus_output,
            random_state=(
                self.config.random_state
            ),
        )

        line_analyzer = DimensionalityAnalyzer(
            output_directory=line_output,
            random_state=(
                self.config.random_state
            ),
        )

                                                                        
                 
                                                                        

        (
            bus_pca,
            bus_coordinates,
            bus_loadings,
        ) = bus_analyzer.run_pca(
            self.bus_data,
            bus_analyzer.BUS_FEATURES,
        )

        bus_analyzer.plot_pca_variance(
            bus_pca,
            filename=(
                "bus_pca_explained_variance.png"
            ),
        )

        bus_analyzer.plot_pca_2d(
            bus_coordinates,
            filename=(
                "bus_pca_2d.png"
            ),
        )

        bus_analyzer.save_loadings(
            bus_loadings,
            filename=(
                "bus_pca_loadings.csv"
            ),
        )

                                                                        
                   
                                                                        

        (
            bus_tsne,
            _,
        ) = bus_analyzer.run_tsne(
            self.bus_data,
            bus_analyzer.BUS_FEATURES,
            perplexity=(
                self.config.tsne_perplexity
            ),
            max_samples=(
                self.config.tsne_max_samples
            ),
        )

        bus_analyzer.plot_tsne(
            bus_tsne,
            filename=(
                "bus_tsne_projection.png"
            ),
        )

        bus_analyzer.save_tsne_coordinates(
            bus_tsne,
            filename=(
                "bus_tsne_coordinates.csv"
            ),
        )

                                                                        
                  
                                                                        

        (
            line_pca,
            line_coordinates,
            line_loadings,
        ) = line_analyzer.run_pca(
            self.line_data,
            line_analyzer.LINE_FEATURES,
        )

        line_analyzer.plot_pca_variance(
            line_pca,
            filename=(
                "line_pca_explained_variance.png"
            ),
        )

        line_analyzer.plot_pca_2d(
            line_coordinates,
            filename=(
                "line_pca_2d.png"
            ),
        )

        line_analyzer.save_loadings(
            line_loadings,
            filename=(
                "line_pca_loadings.csv"
            ),
        )

                                                                        
                    
                                                                        

        (
            line_tsne,
            _,
        ) = line_analyzer.run_tsne(
            self.line_data,
            line_analyzer.LINE_FEATURES,
            perplexity=(
                self.config.tsne_perplexity
            ),
            max_samples=(
                self.config.tsne_max_samples
            ),
        )

        line_analyzer.plot_tsne(
            line_tsne,
            filename=(
                "line_tsne_projection.png"
            ),
        )

        line_analyzer.save_tsne_coordinates(
            line_tsne,
            filename=(
                "line_tsne_coordinates.csv"
            ),
        )

        print(
            "PCA/t-SNE analysis completed."
        )

                                                                        
                            
                                                                        

    def generate_report(self) -> None:

        self._stage_header(
            8,
            "GENERATING EDA REPORT",
        )

        report_path = (
            self.config.output_root
            / "eda_execution_report.csv"
        )

        dataframe = pd.DataFrame(
            [
                {
                    "stage": result.name,
                    "status": result.status,
                    "duration_seconds": (
                        result.duration_seconds
                    ),
                }
                for result in self.stage_results
            ]
        )

        dataframe.to_csv(
            report_path,
            index=False,
        )

        print(
            dataframe.to_string(
                index=False
            )
        )

        print(
            f"\nExecution report:"
            f"\n{report_path}"
        )

                                                                        
                   
                                                                        

    def run(self) -> None:

        print()
        print(
            "#" * 76
        )
        print(
            "# POWER GRID — MILESTONE 2 EDA"
        )
        print(
            "#" * 76
        )

        total_start = perf_counter()

                                                                        
                                                                     
                                                                        

        self._run_stage(
            "Data loading",
            self.load_data,
        )

        self._run_stage(
            "Distribution analysis",
            self.distribution_analysis,
        )

        self._run_stage(
            "Voltage/current analysis",
            self.voltage_current_analysis,
        )

        self._run_stage(
            "Correlation analysis",
            self.correlation_analysis,
        )

        self._run_stage(
            "Temporal analysis",
            self.temporal_analysis,
        )

        self._run_stage(
            "Frequency analysis",
            self.frequency_analysis,
        )

        self._run_stage(
            "PCA/t-SNE",
            self.dimensionality_analysis,
        )

        total_duration = (
            perf_counter()
            - total_start
        )

                                                                        
                       
                                                                        

        self.generate_report()

        print()
        print(
            "#" * 76
        )

        print(
            "# MILESTONE 2 COMPLETE"
        )

        print(
            "#" * 76
        )

        print(
            f"\nTotal EDA runtime: "
            f"{total_duration:.2f} seconds"
        )

        print(
            f"\nResults directory:"
            f"\n{self.config.output_root}"
        )

        print(
            "\nAll EDA stages completed successfully."
        )


                                                                        
      
                                                                        


def main() -> None:

    config = EDAConfig()

    runner = EDARunner(
        config=config
    )

    runner.run()


if __name__ == "__main__":

    main()