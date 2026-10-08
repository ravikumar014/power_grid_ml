"""
Dimensionality-reduction analysis for power-grid measurements.

Methods
-------
PCA:
    Quantifies global variance structure and feature redundancy.

t-SNE:
    Provides a nonlinear 2-D visualization of local sample structure.

Important
---------
PCA is an analytical transformation.

t-SNE is used here ONLY for visualization. It should not be treated
as a production feature transformation or anomaly detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.eda.load_data import EDADataLoader


@dataclass
class PCAResult:
    """Summary of PCA variance structure."""

    n_components: int
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray


class DimensionalityAnalyzer:
    """
    Perform PCA and t-SNE analysis on electrical measurements.
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
            "outputs/eda/dimensionality"
        ),
        random_state: int = 42,
    ) -> None:

        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.random_state = random_state

                                                                        
                      
                                                                        

    @staticmethod
    def prepare_features(
        dataframe: pd.DataFrame,
        features: list[str],
    ) -> pd.DataFrame:
        """
        Extract numerical features and remove invalid rows.
        """

        missing = (
            set(features)
            - set(dataframe.columns)
        )

        if missing:
            raise ValueError(
                f"Missing features: {sorted(missing)}"
            )

        data = dataframe[
            features
        ].copy()

        for feature in features:

            data[feature] = pd.to_numeric(
                data[feature],
                errors="coerce",
            )

        data = data.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        data = data.dropna()

        if data.empty:
            raise ValueError(
                "No valid samples remain after cleaning."
            )

        return data

                                                                        
                     
                                                                        

    @staticmethod
    def standardize(
        data: pd.DataFrame,
    ) -> tuple[
        np.ndarray,
        StandardScaler,
    ]:
        """
        Standardize features before PCA/t-SNE.

        z = (x - mean) / std
        """

        scaler = StandardScaler()

        X = scaler.fit_transform(
            data
        )

        return X, scaler

                                                                        
         
                                                                        

    def run_pca(
        self,
        dataframe: pd.DataFrame,
        features: list[str],
    ) -> tuple[
        PCA,
        np.ndarray,
        pd.DataFrame,
    ]:
        """
        Run PCA using all available components.

        Returns
        -------
        pca:
            Fitted PCA object.

        transformed:
            PCA coordinates.

        loadings:
            Feature loadings.
        """

        data = self.prepare_features(
            dataframe,
            features,
        )

        X, scaler = self.standardize(
            data
        )

        n_components = min(
            X.shape[0],
            X.shape[1],
        )

        pca = PCA(
            n_components=n_components
        )

        transformed = pca.fit_transform(
            X
        )

        loadings = pd.DataFrame(
            pca.components_.T,
            index=features,
            columns=[
                f"PC{i + 1}"
                for i in range(
                    n_components
                )
            ],
        )

        return (
            pca,
            transformed,
            loadings,
        )

                                                                        
                         
                                                                        

    def pca_variance_report(
        self,
        pca: PCA,
    ) -> PCAResult:

        explained = (
            pca.explained_variance_ratio_
        )

        cumulative = np.cumsum(
            explained
        )

        return PCAResult(
            n_components=len(
                explained
            ),
            explained_variance_ratio=(
                explained
            ),
            cumulative_explained_variance=(
                cumulative
            ),
        )

                                                                        
                       
                                                                        

    def plot_pca_variance(
        self,
        pca: PCA,
        filename: str = (
            "pca_explained_variance.png"
        ),
        show: bool = False,
    ) -> Path:
        """
        Plot explained and cumulative variance.
        """

        explained = (
            pca.explained_variance_ratio_
        )

        cumulative = np.cumsum(
            explained
        )

        components = np.arange(
            1,
            len(explained) + 1,
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.plot(
            components,
            explained,
            marker="o",
            label="Explained variance",
        )

        ax.plot(
            components,
            cumulative,
            marker="s",
            linestyle="--",
            label="Cumulative variance",
        )

        ax.set_xlabel(
            "Principal component"
        )

        ax.set_ylabel(
            "Variance ratio"
        )

        ax.set_title(
            "PCA Explained Variance"
        )

        ax.set_xticks(
            components
        )

        ax.set_ylim(
            0,
            1.05,
        )

        ax.legend()

        ax.grid(
            alpha=0.25
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

                                                                        
                        
                                                                        

    def plot_pca_2d(
        self,
        transformed: np.ndarray,
        filename: str = (
            "pca_2d_projection.png"
        ),
        show: bool = False,
    ) -> Path:
        """
        Plot the first two principal components.
        """

        if transformed.shape[1] < 2:
            raise ValueError(
                "At least two PCA components are required."
            )

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        ax.scatter(
            transformed[:, 0],
            transformed[:, 1],
            s=12,
            alpha=0.65,
        )

        ax.set_xlabel(
            "PC1"
        )

        ax.set_ylabel(
            "PC2"
        )

        ax.set_title(
            "PCA Projection"
        )

        ax.grid(
            alpha=0.25
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

                                                                        
                  
                                                                        

    def save_loadings(
        self,
        loadings: pd.DataFrame,
        filename: str = (
            "pca_loadings.csv"
        ),
    ) -> Path:
        """
        Save PCA feature loadings.
        """

        path = (
            self.output_directory
            / filename
        )

        loadings.to_csv(
            path
        )

        return path

                                                                        
           
                                                                        

    def run_tsne(
        self,
        dataframe: pd.DataFrame,
        features: list[str],
        perplexity: float = 30.0,
        max_samples: int = 3000,
    ) -> tuple[
        np.ndarray,
        pd.DataFrame,
    ]:
        """
        Run t-SNE for visualization.

        Parameters
        ----------
        perplexity:
            Effective local-neighborhood scale.

        max_samples:
            Maximum samples used to prevent unnecessary memory/runtime
            consumption on the MacBook during EDA.

        Returns
        -------
        embedding:
            Nx2 t-SNE coordinates.

        data:
            Clean feature dataframe corresponding to the embedding.
        """

        data = self.prepare_features(
            dataframe,
            features,
        )

                                                                        
                                                  
                                                                        

        if len(data) > max_samples:

            data = data.sample(
                n=max_samples,
                random_state=self.random_state,
            )

        X, _ = self.standardize(
            data
        )

                                                        
        if perplexity >= len(X):

            perplexity = max(
                5.0,
                min(
                    30.0,
                    (len(X) - 1) / 3,
                ),
            )

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=self.random_state,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
        )

        embedding = tsne.fit_transform(
            X
        )

        return (
            embedding,
            data,
        )

                                                                        
                
                                                                        

    def plot_tsne(
        self,
        embedding: np.ndarray,
        filename: str = (
            "tsne_projection.png"
        ),
        show: bool = False,
    ) -> Path:
        """
        Plot 2-D t-SNE embedding.
        """

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=12,
            alpha=0.65,
        )

        ax.set_xlabel(
            "t-SNE 1"
        )

        ax.set_ylabel(
            "t-SNE 2"
        )

        ax.set_title(
            "t-SNE Projection"
        )

        ax.grid(
            alpha=0.25
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

                                                                        
                            
                                                                        

    def save_tsne_coordinates(
        self,
        embedding: np.ndarray,
        filename: str = (
            "tsne_coordinates.csv"
        ),
    ) -> Path:
        """
        Save t-SNE coordinates.
        """

        dataframe = pd.DataFrame(
            {
                "tsne_1": embedding[:, 0],
                "tsne_2": embedding[:, 1],
            }
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

    analyzer = DimensionalityAnalyzer()

                                                                    
             
                                                                    

    print()
    print(
        "=" * 70
    )

    print(
        "BUS PCA ANALYSIS"
    )

    print(
        "=" * 70
    )

    (
        bus_pca,
        bus_pca_coordinates,
        bus_loadings,
    ) = analyzer.run_pca(
        dataset.bus,
        analyzer.BUS_FEATURES,
    )

    bus_report = (
        analyzer.pca_variance_report(
            bus_pca
        )
    )

    print(
        "\nExplained variance:"
    )

    for index, value in enumerate(
        bus_report.explained_variance_ratio,
        start=1,
    ):

        print(
            f"  PC{index}: "
            f"{value:.6f}"
        )

    print(
        "\nCumulative variance:"
    )

    for index, value in enumerate(
        bus_report.cumulative_explained_variance,
        start=1,
    ):

        print(
            f"  PC{index}: "
            f"{value:.6f}"
        )

    analyzer.plot_pca_variance(
        bus_pca,
        filename="bus_pca_explained_variance.png",
    )

    analyzer.plot_pca_2d(
        bus_pca_coordinates,
        filename="bus_pca_2d.png",
    )

    analyzer.save_loadings(
        bus_loadings,
        filename="bus_pca_loadings.csv",
    )

    print(
        "\nBus PCA loadings:"
    )

    print(
        bus_loadings.to_string(
            float_format=lambda x:
            f"{x:.6f}"
        )
    )

                                                                    
               
                                                                    

    print()
    print(
        "=" * 70
    )

    print(
        "BUS t-SNE ANALYSIS"
    )

    print(
        "=" * 70
    )

    (
        bus_tsne,
        bus_tsne_data,
    ) = analyzer.run_tsne(
        dataset.bus,
        analyzer.BUS_FEATURES,
        perplexity=30.0,
        max_samples=3000,
    )

    analyzer.plot_tsne(
        bus_tsne,
        filename="bus_tsne_projection.png",
    )

    analyzer.save_tsne_coordinates(
        bus_tsne,
        filename="bus_tsne_coordinates.csv",
    )

    print(
        f"\nt-SNE samples: "
        f"{len(bus_tsne)}"
    )

                                                                    
              
                                                                    

    print()
    print(
        "=" * 70
    )

    print(
        "LINE PCA ANALYSIS"
    )

    print(
        "=" * 70
    )

    (
        line_pca,
        line_pca_coordinates,
        line_loadings,
    ) = analyzer.run_pca(
        dataset.line,
        analyzer.LINE_FEATURES,
    )

    line_report = (
        analyzer.pca_variance_report(
            line_pca
        )
    )

    print(
        "\nExplained variance:"
    )

    for index, value in enumerate(
        line_report.explained_variance_ratio,
        start=1,
    ):

        print(
            f"  PC{index}: "
            f"{value:.6f}"
        )

    print(
        "\nCumulative variance:"
    )

    for index, value in enumerate(
        line_report.cumulative_explained_variance,
        start=1,
    ):

        print(
            f"  PC{index}: "
            f"{value:.6f}"
        )

    analyzer.plot_pca_variance(
        line_pca,
        filename="line_pca_explained_variance.png",
    )

    analyzer.plot_pca_2d(
        line_pca_coordinates,
        filename="line_pca_2d.png",
    )

    analyzer.save_loadings(
        line_loadings,
        filename="line_pca_loadings.csv",
    )

                                                                    
                
                                                                    

    print()
    print(
        "=" * 70
    )

    print(
        "LINE t-SNE ANALYSIS"
    )

    print(
        "=" * 70
    )

    (
        line_tsne,
        line_tsne_data,
    ) = analyzer.run_tsne(
        dataset.line,
        analyzer.LINE_FEATURES,
        perplexity=30.0,
        max_samples=3000,
    )

    analyzer.plot_tsne(
        line_tsne,
        filename="line_tsne_projection.png",
    )

    analyzer.save_tsne_coordinates(
        line_tsne,
        filename="line_tsne_coordinates.csv",
    )

    print(
        f"\nt-SNE samples: "
        f"{len(line_tsne)}"
    )

    print()
    print(
        "=" * 70
    )

    print(
        "DIMENSIONALITY ANALYSIS COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nResults saved to:"
        f"\n{analyzer.output_directory}"
    )