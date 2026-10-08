from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

@dataclass
class ElectricalFeatureConfig:
    nominal_voltage_pu: float = 1.0
    voltage_min_pu: float = 0.95
    voltage_max_pu: float = 1.05
    thermal_limit_percent: float = 100.0
    epsilon: float = 1e-8


class ElectricalFeatureEngineer:
    def __init__(self, config: ElectricalFeatureConfig | None = None) -> None:
        self.config = (
            config
            if config is not None
            else ElectricalFeatureConfig()
        )

        if (self.config.nominal_voltage_pu <= 0):
            raise ValueError("nominal_voltage_pu must be positive.")

        if (self.config.voltage_min_pu >= self.config.voltage_max_pu):
            raise ValueError("voltage_min_pu must be smaller " "than voltage_max_pu.")

        if (self.config.thermal_limit_percent <= 0):
            raise ValueError("thermal_limit_percent must be positive.")
        
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
    def _numeric(
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:

        result = dataframe[
            columns
        ].copy()

        for column in columns:

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

        result = result.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return result


    def generate_bus_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        required = [
            "voltage_pu",
            "angle_deg",
            "active_power_mw",
            "reactive_power_mvar",
        ]

        self._validate_columns(
            dataframe,
            required,
        )

        data = self._numeric(
            dataframe,
            required,
        )

        V = data[
            "voltage_pu"
        ]

        P = data[
            "active_power_mw"
        ]

        Q = data[
            "reactive_power_mvar"
        ]

        voltage_deviation = (
            V
            - self.config.nominal_voltage_pu
        )


        absolute_voltage_deviation = (
            voltage_deviation.abs()
        )


        relative_voltage_deviation = (
            voltage_deviation
            / self.config.nominal_voltage_pu
        )


        lower_voltage_margin = (
            V
            - self.config.voltage_min_pu
        )

        upper_voltage_margin = (
            self.config.voltage_max_pu
            - V
        )


        undervoltage = (
            V
            < self.config.voltage_min_pu
        )

        overvoltage = (
            V
            > self.config.voltage_max_pu
        )

        voltage_violation = (
            undervoltage
            | overvoltage
        )


        apparent_power = np.sqrt(
            P.pow(2)
            + Q.pow(2)
        )

        power_factor = (
            P.abs()
            / (
                apparent_power
                + self.config.epsilon
            )
        )

        power_factor = power_factor.clip(
            lower=0.0,
            upper=1.0,
        )


        power_factor_angle_deg = (
            np.degrees(
                np.arctan2(
                    Q,
                    P,
                )
            )
        )


        pq_ratio = (
            P
            / (
                Q.abs()
                + self.config.epsilon
            )
        )


        pq_magnitude = np.sqrt(
            P.pow(2)
            + Q.pow(2)
        )


        features = pd.DataFrame(
            {
                "voltage_deviation_pu": (
                    voltage_deviation
                ),

                "absolute_voltage_deviation_pu": (
                    absolute_voltage_deviation
                ),

                "relative_voltage_deviation": (
                    relative_voltage_deviation
                ),

                "lower_voltage_margin_pu": (
                    lower_voltage_margin
                ),

                "upper_voltage_margin_pu": (
                    upper_voltage_margin
                ),

                "undervoltage": (
                    undervoltage.astype(int)
                ),

                "overvoltage": (
                    overvoltage.astype(int)
                ),

                "voltage_violation": (
                    voltage_violation.astype(int)
                ),

                "apparent_power_mva": (
                    apparent_power
                ),

                "power_factor": (
                    power_factor
                ),

                "power_factor_angle_deg": (
                    power_factor_angle_deg
                ),

                "pq_ratio": (
                    pq_ratio
                ),

                "pq_magnitude": (
                    pq_magnitude
                ),
            },
            index=dataframe.index,
        )

        return features


    def generate_line_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate electrical features from line measurements.

        Required columns
        ----------------
        current_from_ka
        current_to_ka
        loading_percent
        """

        required = [
            "current_from_ka",
            "current_to_ka",
            "loading_percent",
        ]

        self._validate_columns(
            dataframe,
            required,
        )

        data = self._numeric(
            dataframe,
            required,
        )

        current_from = data[
            "current_from_ka"
        ]

        current_to = data[
            "current_to_ka"
        ]

        loading = data[
            "loading_percent"
        ]


        abs_current_from = (
            current_from.abs()
        )

        abs_current_to = (
            current_to.abs()
        )


        mean_current = (
            (
                abs_current_from
                + abs_current_to
            )
            / 2.0
        )


        current_difference = (
            abs_current_from
            - abs_current_to
        )

        absolute_current_difference = (
            current_difference.abs()
        )


        denominator = (
            pd.concat(
                [
                    abs_current_from,
                    abs_current_to,
                ],
                axis=1,
            )
            .max(axis=1)
            + self.config.epsilon
        )

        current_imbalance = (
            absolute_current_difference
            / denominator
        )


        loading_margin = (
            self.config.thermal_limit_percent
            - loading
        )


        loading_ratio = (
            loading
            / self.config.thermal_limit_percent
        )


        thermal_overload = (
            loading
            > self.config.thermal_limit_percent
        )


        loading_above_90 = (
            loading >= 90.0
        )

        loading_above_95 = (
            loading >= 95.0
        )


        features = pd.DataFrame(
            {
                "absolute_current_from_ka": (
                    abs_current_from
                ),

                "absolute_current_to_ka": (
                    abs_current_to
                ),

                "mean_current_ka": (
                    mean_current
                ),

                "current_difference_ka": (
                    current_difference
                ),

                "absolute_current_difference_ka": (
                    absolute_current_difference
                ),

                "current_imbalance": (
                    current_imbalance
                ),

                "loading_margin_percent": (
                    loading_margin
                ),

                "loading_ratio": (
                    loading_ratio
                ),

                "loading_above_90": (
                    loading_above_90.astype(int)
                ),

                "loading_above_95": (
                    loading_above_95.astype(int)
                ),

                "thermal_overload": (
                    thermal_overload.astype(int)
                ),
            },
            index=dataframe.index,
        )

        return features


    def transform_bus(
        self,
        dataframe: pd.DataFrame,
        preserve_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if preserve_columns is None:
            preserve_columns = []

        self._validate_columns(
            dataframe,
            preserve_columns,
        )

        metadata = (
            dataframe[
                preserve_columns
            ].copy()
        )

        features = (
            self.generate_bus_features(
                dataframe
            )
        )

        return pd.concat(
            [
                metadata.reset_index(
                    drop=True
                ),
                features.reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )


    def transform_line(
        self,
        dataframe: pd.DataFrame,
        preserve_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if preserve_columns is None:
            preserve_columns = []

        self._validate_columns(
            dataframe,
            preserve_columns,
        )

        metadata = (
            dataframe[
                preserve_columns
            ].copy()
        )

        features = (
            self.generate_line_features(
                dataframe
            )
        )

        return pd.concat(
            [
                metadata.reset_index(
                    drop=True
                ),
                features.reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )


    def transform(
        self,
        bus_dataframe: pd.DataFrame,
        line_dataframe: pd.DataFrame,
        bus_metadata: list[str] | None = None,
        line_metadata: list[str] | None = None,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:
        bus_features = self.transform_bus(
            bus_dataframe,
            preserve_columns=bus_metadata,
        )

        line_features = self.transform_line(
            line_dataframe,
            preserve_columns=line_metadata,
        )

        return (
            bus_features,
            line_features,
        )


    @staticmethod
    def save(
        dataframe: pd.DataFrame,
        path: str | Path,
    ) -> Path:
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
        "ELECTRICAL FEATURE ENGINEERING"
    )

    print(
        "=" * 70
    )


    loader = EDADataLoader()

    dataset = loader.load_all()


    config = ElectricalFeatureConfig(
        nominal_voltage_pu=1.0,
        voltage_min_pu=0.95,
        voltage_max_pu=1.05,
        thermal_limit_percent=100.0,
    )

    engineer = (
        ElectricalFeatureEngineer(
            config=config
        )
    )


    print(
        "\nGenerating bus electrical features..."
    )

    bus_features = (
        engineer.transform_bus(
            dataset.bus,
            preserve_columns=[
                "timestamp",
                "bus_id",
            ],
        )
    )

    bus_path = engineer.save(
        bus_features,
        "data/features/bus/"
        "electrical_features.csv",
    )

    print(
        f"Saved to:"
        f"\n{bus_path}"
    )

    print(
        f"\nBus feature matrix shape:"
        f" {bus_features.shape}"
    )

    print(
        "\nBus electrical features:"
    )

    print(
        bus_features.head(
            10
        ).to_string(
            index=False
        )
    )


    print(
        "\nGenerating line electrical features..."
    )

    line_features = (
        engineer.transform_line(
            dataset.line,
            preserve_columns=[
                "timestamp",
                "line_id",
            ],
        )
    )

    line_path = engineer.save(
        line_features,
        "data/features/line/"
        "electrical_features.csv",
    )

    print(
        f"Saved to:"
        f"\n{line_path}"
    )

    print(
        f"\nLine feature matrix shape:"
        f" {line_features.shape}"
    )

    print(
        "\nLine electrical features:"
    )

    print(
        line_features.head(
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
        "ELECTRICAL FEATURE ENGINEERING COMPLETE"
    )

    print(
        "=" * 70
    )