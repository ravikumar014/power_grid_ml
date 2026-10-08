"""
End-to-end dataset construction pipeline.

Pipeline
--------
1. Generate power-grid topology
2. Generate load / DER profiles
3. Run time-series power-flow simulation
4. Validate measurements
5. Preprocess measurements
6. Split chronologically
7. Fit normalization on training data only
8. Generate temporal windows

This module is the reproducible entry point for the complete
data-engineering pipeline.

Usage
-----
python -m src.data.dataset_builder
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.grid_generator import GridGenerator
from src.data.profile_generator import ProfileGenerator
from src.data.time_series_simulator import TimeSeriesSimulator
from src.data.validate_data import MeasurementValidator
from src.data.preprocess import PowerGridPreprocessor
from src.data.split_dataset import TemporalSplitter
from src.data.normalize import PowerGridNormalizer
from src.data.window_generator import TemporalWindowGenerator


@dataclass
class DatasetConfig:
    """
    Configuration for the dataset-building pipeline.
    """

                                                                    
          
                                                                    

    network_name: str = "ieee33"

                                                                    
                
                                                                    

    start: str = "2026-01-01 00:00:00"

    periods: int = 96

    frequency: str = "15min"

    seed: int = 42

                                                                    
         
                                                                    

    der_bus: int = 18

    solar_capacity_mw: float = 0.5

                                                                    
                       
                                                                    

    train_ratio: float = 0.70

    validation_ratio: float = 0.15

    test_ratio: float = 0.15

                                                                    
                   
                                                                    

    normalization_method: str = "standard"

                                                                    
                      
                                                                    

    window_size: int = 60

    window_stride: int = 1

                                                                    
           
                                                                    

    data_root: Path = Path("data")

    @property
    def simulated_root(self) -> Path:
        return self.data_root / "simulated"

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed"

    @property
    def network_path(self) -> Path:
        return (
            self.simulated_root
            / "networks"
            / f"{self.network_name}.json"
        )

    @property
    def profile_path(self) -> Path:
        return (
            self.simulated_root
            / "profiles"
            / "profiles.csv"
        )

    @property
    def measurement_root(self) -> Path:
        return (
            self.simulated_root
            / "measurements"
        )

    @property
    def bus_measurement_path(self) -> Path:
        return (
            self.measurement_root
            / "bus_measurements.csv"
        )

    @property
    def line_measurement_path(self) -> Path:
        return (
            self.measurement_root
            / "line_measurements.csv"
        )

    @property
    def normalized_root(self) -> Path:
        return (
            self.processed_root
            / "normalized"
        )

    @property
    def window_root(self) -> Path:
        return (
            self.processed_root
            / "windows"
        )


class DatasetBuilder:
    """
    Orchestrates the complete power-grid dataset pipeline.
    """

    def __init__(
        self,
        config: DatasetConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else DatasetConfig()
        )

                                                                        
             
                                                                        

    @staticmethod
    def _print_stage(
        number: int,
        title: str,
    ) -> None:

        print()
        print("=" * 72)
        print(
            f"STAGE {number}: {title}"
        )
        print("=" * 72)

                                                                        
                             
                                                                        

    def generate_grid(self) -> None:

        self._print_stage(
            1,
            "GENERATING GRID",
        )

        generator = GridGenerator(
            network_name=self.config.network_name
        )

        generator.generate()

        generator.validate()

        generator.save(
            self.config.network_path
        )

        summary = generator.summary()

        for key, value in summary.items():

            print(
                f"{key:25s}: {value}"
            )

        print(
            f"\nSaved network to:"
            f"\n{self.config.network_path}"
        )

                                                                        
                                 
                                                                        

    def generate_profiles(self) -> None:

        self._print_stage(
            2,
            "GENERATING LOAD + DER PROFILES",
        )

        generator = ProfileGenerator(
            start=self.config.start,
            periods=self.config.periods,
            freq=self.config.frequency,
            seed=self.config.seed,
        )

        profiles = generator.generate_profiles(
            base_load_mw=1.0,
            solar_capacity_mw=(
                self.config.solar_capacity_mw
            ),
        )

        generator.save(
            profiles,
            self.config.profile_path,
        )

        print(
            f"Generated {len(profiles)} timesteps."
        )

        print(
            f"Start: "
            f"{profiles['timestamp'].min()}"
        )

        print(
            f"End:   "
            f"{profiles['timestamp'].max()}"
        )

        print(
            f"\nSaved profiles to:"
            f"\n{self.config.profile_path}"
        )

                                                                        
                                         
                                                                        

    def simulate(self) -> None:

        self._print_stage(
            3,
            "RUNNING POWER-FLOW SIMULATION",
        )

        simulator = TimeSeriesSimulator(
            network_name=self.config.network_name,
            profile_path=self.config.profile_path,
            der_bus=self.config.der_bus,
            solar_capacity_mw=(
                self.config.solar_capacity_mw
            ),
        )

        (
            bus_measurements,
            line_measurements,
        ) = simulator.run()

        simulator.save_results(
            bus_measurements,
            line_measurements,
            output_directory=(
                self.config.measurement_root
            ),
        )

        print(
            f"\nBus rows: "
            f"{len(bus_measurements)}"
        )

        print(
            f"Line rows: "
            f"{len(line_measurements)}"
        )

                                                                        
                        
                                                                        

    def validate(self) -> None:

        self._print_stage(
            4,
            "VALIDATING MEASUREMENTS",
        )

        bus_data = pd.read_csv(
            self.config.bus_measurement_path,
            parse_dates=["timestamp"],
        )

        line_data = pd.read_csv(
            self.config.line_measurement_path,
            parse_dates=["timestamp"],
        )

        validator = MeasurementValidator()

        bus_report = (
            validator.validate_bus_measurements(
                bus_data
            )
        )

        line_report = (
            validator.validate_line_measurements(
                line_data
            )
        )

        print(
            bus_report.summary()
        )

        print(
            line_report.summary()
        )

        if not bus_report.passed:
            raise RuntimeError(
                "Bus measurement validation failed."
            )

        if not line_report.passed:
            raise RuntimeError(
                "Line measurement validation failed."
            )

        print(
            "\nValidation successful."
        )

                                                                        
                          
                                                                        

    def preprocess(self) -> None:

        self._print_stage(
            5,
            "PREPROCESSING MEASUREMENTS",
        )

        bus_data = pd.read_csv(
            self.config.bus_measurement_path,
            parse_dates=["timestamp"],
        )

        preprocessor = PowerGridPreprocessor(
            missing_strategy="interpolate",
            add_time_features=True,
        )

        (
            processed_bus,
            report,
        ) = preprocessor.process(
            bus_data,
            dataset_type="bus",
        )

        output_path = (
            self.config.processed_root
            / "bus_measurements_processed.csv"
        )

        preprocessor.save(
            processed_bus,
            output_path,
        )

        print(
            report.summary()
        )

        print(
            f"\nSaved processed data to:"
            f"\n{output_path}"
        )

                                                                        
                              
                                                                        

    def split(self) -> None:

        self._print_stage(
            6,
            "TEMPORAL TRAIN / VALIDATION / TEST SPLIT",
        )

        input_path = (
            self.config.processed_root
            / "bus_measurements_processed.csv"
        )

        data = pd.read_csv(
            input_path,
            parse_dates=["timestamp"],
        )

        splitter = TemporalSplitter(
            train_ratio=(
                self.config.train_ratio
            ),
            validation_ratio=(
                self.config.validation_ratio
            ),
            test_ratio=(
                self.config.test_ratio
            ),
            entity_column="bus_id",
        )

        (
            train,
            validation,
            test,
            report,
        ) = splitter.split(data)

        splitter.save_splits(
            train,
            validation,
            test,
            output_directory=(
                self.config.processed_root
            ),
        )

        print(
            report.summary()
        )

                                                                        
                         
                                                                        

    def normalize(self) -> None:

        self._print_stage(
            7,
            "NORMALIZATION",
        )

        train_path = (
            self.config.processed_root
            / "train.csv"
        )

        validation_path = (
            self.config.processed_root
            / "validation.csv"
        )

        test_path = (
            self.config.processed_root
            / "test.csv"
        )

        train = pd.read_csv(
            train_path,
            parse_dates=["timestamp"],
        )

        validation = pd.read_csv(
            validation_path,
            parse_dates=["timestamp"],
        )

        test = pd.read_csv(
            test_path,
            parse_dates=["timestamp"],
        )

        normalizer = PowerGridNormalizer(
            method=(
                self.config.normalization_method
            )
        )

                                                                        
                   
                            
                                                                        

        normalizer.fit(
            train
        )

        print(
            "\nNormalizer fitted using TRAIN data only."
        )

        print(
            "\nLearned parameters:"
        )

        for feature, parameters in (
            normalizer.parameters.items()
        ):

            print(
                f"  {feature}: "
                f"{parameters}"
            )

        train_normalized = (
            normalizer.transform(
                train
            )
        )

        validation_normalized = (
            normalizer.transform(
                validation
            )
        )

        test_normalized = (
            normalizer.transform(
                test
            )
        )

        self.config.normalized_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_normalized.to_csv(
            self.config.normalized_root
            / "train.csv",
            index=False,
        )

        validation_normalized.to_csv(
            self.config.normalized_root
            / "validation.csv",
            index=False,
        )

        test_normalized.to_csv(
            self.config.normalized_root
            / "test.csv",
            index=False,
        )

        normalizer.save(
            self.config.normalized_root
            / "normalizer.pkl"
        )

        print(
            "\nNormalized datasets saved."
        )

                                                                        
                                         
                                                                        

    def generate_windows(self) -> None:

        self._print_stage(
            8,
            "GENERATING TEMPORAL WINDOWS",
        )

        feature_columns = [
            "voltage_pu",
            "angle_deg",
            "active_power_mw",
            "reactive_power_mvar",
        ]

        generator = TemporalWindowGenerator(
            window_size=(
                self.config.window_size
            ),
            stride=(
                self.config.window_stride
            ),
            entity_column="bus_id",
            timestamp_column="timestamp",
        )

        self.config.window_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        datasets = {}

        for split_name in [
            "train",
            "validation",
            "test",
        ]:

            input_path = (
                self.config.normalized_root
                / f"{split_name}.csv"
            )

            data = pd.read_csv(
                input_path,
                parse_dates=["timestamp"],
            )

            dataset = generator.generate(
                data,
                feature_columns=feature_columns,
            )

            output_path = (
                self.config.window_root
                / f"{split_name}.npz"
            )

            generator.save(
                dataset,
                output_path,
            )

            datasets[
                split_name
            ] = dataset

            print(
                f"{split_name:12s}: "
                f"{dataset.shape}"
            )

        print(
            "\nTemporal windows saved."
        )

                                                                        
                       
                                                                        

    def build(self) -> None:
        """
        Execute the complete dataset pipeline.
        """

        print()
        print("#" * 72)
        print("# POWER GRID DATASET BUILDER")
        print("#" * 72)

        print(
            "\nConfiguration:"
        )

        print(
            f"  Network       : "
            f"{self.config.network_name}"
        )

        print(
            f"  Timesteps     : "
            f"{self.config.periods}"
        )

        print(
            f"  Frequency     : "
            f"{self.config.frequency}"
        )

        print(
            f"  DER bus       : "
            f"{self.config.der_bus}"
        )

        print(
            f"  Solar capacity: "
            f"{self.config.solar_capacity_mw} MW"
        )

        print(
            f"  Window size   : "
            f"{self.config.window_size}"
        )

                                                                        
                         
                                                                        

        self.generate_grid()

        self.generate_profiles()

        self.simulate()

        self.validate()

        self.preprocess()

        self.split()

        self.normalize()

        self.generate_windows()

                                                                        
                        
                                                                        

        self._print_stage(
            9,
            "PIPELINE COMPLETE",
        )

        print(
            "\nDataset successfully generated."
        )

        print(
            "\nGenerated structure:"
        )

        print(
            f"""
data/
├── simulated/
│   ├── networks/
│   │   └── {self.config.network_name}.json
│   ├── profiles/
│   │   └── profiles.csv
│   └── measurements/
│       ├── bus_measurements.csv
│       └── line_measurements.csv
│
└── processed/
    ├── bus_measurements_processed.csv
    ├── train.csv
    ├── validation.csv
    ├── test.csv
    │
    ├── normalized/
    │   ├── train.csv
    │   ├── validation.csv
    │   ├── test.csv
    │   └── normalizer.pkl
    │
    └── windows/
        ├── train.npz
        ├── validation.npz
        └── test.npz
"""
        )


def main() -> None:

    config = DatasetConfig()

    builder = DatasetBuilder(
        config=config
    )

    builder.build()


if __name__ == "__main__":
    main()