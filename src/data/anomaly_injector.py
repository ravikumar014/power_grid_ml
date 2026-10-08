"""
Synthetic anomaly injection framework for power-grid data.

The injector operates on RAW bus/line measurements, before feature
engineering.

Each injected anomaly produces:
    1. modified measurements
    2. event-level ground truth

The design is intentionally deterministic when a random seed is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


                                                                        
               
                                                                        


@dataclass
class AnomalyEvent:
    """
    Description of one injected anomaly event.
    """

    event_id: int

    anomaly_type: str

    entity_type: Literal[
        "bus",
        "line",
    ]

    entity_id: int

    start_timestamp: pd.Timestamp

    end_timestamp: pd.Timestamp

    severity: float

    affected_measurement: str


                                                                        
                  
                                                                        


class AnomalyInjector:
    """
    Inject synthetic but interpretable anomalies into raw measurements.

    Parameters
    ----------
    random_state:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        random_state: int = 42,
    ) -> None:

        self.random_state = random_state

        self.rng = np.random.default_rng(
            random_state
        )

        self.events: list[
            AnomalyEvent
        ] = []

                                                                        
                
                                                                        

    @staticmethod
    def _validate_timestamp(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Ensure timestamp exists and is datetime.
        """

        if "timestamp" not in dataframe.columns:

            raise ValueError(
                "Dataframe must contain a 'timestamp' column."
            )

        data = dataframe.copy()

        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce",
        )

        if data["timestamp"].isna().any():

            raise ValueError(
                "Invalid timestamps detected."
            )

        return data

    @staticmethod
    def _validate_columns(
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> None:

        missing = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing:

            raise ValueError(
                f"Required columns missing: {missing}"
            )

                                                                        
                    
                                                                        

    def _register_event(
        self,
        anomaly_type: str,
        entity_type: str,
        entity_id: int,
        start_timestamp: pd.Timestamp,
        end_timestamp: pd.Timestamp,
        severity: float,
        affected_measurement: str,
    ) -> AnomalyEvent:

        event = AnomalyEvent(
            event_id=len(self.events),
            anomaly_type=anomaly_type,
            entity_type=entity_type,
            entity_id=int(entity_id),
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            severity=float(severity),
            affected_measurement=affected_measurement,
        )

        self.events.append(
            event
        )

        return event

                                                                        
                     
                                                                        

    def inject_voltage_deviation(
        self,
        bus_data: pd.DataFrame,
        bus_id: int,
        start_timestamp: pd.Timestamp,
        duration_steps: int = 4,
        deviation: float = 0.10,
    ) -> tuple[
        pd.DataFrame,
        AnomalyEvent,
    ]:
        """
        Inject a sustained voltage deviation at a bus.

        Example:
            voltage_pu = 0.98
            deviation = 0.10

        produces approximately:
            voltage_pu = 0.88

        This represents an undervoltage-type event.
        """

        data = self._validate_timestamp(
            bus_data
        )

        self._validate_columns(
            data,
            [
                "bus_id",
                "voltage_pu",
            ],
        )

        timestamps = (
            data[
                "timestamp"
            ]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        if start_timestamp not in set(
            timestamps
        ):

            raise ValueError(
                f"Start timestamp {start_timestamp} "
                "does not exist in the dataset."
            )

        start_index = timestamps[
            timestamps == start_timestamp
        ].index[0]

        end_index = min(
            start_index
            + duration_steps
            - 1,
            len(timestamps) - 1,
        )

        end_timestamp = timestamps[
            end_index
        ]

        affected_timestamps = timestamps[
            start_index:end_index + 1
        ]

        mask = (
            data["bus_id"].eq(
                bus_id
            )
            & data["timestamp"].isin(
                affected_timestamps
            )
        )

        if not mask.any():

            raise ValueError(
                f"Bus {bus_id} was not found "
                "in the selected time window."
            )

        data.loc[
            mask,
            "voltage_pu",
        ] = (
            data.loc[
                mask,
                "voltage_pu",
            ]
            - deviation
        )

        event = self._register_event(
            anomaly_type="voltage_deviation",
            entity_type="bus",
            entity_id=bus_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            severity=abs(deviation),
            affected_measurement="voltage_pu",
        )

        return data, event

                                                                        
                 
                                                                        

    def inject_power_spike(
        self,
        bus_data: pd.DataFrame,
        bus_id: int,
        start_timestamp: pd.Timestamp,
        duration_steps: int = 2,
        multiplier: float = 1.50,
    ) -> tuple[
        pd.DataFrame,
        AnomalyEvent,
    ]:
        """
        Inject an active-power demand/generation spike.

        multiplier > 1 increases magnitude.

        Example:
            P = 10 MW
            multiplier = 1.5

        gives:
            P = 15 MW
        """

        data = self._validate_timestamp(
            bus_data
        )

        self._validate_columns(
            data,
            [
                "bus_id",
                "active_power_mw",
            ],
        )

        timestamps = (
            data[
                "timestamp"
            ]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        if start_timestamp not in set(
            timestamps
        ):

            raise ValueError(
                f"Start timestamp {start_timestamp} "
                "does not exist in the dataset."
            )

        start_index = timestamps[
            timestamps == start_timestamp
        ].index[0]

        end_index = min(
            start_index
            + duration_steps
            - 1,
            len(timestamps) - 1,
        )

        end_timestamp = timestamps[
            end_index
        ]

        affected_timestamps = timestamps[
            start_index:end_index + 1
        ]

        mask = (
            data["bus_id"].eq(
                bus_id
            )
            & data["timestamp"].isin(
                affected_timestamps
            )
        )

        if not mask.any():

            raise ValueError(
                f"Bus {bus_id} was not found "
                "in the selected time window."
            )

        data.loc[
            mask,
            "active_power_mw",
        ] = (
            data.loc[
                mask,
                "active_power_mw",
            ]
            * multiplier
        )

        event = self._register_event(
            anomaly_type="power_spike",
            entity_type="bus",
            entity_id=bus_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            severity=abs(
                multiplier - 1.0
            ),
            affected_measurement="active_power_mw",
        )

        return data, event

                                                                        
                  
                                                                        

    def inject_sensor_spike(
        self,
        bus_data: pd.DataFrame,
        bus_id: int,
        start_timestamp: pd.Timestamp,
        measurement: str = "voltage_pu",
        magnitude: float = 0.20,
    ) -> tuple[
        pd.DataFrame,
        AnomalyEvent,
    ]:
        """
        Inject an isolated measurement spike.

        This represents a sensor-level anomaly rather than necessarily
        a physical grid disturbance.
        """

        data = self._validate_timestamp(
            bus_data
        )

        self._validate_columns(
            data,
            [
                "bus_id",
                measurement,
            ],
        )

        mask = (
            data["bus_id"].eq(
                bus_id
            )
            & data["timestamp"].eq(
                pd.Timestamp(
                    start_timestamp
                )
            )
        )

        if not mask.any():

            raise ValueError(
                "Target bus/timestamp combination "
                "was not found."
            )

        data.loc[
            mask,
            measurement,
        ] = (
            data.loc[
                mask,
                measurement,
            ]
            + magnitude
        )

        timestamp = pd.Timestamp(
            start_timestamp
        )

        event = self._register_event(
            anomaly_type="sensor_spike",
            entity_type="bus",
            entity_id=bus_id,
            start_timestamp=timestamp,
            end_timestamp=timestamp,
            severity=abs(magnitude),
            affected_measurement=measurement,
        )

        return data, event

                                                                        
                   
                                                                        

    def inject_line_overload(
        self,
        line_data: pd.DataFrame,
        line_id: int,
        start_timestamp: pd.Timestamp,
        duration_steps: int = 4,
        multiplier: float = 1.50,
    ) -> tuple[
        pd.DataFrame,
        AnomalyEvent,
    ]:
        """
        Inject a line-loading anomaly.

        The loading percentage and currents are increased together.
        """

        data = self._validate_timestamp(
            line_data
        )

        self._validate_columns(
            data,
            [
                "line_id",
                "loading_percent",
                "current_from_ka",
                "current_to_ka",
            ],
        )

        timestamps = (
            data[
                "timestamp"
            ]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        if start_timestamp not in set(
            timestamps
        ):

            raise ValueError(
                f"Start timestamp {start_timestamp} "
                "does not exist."
            )

        start_index = timestamps[
            timestamps == start_timestamp
        ].index[0]

        end_index = min(
            start_index
            + duration_steps
            - 1,
            len(timestamps) - 1,
        )

        end_timestamp = timestamps[
            end_index
        ]

        affected_timestamps = timestamps[
            start_index:end_index + 1
        ]

        mask = (
            data["line_id"].eq(
                line_id
            )
            & data["timestamp"].isin(
                affected_timestamps
            )
        )

        if not mask.any():

            raise ValueError(
                f"Line {line_id} was not found "
                "in the selected time window."
            )

        data.loc[
            mask,
            "loading_percent",
        ] = (
            data.loc[
                mask,
                "loading_percent",
            ]
            * multiplier
        )

        data.loc[
            mask,
            "current_from_ka",
        ] = (
            data.loc[
                mask,
                "current_from_ka",
            ]
            * multiplier
        )

        data.loc[
            mask,
            "current_to_ka",
        ] = (
            data.loc[
                mask,
                "current_to_ka",
            ]
            * multiplier
        )

        event = self._register_event(
            anomaly_type="line_overload",
            entity_type="line",
            entity_id=line_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            severity=abs(
                multiplier - 1.0
            ),
            affected_measurement="loading_percent",
        )

        return data, event

                                                                        
                  
                                                                        

    def ground_truth(
        self,
    ) -> pd.DataFrame:
        """
        Return registered anomaly events as a dataframe.
        """

        if not self.events:

            return pd.DataFrame(
                columns=[
                    "event_id",
                    "anomaly_type",
                    "entity_type",
                    "entity_id",
                    "start_timestamp",
                    "end_timestamp",
                    "severity",
                    "affected_measurement",
                ]
            )

        return pd.DataFrame(
            [
                {
                    "event_id": event.event_id,
                    "anomaly_type": event.anomaly_type,
                    "entity_type": event.entity_type,
                    "entity_id": event.entity_id,
                    "start_timestamp": event.start_timestamp,
                    "end_timestamp": event.end_timestamp,
                    "severity": event.severity,
                    "affected_measurement": (
                        event.affected_measurement
                    ),
                }
                for event in self.events
            ]
        )

                                                                        
           
                                                                        

    def reset(self) -> None:
        """
        Remove all registered events and reset the RNG.
        """

        self.events = []

        self.rng = np.random.default_rng(
            self.random_state
        )


                                                                        
               
                                                                        


if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "ANOMALY INJECTION FRAMEWORK"
    )
    print("=" * 70)

                                                                    
                                
                                                                    

    timestamps = pd.date_range(
        "2026-01-01 00:00:00",
        periods=10,
        freq="15min",
    )

    bus_data = pd.DataFrame(
        {
            "timestamp": np.repeat(
                timestamps,
                3,
            ),
            "bus_id": np.tile(
                [0, 1, 2],
                len(timestamps),
            ),
            "voltage_pu": np.tile(
                [1.00, 0.98, 1.02],
                len(timestamps),
            ),
            "active_power_mw": np.tile(
                [10.0, 20.0, 15.0],
                len(timestamps),
            ),
            "reactive_power_mvar": np.tile(
                [2.0, 4.0, 3.0],
                len(timestamps),
            ),
        }
    )

                                                                    
              
                                                                    

    injector = AnomalyInjector(
        random_state=42
    )

                                                                    
                                               
                                                                    

    available_timestamps = (
        bus_data["timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    start_timestamp = available_timestamps.iloc[4]

                                                                    
                     
                                                                    

    modified_data, event = (
        injector.inject_voltage_deviation(
            bus_data=bus_data,
            bus_id=1,
            start_timestamp=start_timestamp,
            duration_steps=3,
            deviation=0.10,
        )
    )

    print()
    print(
        "Injected event:"
    )

    print(event)

                                                                    
                  
                                                                    

    print()
    print(
        "Ground truth:"
    )

    print(
        injector.ground_truth()
    )

    print()
    print(
        "Original voltage:"
    )

    print(
        bus_data[
            bus_data["bus_id"] == 1
        ][
            [
                "timestamp",
                "voltage_pu",
            ]
        ]
        .head(8)
        .to_string(index=False)
    )

    print()
    print(
        "Modified voltage:"
    )

    print(
        modified_data[
            modified_data["bus_id"] == 1
        ][
            [
                "timestamp",
                "voltage_pu",
            ]
        ]
        .head(8)
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print(
        "ANOMALY INJECTOR TEST COMPLETE"
    )
    print("=" * 70)