"""
Generate an anomalous evaluation dataset from the raw IEEE-33 data.

IMPORTANT
---------
The original simulated data is never modified.

Anomalies are injected only into the evaluation period.

Outputs
-------
data/anomalies/raw/bus_anomalous.csv
data/anomalies/raw/line_anomalous.csv
data/anomalies/ground_truth.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.anomaly_injector import (
    AnomalyInjector,
)


                                                                        
       
                                                                        

BUS_INPUT = Path(
    "data/simulated/measurements/bus_measurements.csv"
)

LINE_INPUT = Path(
    "data/simulated/measurements/line_measurements.csv"
)

OUTPUT_DIR = Path(
    "data/anomalies"
)

RAW_OUTPUT_DIR = (
    OUTPUT_DIR / "raw"
)

GROUND_TRUTH_OUTPUT = (
    OUTPUT_DIR / "ground_truth.csv"
)


                                                                        
               
                                                                        

RANDOM_STATE = 42

                                     
TEST_START = pd.Timestamp(
    "2026-01-01 19:00:00"
)


                                                                        
      
                                                                        


def main() -> None:

    print()
    print("=" * 70)
    print(
        "GENERATING ANOMALOUS EVALUATION DATASET"
    )
    print("=" * 70)

                                                                    
                   
                                                                    

    print()
    print("Loading raw datasets...")

    bus_data = pd.read_csv(
        BUS_INPUT,
        parse_dates=["timestamp"],
    )

    line_data = pd.read_csv(
        LINE_INPUT,
        parse_dates=["timestamp"],
    )

    print(
        f"Bus data : {bus_data.shape}"
    )

    print(
        f"Line data: {line_data.shape}"
    )

                                                                    
                     
                                                                    

    injector = AnomalyInjector(
        random_state=RANDOM_STATE
    )

                                                                    
                   
                                                                    

    print()
    print(
        "-" * 70
    )
    print(
        "BUS ANOMALIES"
    )
    print(
        "-" * 70
    )

                                                                    
                                 
                                                                    

    bus_data, event = (
        injector.inject_voltage_deviation(
            bus_data=bus_data,
            bus_id=17,
            start_timestamp=TEST_START,
            duration_steps=4,
            deviation=0.10,
        )
    )

    print(
        f"Event {event.event_id}: "
        f"{event.anomaly_type} "
        f"Bus {event.entity_id}"
    )

                                                                    
                             
                                                                    

    bus_data, event = (
        injector.inject_power_spike(
            bus_data=bus_data,
            bus_id=14,
            start_timestamp=(
                TEST_START
                + pd.Timedelta(
                    minutes=60
                )
            ),
            duration_steps=3,
            multiplier=1.50,
        )
    )

    print(
        f"Event {event.event_id}: "
        f"{event.anomaly_type} "
        f"Bus {event.entity_id}"
    )

                                                                    
                              
                                                                    

    bus_data, event = (
        injector.inject_sensor_spike(
            bus_data=bus_data,
            bus_id=29,
            start_timestamp=(
                TEST_START
                + pd.Timedelta(
                    minutes=120
                )
            ),
            measurement="voltage_pu",
            magnitude=0.20,
        )
    )

    print(
        f"Event {event.event_id}: "
        f"{event.anomaly_type} "
        f"Bus {event.entity_id}"
    )

                                                                    
                    
                                                                    

    print()
    print(
        "-" * 70
    )
    print(
        "LINE ANOMALIES"
    )
    print(
        "-" * 70
    )

                                                                    
                               
                                                                    

    line_data, event = (
        injector.inject_line_overload(
            line_data=line_data,
            line_id=1,
            start_timestamp=(
                TEST_START
                + pd.Timedelta(
                    minutes=30
                )
            ),
            duration_steps=4,
            multiplier=1.50,
        )
    )

    print(
        f"Event {event.event_id}: "
        f"{event.anomaly_type} "
        f"Line {event.entity_id}"
    )

                                                                    
                               
                                                                    

    line_data, event = (
        injector.inject_line_overload(
            line_data=line_data,
            line_id=5,
            start_timestamp=(
                TEST_START
                + pd.Timedelta(
                    minutes=180
                )
            ),
            duration_steps=3,
            multiplier=1.40,
        )
    )

    print(
        f"Event {event.event_id}: "
        f"{event.anomaly_type} "
        f"Line {event.entity_id}"
    )

                                                                    
                                  
                                                                    

    print()
    print(
        "-" * 70
    )
    print(
        "VERIFYING TEMPORAL ISOLATION"
    )
    print(
        "-" * 70
    )

    ground_truth = (
        injector.ground_truth()
    )

    earliest_anomaly = ground_truth[
        "start_timestamp"
    ].min()

    print(
        f"Earliest anomaly: "
        f"{earliest_anomaly}"
    )

    if earliest_anomaly < TEST_START:

        raise RuntimeError(
            "Anomaly was injected before "
            "the test period."
        )

    print(
        "Training period remains clean."
    )

                                                                    
          
                                                                    

    RAW_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    bus_data.to_csv(
        RAW_OUTPUT_DIR
        / "bus_anomalous.csv",
        index=False,
    )

    line_data.to_csv(
        RAW_OUTPUT_DIR
        / "line_anomalous.csv",
        index=False,
    )

    ground_truth.to_csv(
        GROUND_TRUTH_OUTPUT,
        index=False,
    )

                                                                    
             
                                                                    

    print()
    print("=" * 70)
    print(
        "ANOMALOUS DATASET GENERATED"
    )
    print("=" * 70)

    print()
    print(
        f"Bus output:\n"
        f"{RAW_OUTPUT_DIR / 'bus_anomalous.csv'}"
    )

    print(
        f"Line output:\n"
        f"{RAW_OUTPUT_DIR / 'line_anomalous.csv'}"
    )

    print(
        f"Ground truth:\n"
        f"{GROUND_TRUTH_OUTPUT}"
    )

    print()
    print(
        "Ground-truth events:"
    )

    print(
        ground_truth.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()