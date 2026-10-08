"""
Ground-truth label generation for power-grid anomaly evaluation.

Converts event-level anomaly descriptions into row-level labels that
can be directly aligned with model predictions.

Bus rows are identified by:
    timestamp + bus_id

Line rows are identified by:
    timestamp + line_id
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


                                                                        
       
                                                                        

GROUND_TRUTH_PATH = Path(
    "data/anomalies/ground_truth.csv"
)

BUS_DATA_PATH = Path(
    "data/anomalies/raw/bus_anomalous.csv"
)

LINE_DATA_PATH = Path(
    "data/anomalies/raw/line_anomalous.csv"
)

OUTPUT_DIR = Path(
    "data/anomalies/labels"
)


                                                                        
            
                                                                        


REQUIRED_EVENT_COLUMNS = [
    "event_id",
    "anomaly_type",
    "entity_type",
    "entity_id",
    "start_timestamp",
    "end_timestamp",
    "severity",
    "affected_measurement",
]


def validate_ground_truth(
    ground_truth: pd.DataFrame,
) -> None:
    """Validate the event-level ground truth."""

    missing = [
        column
        for column in REQUIRED_EVENT_COLUMNS
        if column not in ground_truth.columns
    ]

    if missing:
        raise ValueError(
            f"Ground truth missing columns: {missing}"
        )

    ground_truth["start_timestamp"] = pd.to_datetime(
        ground_truth["start_timestamp"],
        errors="coerce",
    )

    ground_truth["end_timestamp"] = pd.to_datetime(
        ground_truth["end_timestamp"],
        errors="coerce",
    )

    if (
        ground_truth["start_timestamp"].isna().any()
        or ground_truth["end_timestamp"].isna().any()
    ):
        raise ValueError(
            "Ground truth contains invalid timestamps."
        )


                                                                        
                    
                                                                        


def generate_labels(
    measurements: pd.DataFrame,
    ground_truth: pd.DataFrame,
    entity_type: str,
) -> pd.DataFrame:
    """
    Convert event-level ground truth into row-level labels.

    Parameters
    ----------
    measurements:
        Raw anomalous measurements.

    ground_truth:
        Event-level anomaly descriptions.

    entity_type:
        Either 'bus' or 'line'.

    Returns
    -------
    DataFrame containing one row for every measurement observation,
    with anomaly labels attached.
    """

    if entity_type not in {
        "bus",
        "line",
    }:
        raise ValueError(
            "entity_type must be 'bus' or 'line'."
        )

    data = measurements.copy()

                                                                    
               
                                                                    

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    if data["timestamp"].isna().any():

        raise ValueError(
            "Measurement data contains invalid timestamps."
        )

                                                                    
                   
                                                                    

    entity_column = (
        "bus_id"
        if entity_type == "bus"
        else "line_id"
    )

    if entity_column not in data.columns:

        raise ValueError(
            f"Measurement data missing '{entity_column}'."
        )

                                                                    
                       
                                                                    

    data["is_anomaly"] = 0

    data["anomaly_event_id"] = -1

    data["anomaly_type"] = "normal"

    data["anomaly_severity"] = 0.0

    data["affected_measurement"] = ""

                                                                    
                       
                                                                    

    relevant_events = ground_truth[
        ground_truth["entity_type"]
        == entity_type
    ]

    for _, event in relevant_events.iterrows():

        start = pd.Timestamp(
            event["start_timestamp"]
        )

        end = pd.Timestamp(
            event["end_timestamp"]
        )

        entity_id = int(
            event["entity_id"]
        )

        mask = (
            data[entity_column].eq(
                entity_id
            )
            & data["timestamp"].between(
                start,
                end,
                inclusive="both",
            )
        )

        data.loc[
            mask,
            "is_anomaly",
        ] = 1

        data.loc[
            mask,
            "anomaly_event_id",
        ] = int(
            event["event_id"]
        )

        data.loc[
            mask,
            "anomaly_type",
        ] = event[
            "anomaly_type"
        ]

        data.loc[
            mask,
            "anomaly_severity",
        ] = float(
            event["severity"]
        )

        data.loc[
            mask,
            "affected_measurement",
        ] = event[
            "affected_measurement"
        ]

    return data


                                                                        
               
                                                                        


def create_event_summary(
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create an event-level summary from row-level labels.
    """

    anomaly_rows = labels[
        labels["is_anomaly"] == 1
    ].copy()

    if anomaly_rows.empty:

        return pd.DataFrame(
            columns=[
                "anomaly_event_id",
                "anomaly_type",
                "entity_type",
                "entity_id",
                "start_timestamp",
                "end_timestamp",
                "severity",
                "affected_measurement",
                "number_of_rows",
            ]
        )

                                                   

    if "bus_id" in anomaly_rows.columns:

        entity_type = "bus"
        entity_column = "bus_id"

    else:

        entity_type = "line"
        entity_column = "line_id"

    grouped = []

    for event_id, group in anomaly_rows.groupby(
        "anomaly_event_id"
    ):

        grouped.append(
            {
                "anomaly_event_id": int(
                    event_id
                ),
                "anomaly_type": group[
                    "anomaly_type"
                ].iloc[0],
                "entity_type": entity_type,
                "entity_id": int(
                    group[
                        entity_column
                    ].iloc[0]
                ),
                "start_timestamp": group[
                    "timestamp"
                ].min(),
                "end_timestamp": group[
                    "timestamp"
                ].max(),
                "severity": group[
                    "anomaly_severity"
                ].iloc[0],
                "affected_measurement": group[
                    "affected_measurement"
                ].iloc[0],
                "number_of_rows": len(
                    group
                ),
            }
        )

    return pd.DataFrame(
        grouped
    )


                                                                        
      
                                                                        


def main() -> None:

    print()
    print("=" * 70)
    print(
        "GROUND-TRUTH LABEL GENERATION"
    )
    print("=" * 70)

                                                                    
          
                                                                    

    ground_truth = pd.read_csv(
        GROUND_TRUTH_PATH
    )

    bus_data = pd.read_csv(
        BUS_DATA_PATH,
        parse_dates=["timestamp"],
    )

    line_data = pd.read_csv(
        LINE_DATA_PATH,
        parse_dates=["timestamp"],
    )

    validate_ground_truth(
        ground_truth
    )

    print()
    print(
        f"Ground-truth events: "
        f"{len(ground_truth)}"
    )

    print(
        f"Bus observations: "
        f"{len(bus_data)}"
    )

    print(
        f"Line observations: "
        f"{len(line_data)}"
    )

                                                                    
                     
                                                                    

    bus_labels = generate_labels(
        bus_data,
        ground_truth,
        "bus",
    )

    line_labels = generate_labels(
        line_data,
        ground_truth,
        "line",
    )

                                                                    
                     
                                                                    

    bus_events = create_event_summary(
        bus_labels
    )

    line_events = create_event_summary(
        line_labels
    )

                                                                    
                             
                                                                    

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

                                                                    
          
                                                                    

    bus_output = (
        OUTPUT_DIR
        / "bus_labels.csv"
    )

    line_output = (
        OUTPUT_DIR
        / "line_labels.csv"
    )

    bus_event_output = (
        OUTPUT_DIR
        / "bus_event_summary.csv"
    )

    line_event_output = (
        OUTPUT_DIR
        / "line_event_summary.csv"
    )

    bus_labels.to_csv(
        bus_output,
        index=False,
    )

    line_labels.to_csv(
        line_output,
        index=False,
    )

    bus_events.to_csv(
        bus_event_output,
        index=False,
    )

    line_events.to_csv(
        line_event_output,
        index=False,
    )

                                                                    
            
                                                                    

    print()
    print(
        "-" * 70
    )
    print(
        "BUS LABELS"
    )
    print(
        "-" * 70
    )

    print(
        f"Total observations : "
        f"{len(bus_labels)}"
    )

    print(
        f"Anomalous rows     : "
        f"{bus_labels['is_anomaly'].sum()}"
    )

    print(
        f"Normal rows        : "
        f"{(bus_labels['is_anomaly'] == 0).sum()}"
    )

    print()
    print(
        bus_labels[
            bus_labels["is_anomaly"] == 1
        ][
            [
                "timestamp",
                "bus_id",
                "anomaly_event_id",
                "anomaly_type",
                "anomaly_severity",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "-" * 70
    )
    print(
        "LINE LABELS"
    )
    print(
        "-" * 70
    )

    print(
        f"Total observations : "
        f"{len(line_labels)}"
    )

    print(
        f"Anomalous rows     : "
        f"{line_labels['is_anomaly'].sum()}"
    )

    print(
        f"Normal rows        : "
        f"{(line_labels['is_anomaly'] == 0).sum()}"
    )

    print()
    print(
        line_labels[
            line_labels["is_anomaly"] == 1
        ][
            [
                "timestamp",
                "line_id",
                "anomaly_event_id",
                "anomaly_type",
                "anomaly_severity",
            ]
        ].to_string(
            index=False
        )
    )

                                                                    
                  
                                                                    

    print()
    print(
        "-" * 70
    )
    print(
        "SAVED LABELS"
    )
    print(
        "-" * 70
    )

    print(
        f"Bus labels:\n{bus_output}"
    )

    print(
        f"Line labels:\n{line_output}"
    )

    print(
        f"Bus events:\n{bus_event_output}"
    )

    print(
        f"Line events:\n{line_event_output}"
    )

    print()
    print("=" * 70)
    print(
        "GROUND-TRUTH LABEL GENERATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()