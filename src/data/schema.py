"""
Data schemas for the Power Grid Anomaly Detection pipeline.

This module defines the canonical representation of:
    - Measurement records
    - Anomaly/event records
    - Dataset metadata

The schemas are deliberately independent of any particular data source
(SCADA, PMU, DER, or simulation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class DataSource(str, Enum):
    """Origin of a measurement dataset."""

    SCADA = "scada"
    PMU = "pmu"
    DER = "der"
    SMART_METER = "smart_meter"
    SIMULATION = "simulation"


class AnomalyCategory(str, Enum):
    """High-level category of an abnormal event."""

    MEASUREMENT = "measurement"
    PHYSICAL = "physical"
    TOPOLOGICAL = "topological"
    CYBER = "cyber"


class AnomalyType(str, Enum):
    """Specific anomaly/event type."""

    NONE = "none"

                           
    SPIKE = "spike"
    BIAS = "bias"
    DRIFT = "drift"
    DROPOUT = "dropout"
    NOISE = "noise"
    DELAY = "delay"

                             
    LOAD_DISTURBANCE = "load_disturbance"
    DER_FLUCTUATION = "der_fluctuation"
    VOLTAGE_VIOLATION = "voltage_violation"
    LINE_OVERLOAD = "line_overload"
    GENERATOR_DISTURBANCE = "generator_disturbance"

                           
    LINE_OUTAGE = "line_outage"
    SWITCHING_EVENT = "switching_event"
    ISLANDING = "islanding"


@dataclass
class MeasurementRecord:
    """
    Canonical representation of a power-system measurement.

    Not every measurement source will provide every field.
    Missing values are therefore represented using None and handled
    later by the preprocessing pipeline.
    """

    timestamp: datetime

    bus_id: Optional[int] = None
    line_id: Optional[int] = None

                             
    voltage_pu: Optional[float] = None
    angle_deg: Optional[float] = None

    current_ka: Optional[float] = None

    active_power_mw: Optional[float] = None
    reactive_power_mvar: Optional[float] = None

    apparent_power_mva: Optional[float] = None
    power_factor: Optional[float] = None

    frequency_hz: Optional[float] = None

    line_loading_pct: Optional[float] = None

                             
    der_generation_mw: Optional[float] = None

                     
    source: DataSource = DataSource.SIMULATION

                                
    quality: Optional[float] = None


@dataclass
class EventRecord:
    """
    Ground-truth description of an anomaly or grid event.

    This will become particularly important when we evaluate
    unsupervised anomaly detectors against known injected events.
    """

    event_id: str

    start_time: datetime
    end_time: datetime

    anomaly_category: AnomalyCategory
    anomaly_type: AnomalyType

    severity: float = 0.0

    affected_bus: Optional[int] = None
    affected_line: Optional[int] = None

    description: Optional[str] = None

    metadata: dict = field(default_factory=dict)


@dataclass
class DatasetMetadata:
    """
    Metadata describing a generated or imported dataset.
    """

    name: str
    source: DataSource

    network_name: Optional[str] = None

    sampling_interval_seconds: Optional[int] = None

    num_buses: Optional[int] = None
    num_lines: Optional[int] = None

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    contains_anomalies: bool = False

    description: Optional[str] = None

    metadata: dict = field(default_factory=dict)