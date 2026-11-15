from __future__ import annotations

from pathlib import Path

import pandas as pd

CPU_READINGS_DTYPES = {
    "timestamp": "int64",
    "vm_id": "str",
    "min_cpu": "float64",
    "max_cpu": "float64",
    "avg_cpu": "float64",
}

VM_METADATA_DTYPES = {
    "vm_id": "str",
    "subscription_id": "str",
    "deployment_id": "str",
    "timestamp_vm_created": "int64",
    "timestamp_vm_deleted": "int64",
    "max_cpu": "float64",
    "avg_cpu": "float64",
    "p95_max_cpu": "float64",
    "vm_category": "str",
    "virtual_core_bucket": "str",
    "memory_gb_bucket": "str",
}


def load_cpu_readings(data_dir: str | Path) -> pd.DataFrame:
    """Load vm_cpu_readings.csv.gz with explicit dtypes.

    `timestamp` is left as the raw integer seconds-relative-to-trace value
    given in the source file; it is never converted to a datetime.
    """
    path = Path(data_dir) / "vm_cpu_readings.csv.gz"
    return pd.read_csv(path, dtype=CPU_READINGS_DTYPES)


def load_vm_metadata(data_dir: str | Path) -> pd.DataFrame:
    """Load vm_metadata.csv.gz with explicit dtypes."""
    path = Path(data_dir) / "vm_metadata.csv.gz"
    return pd.read_csv(path, dtype=VM_METADATA_DTYPES)
