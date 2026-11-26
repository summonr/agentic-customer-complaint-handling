"""Parse and normalize time series data from attachments."""

import csv
import json
import statistics
from io import StringIO
from typing import Any


class TimeSeriesParseError(Exception):
    """Raised when time series parsing fails."""

    pass


def parse_csv_timeseries(data: bytes) -> dict[str, Any] | None:
    """Parse CSV data as time series. Returns dict with numeric columns or None."""
    try:
        text = data.decode("utf-8")
        reader = csv.DictReader(StringIO(text))
        rows = list(reader)
        if not rows:
            return None

        numeric_cols = {}
        for key in rows[0].keys():
            try:
                values = [float(row[key]) for row in rows]
                numeric_cols[key] = values
            except (ValueError, KeyError, TypeError):
                pass

        return {"format": "csv", "series": numeric_cols, "length": len(rows)} if numeric_cols else None
    except Exception:
        return None


def parse_json_timeseries(data: bytes) -> dict[str, Any] | None:
    """Parse JSON data as time series. Handles arrays or objects with numeric arrays."""
    try:
        obj = json.loads(data.decode("utf-8"))

        if isinstance(obj, list) and obj and isinstance(obj[0], (int, float)):
            return {"format": "json", "series": {"values": obj}, "length": len(obj)}

        if isinstance(obj, dict):
            numeric_arrays = {}
            for key, val in obj.items():
                if isinstance(val, list) and val and isinstance(val[0], (int, float)):
                    numeric_arrays[key] = val

            return {"format": "json", "series": numeric_arrays, "length": len(next(iter(numeric_arrays.values())))} if numeric_arrays else None

        return None
    except Exception:
        return None


def detect_and_parse_timeseries(filename: str, data: bytes) -> dict[str, Any] | None:
    """Auto-detect format from filename and parse. Returns normalized dict or None."""
    if not filename or not data:
        return None

    lower_name = filename.lower()

    if lower_name.endswith(".csv"):
        return parse_csv_timeseries(data)
    elif lower_name.endswith(".json"):
        return parse_json_timeseries(data)

    return None


def normalize_for_opentslm(ts_data: dict[str, Any], signal_name: str | None = None) -> dict[str, Any] | None:
    """
    Convert parsed time series to OpenTSLM input format.

    Returns dict with 'label', 'values', and 'metadata', or None if invalid.
    """
    if not ts_data or not ts_data.get("series"):
        return None

    series_dict = ts_data["series"]
    if not series_dict:
        return None

    key = signal_name or next(iter(series_dict.keys()), None)
    if not key:
        return None

    values = series_dict[key]
    if not values:
        return None

    try:
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    except statistics.StatisticsError:
        return None

    return {
        "label": key,
        "values": values,
        "metadata": {
            "count": len(values),
            "mean": round(mean, 2),
            "std": round(stdev, 2),
            "duration": f"{len(values)} samples",
            "sampling_rate": 1.0,
        },
    }
