"""Tests for time series parsing and normalization."""

import json

from agent.timeseries import (
    detect_and_parse_timeseries,
    normalize_for_opentslm,
    parse_csv_timeseries,
    parse_json_timeseries,
)


def test_parse_csv_timeseries():
    """Test CSV parsing with numeric columns."""
    csv_data = b"""timestamp,heart_rate,blood_pressure
2024-01-01T10:00:00,72,120
2024-01-01T10:01:00,75,118
2024-01-01T10:02:00,73,122"""

    result = parse_csv_timeseries(csv_data)
    assert result is not None
    assert result["format"] == "csv"
    assert result["length"] == 3
    assert "heart_rate" in result["series"]
    assert result["series"]["heart_rate"] == [72.0, 75.0, 73.0]
    assert result["series"]["blood_pressure"] == [120.0, 118.0, 122.0]


def test_parse_csv_ignores_non_numeric():
    """Test that CSV parser skips non-numeric columns."""
    csv_data = b"""name,value
Alice,72
Bob,75
Charlie,73"""

    result = parse_csv_timeseries(csv_data)
    assert result is not None
    assert "value" in result["series"]
    assert "name" not in result["series"]


def test_parse_csv_returns_none_for_no_numeric_columns():
    """Test that CSV parser returns None when no numeric columns exist."""
    csv_data = b"""timestamp,event
2024-01-01,alarm
2024-01-02,alert"""

    result = parse_csv_timeseries(csv_data)
    assert result is None


def test_parse_csv_handles_invalid_data():
    """Test that CSV parser handles malformed data gracefully."""
    result = parse_csv_timeseries(b"not valid utf-8: \xFF\xFE")
    assert result is None

    result = parse_csv_timeseries(b"")
    assert result is None


def test_parse_json_array():
    """Test JSON parsing with simple numeric array."""
    json_data = b"[72, 75, 73, 76, 78]"
    result = parse_json_timeseries(json_data)
    assert result is not None
    assert result["format"] == "json"
    assert result["length"] == 5
    assert result["series"]["values"] == [72, 75, 73, 76, 78]


def test_parse_json_object_with_array():
    """Test JSON parsing with object containing numeric arrays."""
    json_data = json.dumps({
        "heart_rate": [72, 75, 73],
        "blood_pressure": [120, 118, 122]
    }).encode()

    result = parse_json_timeseries(json_data)
    assert result is not None
    assert result["format"] == "json"
    assert "heart_rate" in result["series"]
    assert result["series"]["heart_rate"] == [72, 75, 73]


def test_parse_json_returns_none_for_no_arrays():
    """Test that JSON parser returns None when no numeric arrays exist."""
    json_data = b'{"name": "test", "status": "ok"}'
    result = parse_json_timeseries(json_data)
    assert result is None


def test_detect_and_parse_csv_file():
    """Test auto-detection of CSV format."""
    csv_data = b"heart_rate,bp\n72,120\n75,118"
    result = detect_and_parse_timeseries("vitals.csv", csv_data)
    assert result is not None
    assert result["format"] == "csv"


def test_detect_and_parse_json_file():
    """Test auto-detection of JSON format."""
    json_data = b"[72, 75, 73]"
    result = detect_and_parse_timeseries("readings.json", json_data)
    assert result is not None
    assert result["format"] == "json"


def test_detect_and_parse_unsupported_format():
    """Test that unsupported formats return None."""
    result = detect_and_parse_timeseries("data.txt", b"some text")
    assert result is None


def test_normalize_csv_data():
    """Test normalization of parsed CSV data."""
    csv_data = b"heart_rate\n72\n75\n73\n76\n78"
    parsed = parse_csv_timeseries(csv_data)
    normalized = normalize_for_opentslm(parsed)

    assert normalized is not None
    assert normalized["label"] == "heart_rate"
    assert normalized["values"] == [72.0, 75.0, 73.0, 76.0, 78.0]
    assert normalized["metadata"]["count"] == 5
    assert normalized["metadata"]["mean"] == 74.8
    assert normalized["metadata"]["std"] > 0
    assert "sampling_rate" in normalized["metadata"]


def test_normalize_with_signal_name():
    """Test normalization with explicit signal name."""
    json_data = json.dumps({
        "heart_rate": [72, 75, 73],
        "blood_pressure": [120, 118, 122]
    }).encode()
    parsed = parse_json_timeseries(json_data)
    normalized = normalize_for_opentslm(parsed, signal_name="blood_pressure")

    assert normalized["label"] == "blood_pressure"
    assert normalized["values"] == [120, 118, 122]
    assert normalized["metadata"]["mean"] == 120.0


def test_normalize_single_value_no_stdev():
    """Test normalization with single value (no standard deviation)."""
    csv_data = b"value\n42"
    parsed = parse_csv_timeseries(csv_data)
    normalized = normalize_for_opentslm(parsed)

    assert normalized is not None
    assert normalized["values"] == [42.0]
    assert normalized["metadata"]["std"] == 0.0


def test_normalize_returns_none_for_invalid_input():
    """Test that normalization returns None for invalid data."""
    assert normalize_for_opentslm(None) is None
    assert normalize_for_opentslm({}) is None
    assert normalize_for_opentslm({"series": {}}) is None
    assert normalize_for_opentslm({"series": {"values": []}}) is None
