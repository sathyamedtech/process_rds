#!/usr/bin/env python3
"""
process_rds.py

Fetch raw device records from an RDS SQL Server table, run preprocessing + feature extraction,
and write processed results back into RDS.

Assumptions about raw table (device_raw):
- id              : primary key (int)
- patient_id      : int
- device_id       : varchar
- visit           : varchar / int
- game_id         : varchar / int
- samp_freq       : int
- channel_names   : varchar (comma separated) OR NULL
- channel_values  : nvarchar(max) OR varbinary(max) -- stores the signal data
                   * Supported formats:
                     - JSON string: [[ch1_samples...], [ch2_samples...], ...]
                     - base64-encoded pickled numpy array (ndarray of shape (n_channels, n_times))
- processed       : bit / int (0 or 1)
- created_at      : datetime

Note: adapt the column names in SQL queries below if your schema differs.
"""

import json
import base64
import pickle
import logging
import sys
import numpy as np
from typing import Tuple, Optional, Any, List

import mne
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine
from sqlalchemy.sql import text as sql_text

from feature_extraction import Feature_Extraction
from logger_config import logging as logger  # your existing logger_config.py
from auto_config import RDS_SQLALCHEMY_URI, SHOW_SQL, MNE_LOG_LEVEL

# Make sure MNE logs follow config
mne.set_log_level(MNE_LOG_LEVEL)

# Configure local logging (logger_config sets file handler); add console handler for convenience.
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# Processing settings
BATCH_SIZE = 10  # number of raw rows to fetch at a time (tune for memory)
RAW_TABLE = "Device_Data"
PROCESSED_TABLE = "device_processed"
COMPLEXITY_TABLE = "complexity"
POWER_RATIO_TABLE = "power_ratios"
COHERENCE_TABLE = "coherence"

def create_db_engine() -> Engine:
    try:
        engine = create_engine(RDS_SQLALCHEMY_URI, pool_pre_ping=True)
        if SHOW_SQL:
            logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
        return engine
    except Exception as e:
        logging.critical("Failed to create database engine: %s", e)
        raise

def _decode_channel_values(raw_value: Any) -> Optional[np.ndarray]:
    """
    Try multiple ways to decode channel_values field into a numpy ndarray shaped (n_channels, n_times).
    Supported encodings:
    - JSON string: list of lists -> np.ndarray
    - bytes (pickled ndarray) base64-encoded -> decode -> pickle.loads -> np.ndarray
    - raw bytes pickled (if client stored varbinary directly)
    """
    if raw_value is None:
        return None
    # If already a bytes-like object (SQLAlchemy may return bytes for varbinary)
    try:
        if isinstance(raw_value, (bytes, bytearray)):
            # Try to unpickle directly
            try:
                arr = pickle.loads(raw_value)
                return np.asarray(arr)
            except Exception:
                # Maybe it's base64 encoded bytes -> decode then unpickle
                try:
                    decoded = base64.b64decode(raw_value)
                    arr = pickle.loads(decoded)
                    return np.asarray(arr)
                except Exception:
                    # give up
                    logging.debug("Failed to decode raw bytes for channel_values")
                    return None

        # If string: try JSON then base64+pickle
        if isinstance(raw_value, str):
            # JSON first
            try:
                parsed = json.loads(raw_value)
                arr = np.asarray(parsed, dtype=float)
                if arr.ndim == 1:
                    # maybe flattened - try to reshape later
                    return arr
                return arr
            except Exception:
                # Try as base64 pickled
                try:
                    decoded = base64.b64decode(raw_value)
                    arr = pickle.loads(decoded)
                    return np.asarray(arr)
                except Exception as e:
                    logging.debug("Failed to decode string channel_values: %s", e)
                    return None
    except Exception as e:
        logging.exception("Unexpected error when decoding channel_values: %s", e)
        return None

    return None

def assemble_raw_array(channel_names: Optional[List[str]], channel_values: np.ndarray, samp_freq: int) -> mne.io.Raw:
    """
    Convert channel_values and channel_names into an mne RawArray (n_channels, n_times).
    If channel_names is None, generate default names 'CH1','CH2',...
    """
    if channel_values is None:
        raise ValueError("channel_values is None; cannot assemble RawArray")

    arr = np.asarray(channel_values)
    # If arr is 1D, assume single-channel; make shape (1, n)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    # If arr has shape (n_times, n_channels) but we expect (n_channels, n_times), try to detect:
    # Heuristic: if first dimension length matches typical sampling frequencies (>= 100) and second is <= 64, it might be (n_times, n_channels)
    if arr.shape[0] > arr.shape[1] and arr.shape[0] > 64:
        # Common orientation is (n_channels, n_times) but some clients store transposed. Compare to samp_freq to guide decision.
        # If arr.shape[0] corresponds to time samples (>> channels), transpose.
        logging.debug("Detected possible transposed array; transposing to (n_channels, n_times)")
        arr = arr.T

    n_channels = arr.shape[0]
    ch_names = channel_names if channel_names else [f"CH{i+1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=float(samp_freq), ch_types='eeg')
    raw = mne.io.RawArray(arr, info)
    return raw

def fetch_unprocessed_rows(engine: Engine, limit: int = BATCH_SIZE):
    """
    Fetch unprocessed rows from RAW_TABLE.
    Returns list of dict rows (SQLAlchemy result proxies).
    """
    query = sql_text(f"""
        SELECT TOP (:limit) *
        FROM {RAW_TABLE}
        WHERE processed = 0 OR processed IS NULL
        ORDER BY created_at ASC
    """)  # TOP is SQL Server-specific
    with engine.connect() as conn:
        res = conn.execute(query, {"limit": limit})
        rows = [dict(row) for row in res]
    return rows

def mark_raw_processed(engine: Engine, raw_id: int):
    stmt = sql_text(f"UPDATE {RAW_TABLE} SET processed = 1 WHERE id = :id")
    with engine.begin() as conn:
        conn.execute(stmt, {"id": raw_id})

def insert_complexity(engine: Engine, conn, raw_path: str, channel_complexities: dict, patient_id, visit, game_id):
    # channel_complexities: dict(channel_name -> np.array)
    insert_stmt = sql_text(f"""
        INSERT INTO {COMPLEXITY_TABLE} (raw_id, channel, scale_idx, scale_value, patient_id, visit, game_id)
        VALUES (:raw_id, :channel, :scale_idx, :scale_value, :patient_id, :visit, :game_id)
    """)
    # flatten scales per channel
    for ch, arr in channel_complexities.items():
        arr = np.asarray(arr)
        for idx, val in enumerate(arr.tolist()):
            conn.execute(insert_stmt, {
                "raw_id": raw_path, "channel": ch, "scale_idx": idx, "scale_value": float(val),
                "patient_id": patient_id, "visit": visit, "game_id": game_id
            })

def insert_power_ratios(engine: Engine, conn, raw_path: str, channel_ratios: dict, patient_id, visit, game_id, combined_ratio):
    insert_stmt = sql_text(f"""
        INSERT INTO {POWER_RATIO_TABLE} (raw_id, channel, power_ratio, combined_ratio, patient_id, visit, game_id)
        VALUES (:raw_id, :channel, :power_ratio, :combined_ratio, :patient_id, :visit, :game_id)
    """)
    for ch, val in channel_ratios.items():
        conn.execute(insert_stmt, {
            "raw_id": raw_path, "channel": ch, "power_ratio": float(val),
            "combined_ratio": float(combined_ratio),
            "patient_id": patient_id, "visit": visit, "game_id": game_id
        })

def insert_coherence(engine: Engine, conn, raw_path: str, coh_mean2: list, patient_id, visit, game_id):
    # coh_mean2 is expected as list of tuples (pair_name, mean_value)
    insert_stmt = sql_text(f"""
        INSERT INTO {COHERENCE_TABLE} (raw_id, channel_pair, coherence, patient_id, visit, game_id)
        VALUES (:raw_id, :channel_pair, :coherence, :patient_id, :visit, :game_id)
    """)
    for pair_name, mean_val in coh_mean2:
        conn.execute(insert_stmt, {
            "raw_id": raw_path, "channel_pair": pair_name, "coherence": float(mean_val),
            "patient_id": patient_id, "visit": visit, "game_id": game_id
        })

def process_row(engine: Engine, row: dict, feature_extractor: Feature_Extraction):
    """
    Process a single raw row and store results.
    row keys depend on table; expected keys: id, patient_id, device_id, visit, game_id, samp_freq,
    channel_names, channel_values
    """
    raw_id = row.get("id")
    patient_id = row.get("patient_id")
    device_id = row.get("device_id")
    visit = row.get("visit")
    game_id = row.get("game_id")
    samp_freq = int(row.get("samp_freq") or 250)

    logging.info(f"Processing row id={raw_id}, patient={patient_id}, device={device_id}, visit={visit}, game={game_id}")

    # Decode channel_values
    channel_values_field = row.get("channel_values")
    channel_values = _decode_channel_values(channel_values_field)
    if channel_values is None:
        logging.error("Could not decode channel_values for id=%s. Skipping.", raw_id)
        return False

    # Get channel names if present
    ch_names_field = row.get("channel_names")
    ch_names = None
    if ch_names_field:
        if isinstance(ch_names_field, str):
            ch_names = [c.strip() for c in ch_names_field.split(",") if c.strip()]
        elif isinstance(ch_names_field, (list, tuple)):
            ch_names = list(ch_names_field)

    try:
        raw = assemble_raw_array(ch_names, channel_values, samp_freq)
    except Exception as e:
        logging.exception("Failed to construct RawArray for id=%s: %s", raw_id, e)
        return False

    try:
        # Use existing feature extraction workflow but adapted for raw object inputs
        # Downsample and set montage as the existing code expects
        raw_downsampled = feature_extractor.downsample(raw, feature_extractor_downsample_rate(), feature_extractor_montage())
        events, event_id, raw_downsampled = feature_extractor.create_events(raw_downsampled)
        if events is None or len(events) == 0:
            logging.error("No events created for id=%s. Skipping.", raw_id)
            return False
        raw_data, data, epochs = feature_extractor.create_epochs(raw_downsampled, events, event_id)
        channel_ratios, combined_ratio, psd_values = feature_extractor.power_ratio(epochs)
        _, _, _, _, _, _, _, coh_mean2 = feature_extractor.full_coherence(raw_data)
        channel_complexities = feature_extractor.complexity(data, fuzzy_embedding=2)

    except Exception as e:
        logging.exception("Feature extraction failed for id=%s: %s", raw_id, e)
        return False

    # Insert results into RDS tables
    try:
        with engine.begin() as conn:
            # Insert a main processed record (device_processed)
            insert_main = sql_text(f"""
                INSERT INTO {PROCESSED_TABLE} (
                    raw_id, patient_id, device_id, visit, game_id, samp_freq, processed_at
                ) VALUES (
                    :raw_id, :patient_id, :device_id, :visit, :game_id, :samp_freq, GETUTCDATE()
                )
            """)
            conn.execute(insert_main, {
                "raw_id": raw_id, "patient_id": patient_id, "device_id": device_id,
                "visit": visit, "game_id": game_id, "samp_freq": samp_freq
            })

            # Insert subtables
            insert_complexity(engine, conn, raw_id, channel_complexities, patient_id, visit, game_id)
            insert_power_ratios(engine, conn, raw_id, channel_ratios, patient_id, visit, game_id, combined_ratio)
            insert_coherence(engine, conn, raw_id, coh_mean2, patient_id, visit, game_id)
            # Mark raw processed
            mark_raw_processed(engine, raw_id)
            logging.info("Finished inserting features for id=%s", raw_id)
    except SQLAlchemyError as e:
        logging.exception("Database insertion failed for id=%s: %s", raw_id, e)
        return False
    except Exception as e:
        logging.exception("Unexpected error during DB insert for id=%s: %s", raw_id, e)
        return False

    return True

# Helpers to read expected values from auto_config or defaults
from auto_config import OUTPUT_FREQ, MONTAGE
def feature_extractor_downsample_rate() -> int:
    return int(globals().get("OUTPUT_FREQ", OUTPUT_FREQ if 'OUTPUT_FREQ' in globals() else 250))

def feature_extractor_montage() -> str:
    return globals().get("MONTAGE", MONTAGE if 'MONTAGE' in globals() else "standard_alphabetic")

def main():
    engine = create_db_engine()
    fe = Feature_Extraction()

    while True:
        rows = fetch_unprocessed_rows(engine, limit=BATCH_SIZE)
        if not rows:
            logging.info("No unprocessed rows found. Exiting.")
            break

        for row in rows:
            try:
                success = process_row(engine, row, fe)
                if not success:
                    logging.error("Processing failed for row id=%s; continuing with next.", row.get("id"))
            except Exception as e:
                logging.exception("Unhandled exception processing id=%s: %s", row.get("id"), e)
                # continue to next row

if __name__ == "__main__":
    main()
