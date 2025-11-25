# ===========================
# AUTO CONFIG FOR RDS WORKFLOW
# ===========================

# -------------------------------------
# EEG CHANNEL LIST (update as required)
# -------------------------------------
CHANNELS = [
    "C3", "C4", "F3", "F4", "P3", "P4",
    "O1", "O2", "T3", "T4", "F7", "F8", "CZ"
]

# -----------------------------------------
# EEG Frequency Bands for Feature Extraction
# -----------------------------------------
DEFINED_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45)
}

# -------------------------------------------------------
# Event window duration (in seconds) for EEG processing
# -------------------------------------------------------
EVENT_DURATION = 2  # Example: 2-second processing window

# -------------------------------------------------------------------
# Filter constraints to avoid edge artifacts in the EEG time-series
# -------------------------------------------------------------------
FILTER_MIN_DISTANCE_FROM_START = 0.2   # seconds
FILTER_MIN_DISTANCE_FROM_END = 0.2     # seconds

# ------------------------------------------------------------
# Sampling rate of your device stream (UPDATE IF DIFFERENT)
# ------------------------------------------------------------
SAMPLING_RATE = 256  # Hz

# Output frequency (Hz) after processing
OUTPUT_FREQ = 256

# Montage name (kept only for compatibility with existing code)
MONTAGE = "standard_1020"

# -----------------------------------------
# FFT Configuration
# -----------------------------------------
N_FFT = 256   # Number of points for FFT
N_OVERLAP = N_FFT // 2
# --------------------------------------------------------------
# RDS TABLES — update names based on your actual SQL Server schema
# --------------------------------------------------------------
RDS_SOURCE_TABLE = "raw_eeg_data"          # Table from where device data is read
RDS_PROCESSED_TABLE = "processed_eeg_data" # Table where results will be stored

# -----------------------------------------------------------------------
# SQL Server uses ODBC Driver 18 on Ubuntu (install if you haven't):
# sudo apt-get install msodbcsql18 mssql-tools18 unixodbc-dev
# -----------------------------------------------------------------------
RDS_CONNECTION_STRING = (
    "mssql+pyodbc://darshtp:darshtp_boja@device-data.c7y0avuyjpgj.eu-west-2.rds.amazonaws.com:1433/DeviceManagement"
    "?driver=ODBC+Driver+18+for+SQL+Server"
    "&Encrypt=yes&TrustServerCertificate=yes"
)

# ------------------------------------------------
# Logging configuration
# ------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FILE = "process_rds.log"

# ------------------------------------------------
# Additional settings required by process_rds.py
# ------------------------------------------------
RDS_SQLALCHEMY_URI = RDS_CONNECTION_STRING
SHOW_SQL = False
MNE_LOG_LEVEL = "WARNING"
