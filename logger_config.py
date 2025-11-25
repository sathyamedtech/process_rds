import logging
from auto_config import (LOG_LEVEL, LOG_FILE)

log_level_value = getattr(logging, LOG_LEVEL, logging.INFO)


# Remove all handlers associated with the root logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=log_level_value,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
    ]
)
