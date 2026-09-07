import logging

from logger.logger_config import create_log, register_custom_levels

## ----------- Logging -----------

create_log()

logger: logging.Logger = logging.getLogger(__name__)