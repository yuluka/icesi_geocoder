import logging
import os
from typing import Dict, Optional

from config import REPO_ROOT

"""
Define the logging configuration for the project. 

This is necessary to manage the project logs. These are saved in a file with the specified name.
"""

CUSTOM_LEVELS: Dict[str, int] = {
    "DATA-QUALITY": 31,
}


def register_custom_levels() -> None:
    """
    Add custom levels to the logger.

    :return: None
    """

    for level_name, level_value in CUSTOM_LEVELS.items():
        logging.addLevelName(level_value, level_name)

        def log_for_level(self, message, *args, _level=level_value, **kwargs):
            if self.isEnabledFor(_level):
                self._log(_level, message, args, **kwargs)

        method_name = level_name.lower().replace("-", "_")

        setattr(logging.Logger, method_name, log_for_level)


def resolve_log_path(logs_file: Optional[str] = None) -> str:
    """
    Resolve the log file path, using settings.LOGS_PATH if not explicitly provided.
    Always returns a normalized absolute path.
    """

    if logs_file and logs_file.strip():
        path_str = logs_file.strip()

    else:
        try:
            from config import settings, REPO_ROOT
            path_str = (settings.LOGS_PATH or "").strip()

            if not path_str:
                path_str = str(REPO_ROOT / "logs" / "general_logs.log")

            elif not os.path.isabs(path_str):
                path_str = str((REPO_ROOT / path_str.replace("\\", "/").lstrip("./")).resolve())

        except Exception:
            path_str = os.path.abspath(os.path.join("logs", "general_logs.log"))

    if not os.path.isabs(path_str):
        try:
            path_str = str((REPO_ROOT / path_str.replace("\\", "/").lstrip("./")).resolve())

        except Exception:
            path_str = os.path.abspath(path_str)

    return os.path.normpath(path_str)


def create_log(logs_file: Optional[str] = None) -> None:
    """
    Create a log file to store logs.

    :param logs_file: Path to the log file. If None, resolves from settings.LOGS_PATH.
    :type logs_file: Optional[str]
    :return: None
    """

    register_custom_levels()

    resolved_path: str = resolve_log_path(logs_file)
    log_dir: str = os.path.dirname(resolved_path)

    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Check existing handlers to avoid duplicate handlers
    target_abs = os.path.abspath(resolved_path)
    has_file_handler = False

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if os.path.abspath(handler.baseFilename) == target_abs:
                    has_file_handler = True
                    break

            except Exception:
                pass

    if not has_file_handler:
        file_handler = logging.FileHandler(resolved_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    has_stream_handler = False

    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            has_stream_handler = True
            handler.setFormatter(formatter)

            break

    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)


def close_log_file_handlers() -> None:
    """
    Close all file handlers in the logging module.

    :return: None
    """

    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logging.root.removeHandler(handler)


# Register custom levels on import so they are immediately available
register_custom_levels()
