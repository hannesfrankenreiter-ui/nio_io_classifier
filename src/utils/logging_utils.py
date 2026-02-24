"""
src/utils/logging_utils.py
Erstellt einen Logger der gleichzeitig auf Konsole und in eine Datei schreibt.
"""
import logging
import sys


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Gibt einen konfigurierten Logger zurück.
    - Immer: StreamHandler (stdout)
    - Optional: FileHandler (log_file)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Doppelte Handler verhindern (z.B. bei mehrfachem Aufruf in Tests)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger