"""
ml_service/src/utils/memory_guard.py
─────────────────────────────────────
Enforces the 8 GB RAM hard constraint on the development machine.

Two public functions:

    assert_memory_headroom(min_free_mb)
        Raises MemoryError if less than `min_free_mb` MB of RAM is
        available. Call this before loading a large CSV or starting
        model training.

    safe_read_csv(path, usecols, chunksize, ...)
        Chunked CSV reader that streams the file in manageable pieces
        rather than loading it all at once.  Returns a single
        concatenated DataFrame if collect=True (safe for files that
        fit comfortably after filtering), or a generator of chunks if
        collect=False (for very large files or streaming pipelines).

Usage
-----
    from src.utils.memory_guard import assert_memory_headroom, safe_read_csv

    assert_memory_headroom(min_free_mb=1024)   # need >= 1 GB free
    df = safe_read_csv(
        Paths.TUESDAY_CSV,
        usecols=FEATURE_COLS + ["Label"],
        dtype={"Destination Port": "int32"},
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Sequence

import pandas as pd
import psutil

logger = logging.getLogger(__name__)

# Default chunk size: 50 000 rows ≈ ~25 MB per chunk for 80 float64 columns.
_DEFAULT_CHUNKSIZE = 50_000

# Minimum free RAM required before we start loading data.
_DEFAULT_MIN_FREE_MB = 1_024  # 1 GB


def assert_memory_headroom(min_free_mb: int = _DEFAULT_MIN_FREE_MB) -> None:
    """Raise MemoryError if available RAM is below `min_free_mb` megabytes.

    Call this at the top of any function that is about to load a large
    DataFrame or kick off model training.

    Parameters
    ----------
    min_free_mb:
        Minimum free RAM in MB required to proceed. Default 1024 (1 GB).

    Raises
    ------
    MemoryError
        If available RAM is below the threshold.
    """
    available_mb = psutil.virtual_memory().available / (1024 ** 2)
    logger.debug("Available RAM: %.0f MB (threshold: %d MB)", available_mb, min_free_mb)
    if available_mb < min_free_mb:
        raise MemoryError(
            f"Insufficient RAM: {available_mb:.0f} MB available, "
            f"but {min_free_mb} MB required. "
            "Free memory before loading data or reduce chunk size."
        )


def safe_read_csv(
    path: str | Path,
    usecols: Sequence[str] | None = None,
    chunksize: int = _DEFAULT_CHUNKSIZE,
    dtype: dict | None = None,
    collect: bool = True,
    min_free_mb: int = _DEFAULT_MIN_FREE_MB,
    **kwargs,
) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
    """Read a CSV file in RAM-safe chunks.

    Parameters
    ----------
    path:
        Path to the CSV file.
    usecols:
        Columns to load. Always specify this to avoid reading unused columns.
        Passing None loads all columns (use with caution on large files).
    chunksize:
        Number of rows per chunk. Default 50 000.
    dtype:
        Optional dict of column → dtype overrides (e.g. {"Port": "int32"}).
        Use narrow dtypes to reduce memory footprint.
    collect:
        If True (default), concatenates all chunks into a single DataFrame and
        returns it. Use when the filtered data fits comfortably in RAM.
        If False, returns a generator yielding one chunk at a time — use for
        very large files or streaming aggregations.
    min_free_mb:
        Minimum free RAM required before starting to read. Default 1024 MB.
    **kwargs:
        Additional keyword arguments forwarded to pandas.read_csv().

    Returns
    -------
    pd.DataFrame
        If collect=True.
    Generator[pd.DataFrame, None, None]
        If collect=False.

    Notes
    -----
    - Column names in CIC-IDS2017 have leading whitespace. This function
      strips them automatically via `skipinitialspace=True` and a post-strip
      rename step so callers can use clean names like "Destination Port".
    - `Infinity` values are replaced with NaN so downstream callers can
      handle them consistently with pandas NA tools.
    """
    assert_memory_headroom(min_free_mb=min_free_mb)

    path = Path(path)
    logger.info("Reading %s (chunksize=%d, usecols=%s)", path.name, chunksize, usecols)

    reader = pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunksize,
        dtype=dtype,
        skipinitialspace=True,   # strips leading whitespace from column names
        on_bad_lines="warn",     # log bad rows instead of crashing
        **kwargs,
    )

    def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
        # Strip any residual whitespace from column names.
        chunk.columns = chunk.columns.str.strip()
        # Replace Infinity values (common in CIC-IDS2017 byte-rate cols).
        chunk.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
        return chunk

    if not collect:
        return (_clean_chunk(chunk) for chunk in reader)

    chunks = [_clean_chunk(chunk) for chunk in reader]
    if not chunks:
        logger.warning("No data read from %s", path)
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    logger.info(
        "Loaded %s: %d rows × %d cols (%.1f MB in memory)",
        path.name,
        len(df),
        len(df.columns),
        df.memory_usage(deep=True).sum() / (1024 ** 2),
    )
    return df
