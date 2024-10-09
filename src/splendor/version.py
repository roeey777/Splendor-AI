import importlib

from functools import cache


@cache
def get_version() -> str:
    """
    extract the package version using importlib.
    """
    return importlib.metadata.version("splendor")
