from importlib.metadata import version

from functools import cache


@cache
def get_version() -> str:
    """
    extract the package version using importlib.
    """
    return version("splendor")
