from functools import cache
from importlib.metadata import version


@cache
def get_version() -> str:
    """
    extract the package version using importlib.
    """
    return version("splendor")
