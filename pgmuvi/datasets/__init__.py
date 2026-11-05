"""Small datasets bundled with pgmuvi for tests, examples and documentation.

Larger datasets (roadmap 3.12) will be fetched on demand rather than shipped.
"""

from pathlib import Path

__all__ = ["DATA_DIR", "get_path"]

DATA_DIR = Path(__file__).resolve().parent


def get_path(name: str) -> Path:
    """Return the path of a bundled data file.

    Parameters
    ----------
    name : str
        File name inside ``pgmuvi/datasets``, for example
        ``"AlfOriAAVSO_Vband.csv"``.

    Returns
    -------
    pathlib.Path
        Absolute path to the file.

    Raises
    ------
    FileNotFoundError
        If no bundled file has that name.
    """
    path = DATA_DIR / name
    if not path.is_file():
        available = sorted(p.name for p in DATA_DIR.iterdir() if p.suffix == ".csv")
        raise FileNotFoundError(
            f"No bundled dataset named {name!r}; available: {available}"
        )
    return path
