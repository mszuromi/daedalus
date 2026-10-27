"""daedalus._resources -- paths of the data files shipped inside the package
(the model files, the shipped prediagram cells and their manifests), found
with ``importlib.resources`` so they resolve the same in a source checkout
and in an installed wheel.  The package is installed unpacked
(``zip-safe = false``), so every path is a real directory on disk."""
import importlib.resources
import os


def data_path(*parts: str) -> str:
    """``daedalus/<parts...>`` as an absolute, normalised filesystem path
    (the package may have been imported through ``<repo>/tests/..``)."""
    return os.path.abspath(os.fspath(importlib.resources.files('daedalus').joinpath(*parts)))
