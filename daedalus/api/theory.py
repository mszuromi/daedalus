"""Deprecated shim — this module was renamed to :mod:`api.model` (July 2026,
theory -> model rename).  Import from ``api.model`` in new code."""
from api.model import *                                    # noqa: F401,F403
from api.model import (ModelBuilder, TemporalModelBuilder,  # noqa: F401
                       SpatialModelBuilder, TheoryBuilder,
                       TemporalTheoryBuilder, SpatialTheoryBuilder)
