"""Deprecated shim — this module was renamed to :mod:`daedalus.api.model` (July 2026,
theory -> model rename).  Import from ``daedalus.api.model`` in new code."""
from daedalus.api.model import *                                    # noqa: F401,F403
from daedalus.api.model import (ModelBuilder, TemporalModelBuilder,  # noqa: F401
                       SpatialModelBuilder, TheoryBuilder,
                       TemporalTheoryBuilder, SpatialTheoryBuilder)
