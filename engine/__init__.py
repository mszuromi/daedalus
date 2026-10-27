"""Old top-level name of ``daedalus.engine`` (deprecated): forwards to it.  Kept in the
source checkout only, for one release; not part of the installed package."""
from daedalus import _compat

_compat.forward(__name__)
