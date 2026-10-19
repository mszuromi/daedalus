"""
engine._warn_helpers
====================
Warning helpers shared by the engine and ``api`` (one copy:
``api.save`` and ``final_integral`` each had a byte-identical copy).  Both
modules re-export :func:`ensure_warning_shown` as ``_ensure_warning_shown``.
"""
import re
import sys
import warnings


def ensure_warning_shown(category, action='always', message=None,
                         module=None, stacklevel=None):
    """Show ``category`` also where the ENVIRONMENT ignores warnings: the
    Sage launcher exports PYTHONWARNINGS=ignore, which hides every warning in
    ``sage -python`` and the Sage Jupyter kernel.  Only that case is
    overridden: a filter for this category, and every filter set in code or
    by an ``error`` option (e.g. ``-W error``), is left alone -- a filter is
    added only when the first filter that applies to the warning is the
    environment's own ignore-all.  The action is 'always': callers
    deduplicate their own warnings (a 'default' filter would also swallow
    repeats that a caller's own 'always' filter asked for).

    ``message`` (the text about to be warned) and ``module`` (the module the
    warning is attributed to; or ``stacklevel`` as in ``warnings.warn``,
    counted from the caller) let the helper skip filters aimed at OTHER
    warnings: Sage installs message-specific ``ignore`` filters for numpy at
    startup, and without the message such a filter was taken for a user's
    decision and hid the warning.  A filter whose message / module pattern
    matches, or any category-wide ``ignore``, is still respected; with the
    message (module) unknown, message- (module-) specific filters are
    respected."""
    opts = [o.split(':')[0].strip().lower() for o in sys.warnoptions]
    if not (opts and opts[-1] in ('ignore', 'i')):
        return
    filters = warnings.filters
    # the environment's ignore-all is the LAST unconditional ignore (startup
    # puts it behind every filter set later in code).  A plain
    # simplefilter('ignore') set in code IS that same filter (Python merges
    # identical filters), so it cannot be told apart and is overridden.
    env = [i for i, f in enumerate(filters)
           if f[0] == 'ignore' and f[1] is None and f[2] is Warning
           and f[3] is None and not f[4]]
    if not env:
        return
    if module is None and stacklevel is not None:
        try:
            module = sys._getframe(stacklevel).f_globals.get('__name__')
        except ValueError:
            module = None
    text = None if message is None else str(message)
    for i, (_action, msg, cat, mod, lineno) in enumerate(filters):
        if not issubclass(category, cat):
            continue
        if msg is not None:
            if text is not None and not msg.match(text):
                continue    # aimed at other warnings: not ours to decide
            return          # a targeted filter set in code: respect it
        if mod is not None:
            if module is not None and not mod.match(module):
                continue
            return
        if lineno:
            return
        if i == env[-1]:
            if text is None:
                warnings.filterwarnings(action, category=category)
            else:           # this warning only: later ones get their own turn
                warnings.filterwarnings(
                    action, message='(?s)' + re.escape(text),
                    category=category)
        return              # the first applicable filter decides
