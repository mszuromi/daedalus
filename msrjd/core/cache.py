"""
msrjd.core.cache
=================
Persistent, stage-based cache for Feynman diagram pipeline results.

Each model's computed results are stored in a single directory.  Pipeline
stages (enumeration, filtering, type assignment, deduplication, kernel
grouping) are saved as SageMath ``.sobj`` files keyed by ``(k, loop_order)``
so that restarting a notebook can skip expensive re-computation.

Typical usage in a notebook::

    from msrjd.core.cache import PipelineCache

    cache = PipelineCache("saved_results/hawkes_2pop")

    # Expensive enumeration — run once, then reload from cache.
    pds_0, counts_0 = cache.get_or_compute(
        "prediagrams", k=2, loop_order=0,
        compute_fn=lambda: do_expensive_enumeration(),
    )

    # Or manual save/load:
    cache.save("unique_typed", k=2, loop_order=1, data=unique_1loop)
    unique_1loop = cache.load("unique_typed", k=2, loop_order=1)

Build Phase A.
"""

import json
import os
from datetime import datetime

from sage.all import save as sage_save, load as sage_load


class PipelineCache:
    """
    Stage-based disk cache for diagram pipeline results.

    Parameters
    ----------
    root : str
        Path to the cache directory (created on first write).
    """

    # Well-known stage names (informational, not enforced).
    STAGES = (
        'prediagrams',      # (trees, topos, pds, counts)
        'filtered',         # (kept, discarded_count)
        'typed',            # list of TypedDiagram
        'unique_typed',     # deduplicated list of TypedDiagram
        'kernel_groups',    # output of group_diagrams_by_kernel
        'integrand_results', # per-diagram build_integrand_stationary output
    )

    def __init__(self, root):
        self.root = os.path.expanduser(root)

    # ── Key helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _stage_key(stage, k=None, loop_order=None):
        """Build a filename stem from stage name and optional (k, ℓ)."""
        parts = [stage]
        if k is not None:
            parts.append(f'k{k}')
        if loop_order is not None:
            parts.append(f'l{loop_order}')
        return '_'.join(parts)

    def _sobj_path(self, stage, k=None, loop_order=None):
        key = self._stage_key(stage, k, loop_order)
        return os.path.join(self.root, key + '.sobj')

    # ── Core API ─────────────────────────────────────────────────────────

    def exists(self, stage, k=None, loop_order=None):
        """Check whether a cached result exists for this stage/key."""
        return os.path.isfile(self._sobj_path(stage, k, loop_order))

    def save(self, stage, data, k=None, loop_order=None):
        """
        Save pipeline data to disk.

        Parameters
        ----------
        stage : str
            Stage name (e.g. ``'prediagrams'``, ``'unique_typed'``).
        data : object
            Any picklable / SageMath-serializable object.
        k : int, optional
            Number of external legs.
        loop_order : int, optional
            Loop order ℓ.
        """
        os.makedirs(self.root, exist_ok=True)
        path = self._sobj_path(stage, k, loop_order)
        sage_save(data, path.removesuffix('.sobj'))  # sage_save adds .sobj
        self._update_manifest(stage, k, loop_order)

    def load(self, stage, k=None, loop_order=None):
        """
        Load cached pipeline data.

        Raises ``FileNotFoundError`` if not cached.
        """
        path = self._sobj_path(stage, k, loop_order)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'No cached data for stage={stage!r}, k={k}, '
                f'loop_order={loop_order} at {path}'
            )
        return sage_load(path)

    def get_or_compute(self, stage, compute_fn, k=None, loop_order=None):
        """
        Return cached data if available, otherwise call *compute_fn*,
        cache the result, and return it.

        Parameters
        ----------
        stage : str
        compute_fn : callable
            Called with no arguments.  Its return value is saved.
        k, loop_order : int, optional

        Returns
        -------
        object
            The cached or freshly computed data.
        """
        if self.exists(stage, k, loop_order):
            return self.load(stage, k, loop_order)
        data = compute_fn()
        self.save(stage, data, k, loop_order)
        return data

    # ── Manifest (human-readable index) ──────────────────────────────────

    def _manifest_path(self):
        return os.path.join(self.root, 'manifest.json')

    def _update_manifest(self, stage, k, loop_order):
        """Append an entry to the JSON manifest."""
        mp = self._manifest_path()
        if os.path.isfile(mp):
            with open(mp) as f:
                manifest = json.load(f)
        else:
            manifest = {'entries': [], 'created': datetime.now().isoformat()}

        key = self._stage_key(stage, k, loop_order)
        # Remove old entry for this key if present.
        manifest['entries'] = [
            e for e in manifest['entries'] if e.get('key') != key
        ]
        manifest['entries'].append({
            'key': key,
            'stage': stage,
            'k': k,
            'loop_order': loop_order,
            'saved_at': datetime.now().isoformat(),
        })
        manifest['updated'] = datetime.now().isoformat()

        with open(mp, 'w') as f:
            json.dump(manifest, f, indent=2)

    def list_cached(self):
        """Return a list of dicts describing all cached entries."""
        mp = self._manifest_path()
        if not os.path.isfile(mp):
            return []
        with open(mp) as f:
            manifest = json.load(f)
        return manifest.get('entries', [])

    def clear(self, stage=None, k=None, loop_order=None):
        """
        Remove cached data.

        If all three parameters are None, removes the entire cache
        directory.  Otherwise removes only the matching entry.
        """
        if stage is None and k is None and loop_order is None:
            import shutil
            if os.path.isdir(self.root):
                shutil.rmtree(self.root)
            return

        path = self._sobj_path(stage, k, loop_order)
        if os.path.isfile(path):
            os.remove(path)
        # Update manifest.
        mp = self._manifest_path()
        if os.path.isfile(mp):
            key = self._stage_key(stage, k, loop_order)
            with open(mp) as f:
                manifest = json.load(f)
            manifest['entries'] = [
                e for e in manifest['entries'] if e.get('key') != key
            ]
            with open(mp, 'w') as f:
                json.dump(manifest, f, indent=2)

    def __repr__(self):
        entries = self.list_cached()
        return f'PipelineCache({self.root!r}, {len(entries)} entries)'
