"""
engine.core.serialize
====================
Save and load expanded field theories and their propagator data.

File format
-----------
Each saved model is a directory containing:
    metadata.json       — plain-Python data (field names, taylor_order, etc.)
    symbolic_data.sobj  — SageMath symbolic objects (matrices, polynomials)

The model dict itself (which contains lambdas) is NOT serialized.
Instead, metadata.json stores the path to the model .py file and the
variable name, so the model can be re-imported for re-expansion.

Build Phase A.
"""

import json
import os
from datetime import datetime

from sage.all import save as sage_save, load as sage_load, version as sage_version


# ── Helpers ──────────────────────────────────────────────────────────────────

def _strip_callables(spec_list):
    """
    Return a copy of a list of spec dicts with all callable values removed.
    Keeps only JSON-serializable fields (str, int, float, bool, list, dict, None).
    """
    clean = []
    for spec in spec_list:
        d = {}
        for k, v in spec.items():
            if callable(v):
                continue
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                d[k] = v
        clean.append(d)
    return clean


def _jsonable_index_sets(index_sets):
    """Convert index_sets values to plain Python lists (from range, etc.)."""
    return {k: list(v) for k, v in index_sets.items()}


# ── Save ─────────────────────────────────────────────────────────────────────

def save_model(path, ft, propagator_data=None, stationarity=True,
                model_file=None, model_var_name=None):
    """
    Save an expanded FieldTheory and its propagator data to disk.

    Parameters
    ----------
    path : str
        Directory path to save into (created if needed).
    ft : FieldTheory
        An expanded FieldTheory instance (expand() must have been called).
    propagator_data : dict or None
        Propagator computation results.  Expected keys (all optional):
            K_ft, G_ft, adj_ft, D_omega, pole_vals, C_mats, G_t,
            G_ft_explicit, propagator_branch, nf, resp_names, phys_names,
            resp_sr, phys_sr
        If None, only the action/bigrade data is saved (no propagator).
    stationarity : bool
        Whether the system is time-translation invariant.
    model_file : str or None
        Path to the model .py file (relative to project root).
    model_var_name : str or None
        Variable name of the model dict in the model file
        (e.g. 'HAWKES_MODEL').
    """
    ft._require_expanded()
    os.makedirs(path, exist_ok=True)

    m   = ft.model
    ns  = ft._ns
    pd  = propagator_data or {}

    # ── Build metadata (JSON-serializable) ────────────────────────────────
    meta = {
        'format_version':    1,
        'sage_version':      str(sage_version()),
        'timestamp':         datetime.now().isoformat(),

        # Model identity
        'model_name':        m.get('name', ''),
        'model_file':        model_file,
        'model_var_name':    model_var_name,

        # Expansion info
        'taylor_order':      ft.taylor_order,
        'n_tilde':           ft._n_tilde,
        'ring_var_names':    list(ns._ring_var_names),

        # Field metadata
        'index_sets':        _jsonable_index_sets(m['index_sets']),
        'resp_field_specs':  _strip_callables(m.get('response_fields', [])),
        'phys_field_specs':  _strip_callables(m.get('physical_fields', [])),
        'param_specs':       _strip_callables(m.get('parameters', [])),
        'kernel_specs':      _strip_callables(m.get('kernels', [])),
        'operator_specs':    _strip_callables(m.get('operators', [])),

        # Stationarity and time-dependence
        'stationarity':      stationarity,
        'time_dependent_parameters': m.get('time_dependent_parameters', []),
        'noise_structure':   m.get('noise_structure', {
            'temporal_type': 'white',
            'amplitude_params': [],
        }),

        # Propagator info
        'nf':                pd.get('nf', None),
        'resp_names':        pd.get('resp_names', None),
        'phys_names':        pd.get('phys_names', None),
        'propagator_branch': pd.get('propagator_branch', None),
        'G_ft_explicit':     pd.get('G_ft_explicit', False),
        'n_poles':           len(pd.get('pole_vals', [])),

        # Sector summary (for quick inspection without loading .sobj)
        'nonzero_sectors':   sorted(
            [list(k) for k in ft.sectors().keys()]
        ),
    }

    with open(os.path.join(path, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # ── Build symbolic data dict ──────────────────────────────────────────
    sym = {
        'R':          ft._R,
        'S_raw':      ft._S_raw,
        'by_tp':      ft._by_tp,
        'n_tilde':    ft._n_tilde,

        # Propagator objects (may be None / empty)
        'K_ft':       pd.get('K_ft', None),
        'G_ft':       pd.get('G_ft', None),
        'adj_ft':     pd.get('adj_ft', None),
        'D_omega':    pd.get('D_omega', None),
        'pole_vals':  pd.get('pole_vals', []),
        'C_mats':     pd.get('C_mats', []),
        'G_t':        pd.get('G_t', None),
    }

    sage_save(sym, os.path.join(path, 'symbolic_data'))
    # sage_save appends .sobj automatically


# ── Load ─────────────────────────────────────────────────────────────────────

def load_model(path):
    """
    Load a saved model from disk.

    Parameters
    ----------
    path : str
        Directory containing metadata.json and symbolic_data.sobj.

    Returns
    -------
    meta : dict
        The metadata (JSON contents).
    data : dict
        The symbolic data dict with keys:
            R, S_raw, by_tp, n_tilde,
            K_ft, G_ft, adj_ft, D_omega, pole_vals, C_mats, G_t
    """
    meta_path = os.path.join(path, 'metadata.json')
    sobj_path = os.path.join(path, 'symbolic_data.sobj')

    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f'No metadata.json in {path}')
    if not os.path.isfile(sobj_path):
        raise FileNotFoundError(f'No symbolic_data.sobj in {path}')

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    data = sage_load(sobj_path)

    return meta, data


# ── Reload model for re-expansion ────────────────────────────────────────────

def reload_model(meta, project_root=None):
    """
    Re-import the model dict from the stored model_file path.

    Parameters
    ----------
    meta : dict
        The metadata dict (from load_model).
    project_root : str or None
        Root directory to resolve model_file relative to.
        If None, uses the current working directory.

    Returns
    -------
    model : dict
        The model specification dict, ready to pass to FieldTheory().

    Raises
    ------
    ValueError
        If model_file or model_var_name is not set in metadata.
    FileNotFoundError
        If the model file does not exist.
    AttributeError
        If the model variable is not found after loading the file.
    """
    model_file = meta.get('model_file')
    model_var  = meta.get('model_var_name')

    if not model_file:
        raise ValueError('metadata has no model_file — cannot reload model')
    if not model_var:
        raise ValueError('metadata has no model_var_name — cannot reload model')

    root = project_root or os.getcwd()
    full_path = os.path.join(root, model_file)

    if not os.path.isfile(full_path):
        raise FileNotFoundError(f'Model file not found: {full_path}')

    # Use SageMath's load() which handles .py files (executes them in a namespace)
    # We need to capture the resulting variable, so we exec into a dict.
    ns = {}
    with open(full_path, 'r') as f:
        code = f.read()
    exec(compile(code, full_path, 'exec'), ns)

    if model_var not in ns:
        raise AttributeError(
            f'Variable {model_var!r} not found in {full_path}. '
            f'Available names: {[k for k in ns if not k.startswith("_")]}'
        )

    return ns[model_var]
