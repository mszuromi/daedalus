"""
engine.enumeration.degree_scan
==============================
Scan prediagrams for max vertex degree, compare to stored Taylor order,
and trigger re-expansion if needed.  This bridges the enumeration output
(Phase 2) back to the theory data (Phase 1).

Build Phase C.
"""

import os

from engine.core.serialize import load_theory, save_theory, reload_model
from engine.core.field_theory import FieldTheory


def max_vertex_degree(prediagrams):
    """
    Compute the maximum total degree (in_degree + out_degree) across all
    non-leaf vertices in a collection of prediagrams.

    Parameters
    ----------
    prediagrams : list of (D, G, leaves, internal)
        Output from enumerate_prediagrams().  D is a SageMath DiGraph,
        leaves and internal are vertex lists.

    Returns
    -------
    int
        Maximum total degree, or 0 if no internal vertices exist.
    """
    max_deg = 0
    for (D, G, leaves, internal) in prediagrams:
        leaf_set = set(leaves)
        for v in D.vertices():
            if v in leaf_set:
                continue
            deg = D.in_degree(v) + D.out_degree(v)
            if deg > max_deg:
                max_deg = deg
    return max_deg


def scan_source_vertices(prediagrams):
    """
    Identify source vertices (in_degree = 0, not a leaf) across all
    prediagrams and return the set of out-degrees needed.

    Parameters
    ----------
    prediagrams : list of (D, G, leaves, internal)

    Returns
    -------
    set of int
        Out-degrees required for source vertices.
    """
    source_out_degrees = set()
    for (D, G, leaves, internal) in prediagrams:
        leaf_set = set(leaves)
        for v in D.vertices():
            if v in leaf_set:
                continue
            if D.in_degree(v) == 0:
                source_out_degrees.add(D.out_degree(v))
    return source_out_degrees


def check_taylor_order(meta, max_degree):
    """
    Check whether the saved theory's Taylor order is sufficient.

    Parameters
    ----------
    meta : dict
        Metadata from load_theory().
    max_degree : int
        Maximum vertex degree from max_vertex_degree().

    Returns
    -------
    sufficient : bool
    current_order : int
    required_order : int
    """
    current = meta['taylor_order']
    return current >= max_degree, current, max_degree


def ensure_taylor_order(theory_path, prediagrams, project_root=None):
    """
    Load a saved theory and re-expand if the Taylor order is too low
    for the given prediagrams.

    Parameters
    ----------
    theory_path : str
        Path to the saved theory directory.
    prediagrams : list of (D, G, leaves, internal)
        Prediagrams to scan.
    project_root : str or None
        Root directory for resolving the model file path.

    Returns
    -------
    meta : dict
    data : dict
    """
    meta, data = load_theory(theory_path)
    max_deg = max_vertex_degree(prediagrams)

    sufficient, current, required = check_taylor_order(meta, max_deg)

    if sufficient:
        return meta, data

    # Re-expand at the required order
    print(f'Re-expanding theory from order {current} to {required}...')

    model = reload_model(meta, project_root=project_root)
    ft = FieldTheory(model, taylor_order=required)
    ft.expand()

    # Re-save, preserving model identity info
    save_theory(
        theory_path, ft,
        propagator_data=None,  # propagator must be recomputed separately
        stationarity=meta.get('stationarity', True),
        model_file=meta.get('model_file'),
        model_var_name=meta.get('model_var_name'),
    )
    print(f'Theory re-expanded and saved. Note: propagator data must be recomputed.')

    return load_theory(theory_path)
