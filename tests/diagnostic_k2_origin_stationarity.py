"""
Diagnostic (NOT a regression test): for the nondiagonal k=2 cross-population
tree diagram, verify that origin_leaf_idx=0 and origin_leaf_idx=1 satisfy
stationarity:

    origin=1, total_C(τ, 0)  ==  origin=0, total_C(0, -τ)   [mirror]

The user reports that origin_leaf_idx=0 produces the τ-flipped (mirror)
curve relative to sim.  If this test fails, the bug is in the origin=0
code path.

Run with:
    sage -python -m pytest tests/diagnostic_k2_origin_stationarity.py -v
"""

import pytest
from sage.all import SR

# Use the fixtures from the main test file.
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from test_time_domain import (
    _tree_k2_cross_population,
    _propagator_data_2pop_nondiagonal,
)
from msrjd.integration.time_domain.pipeline import compute_correction_td


def _build_two_results():
    """Build Phase J total_C callables with origin_leaf_idx=0 and origin_leaf_idx=1."""
    pd = _propagator_data_2pop_nondiagonal()
    td = _tree_k2_cross_population()

    # origin=1 (existing test convention: pin t_2, free t_1)
    res1 = compute_correction_td(
        typed_diagrams=[td],
        prefactors=[SR(1)],
        propagator_data=pd,
        k=2,
        num_params=None,
        origin_leaf_idx=1,
    )
    assert not res1['skipped_kernel_ids']
    total_C_1 = res1['total_C']

    # origin=0 (new canonical convention: pin t_1, free t_2)
    res0 = compute_correction_td(
        typed_diagrams=[td],
        prefactors=[SR(1)],
        propagator_data=pd,
        k=2,
        num_params=None,
        origin_leaf_idx=0,
    )
    assert not res0['skipped_kernel_ids']
    total_C_0 = res0['total_C']

    return total_C_0, total_C_1


def test_origin_0_and_origin_1_match_by_stationarity():
    """
    By stationarity:
        ⟨φ_0(t_1) · φ_1(t_2)⟩ = ⟨φ_0(t_1+c) · φ_1(t_2+c)⟩

    With origin_idx=1: total_C(τ, 0) = ⟨φ_0(τ) · φ_1(0)⟩.
    With origin_idx=0: total_C(0, -τ) = ⟨φ_0(0) · φ_1(-τ)⟩.
    Shifting the second by +τ: = ⟨φ_0(τ) · φ_1(0)⟩ ✓

    So origin=1 @ (τ, 0)  MUST equal  origin=0 @ (0, -τ).
    """
    total_C_0, total_C_1 = _build_two_results()

    tau_vals = [-3.0, -1.0, -0.3, 0.3, 1.0, 3.0]

    # Check 1: origin=1, total_C(τ, 0)  ==  origin=0, total_C(0, -τ)
    mismatches = []
    for tv in tau_vals:
        v1 = complex(total_C_1(tv, 0.0)).real
        v0 = complex(total_C_0(0.0, -tv)).real
        if abs(v1 - v0) > 1e-4 * max(abs(v1), abs(v0), 1e-12):
            mismatches.append((tv, v1, v0))

    assert not mismatches, (
        f"Stationarity violated: origin=1 @ (τ,0) != origin=0 @ (0,-τ) at:\n"
        + '\n'.join(f'  τ={t:.2f}: origin=1 → {v1:.6e}, origin=0 → {v0:.6e}, '
                    f'rel_diff={abs(v1-v0)/max(abs(v1),abs(v0),1e-12):.3e}'
                    for t, v1, v0 in mismatches)
    )


def test_origin_0_values_directly():
    """
    Print origin=0 @ (0, τ) values for inspection.  These should be the
    physical cross-correlator at lag +τ (matching sim).
    """
    total_C_0, total_C_1 = _build_two_results()

    tau_vals = [-3.0, -1.0, -0.3, 0.3, 1.0, 3.0]
    print()
    print('Direct comparison at matching physical correlator:')
    print(f'{"tau":>8s}  {"origin=1 C(τ,0)":>18s}  {"origin=0 C(0,-τ)":>18s}  '
          f'{"origin=0 C(0,+τ)":>18s}')
    print('-' * 80)
    for tv in tau_vals:
        v1_pos = complex(total_C_1(tv, 0.0)).real
        v0_neg = complex(total_C_0(0.0, -tv)).real
        v0_pos = complex(total_C_0(0.0, tv)).real
        print(f'{tv:>8.3f}  {v1_pos:>18.6e}  {v0_neg:>18.6e}  {v0_pos:>18.6e}')
