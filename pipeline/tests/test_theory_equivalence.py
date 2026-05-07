"""
Equivalence test: each generated theory in pipeline/theories/ must
produce a HAWKES_MODEL dict whose FieldTheory.expand(): bigrade
sectors match the corresponding hand-written model in models/.

Comparison is done at the BIGRADE-SECTOR level (not byte-exact dict
equality), because the lambdas in the model dict are different
function objects but should produce identical SR expressions when
evaluated by FieldTheory.

Run with:

    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES sage -python \
        pipeline/tests/test_theory_equivalence.py

A pair is considered "equivalent" when:
  * the (n_tilde, n_phys) sector keys match,
  * each sector polynomial has the same set of monomials and
    coefficients (compared via SR equality / simplify_full).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from sage.all import SR
from msrjd.core.field_theory import FieldTheory
from msrjd.core.vertices import (
    extract_source_types, extract_vertex_types, NoiseSourceType,
)


def _expand_and_summarise(model, label, taylor_order=4):
    """Run FT.expand and return a summary dict of bigrade sectors."""
    ft = FieldTheory(model, taylor_order=taylor_order)
    ft.expand()
    sanity = ft.sanity_check()
    summary = {
        'label':    label,
        'sanity':   sanity,
        'n_tilde':  ft._n_tilde,
        'sectors':  {},
    }
    for (n_t, n_p), poly in ft.sectors().items():
        # Convert to a dict keyed by exponent vector for comparison
        summary['sectors'][(n_t, n_p)] = dict(poly.dict())
    summary['vtypes_count']  = len(extract_vertex_types(ft))
    stypes = extract_source_types(ft)
    summary['stypes_count']  = len(stypes)
    summary['noise_count']   = sum(
        1 for s in stypes if isinstance(s, NoiseSourceType)
    )
    return summary


def _compare(s_hand, s_gen):
    """Return list of differences between two summaries."""
    diffs = []
    for fld in ('sanity', 'n_tilde', 'vtypes_count', 'stypes_count',
                'noise_count'):
        if s_hand[fld] != s_gen[fld]:
            diffs.append(
                f'{fld}: hand={s_hand[fld]} vs generated={s_gen[fld]}'
            )

    keys_h = set(s_hand['sectors'].keys())
    keys_g = set(s_gen['sectors'].keys())
    if keys_h != keys_g:
        diffs.append(
            f'sector keys mismatch: only-hand={keys_h - keys_g}, '
            f'only-gen={keys_g - keys_h}'
        )

    for key in sorted(keys_h & keys_g):
        d_h = s_hand['sectors'][key]
        d_g = s_gen['sectors'][key]
        if set(d_h.keys()) != set(d_g.keys()):
            diffs.append(
                f'sector {key}: monomial keys differ '
                f'(only-hand={len(set(d_h.keys()) - set(d_g.keys()))}, '
                f'only-gen={len(set(d_g.keys()) - set(d_h.keys()))})'
            )
            continue
        for mono in d_h:
            ch = SR(d_h[mono])
            cg = SR(d_g[mono])
            if not (ch - cg).simplify_full().is_zero():
                diffs.append(
                    f'sector {key} mono {mono}: '
                    f'hand={ch} vs gen={cg}'
                )
    return diffs


# Pairs of (generated theory module path, hand-written model module path,
# description).  Hand-written paths are imported relative to the repo root.
PAIRS = [
    ('pipeline.theories.linear_hawkes_2pop_delta',
     'models.hawkes_linear_sage',
     'Linear Hawkes (delta synapse)'),
    ('pipeline.theories.linear_hawkes_2pop_expg',
     'models.hawkes_linear_expg',
     'Linear Hawkes (exp synapse)'),
    ('pipeline.theories.quad_hawkes_2pop_expg',
     'models.hawkes_quad_expg',
     'Quadratic Hawkes (exp synapse)'),
    ('pipeline.theories.linear_hawkes_2pop_expg_gtas',
     'models.hawkes_linear_expg_gtas',
     'Linear Hawkes + GTaS'),
    ('pipeline.theories.quad_hawkes_2pop_expg_gtas',
     'models.hawkes_quad_expg_gtas',
     'Quadratic Hawkes + GTaS'),
]


def main():
    import importlib
    overall_pass = True
    for gen_path, hand_path, desc in PAIRS:
        print(f'\n{"=" * 70}')
        print(f'  {desc}')
        print(f'  hand: {hand_path}')
        print(f'  gen:  {gen_path}')
        print('=' * 70)
        try:
            hand_mod = importlib.import_module(hand_path)
            gen_mod  = importlib.import_module(gen_path)
        except Exception as e:
            print(f'  IMPORT FAILED: {e!r}')
            overall_pass = False
            continue

        try:
            s_hand = _expand_and_summarise(hand_mod.HAWKES_MODEL, 'hand')
            s_gen  = _expand_and_summarise(gen_mod.HAWKES_MODEL, 'gen')
        except Exception as e:
            print(f'  EXPAND FAILED: {e!r}')
            import traceback; traceback.print_exc()
            overall_pass = False
            continue

        diffs = _compare(s_hand, s_gen)
        if diffs:
            print(f'  [FAIL] {len(diffs)} differences:')
            for d in diffs[:30]:
                print(f'    - {d}')
            if len(diffs) > 30:
                print(f'    ... and {len(diffs) - 30} more')
            overall_pass = False
        else:
            print(f'  [PASS] {len(s_hand["sectors"])} sectors match, '
                  f'vtypes={s_hand["vtypes_count"]}, '
                  f'sources={s_hand["stypes_count"]} '
                  f'(NoiseSourceType={s_hand["noise_count"]})')

    print()
    print('=' * 70)
    print(f'OVERALL: {"PASS" if overall_pass else "FAIL"}')
    print('=' * 70)
    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())
