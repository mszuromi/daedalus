"""proptd_: prototype setup-cost levers WITHOUT editing tracked files (monkeypatch / exec-modified copies).
usage: sage -python proptd_proto.py <ou2|ou1> <max_ell> <levers comma list or 'none'> [k]
levers:
  prune  : drop vertex types whose coefficient is EXACTLY 0 at num_params before enumeration/typing
  zskip  : drop typed diagrams whose numeric combined prefactor is exactly 0 before Phase J
  gtmemo : build_G_t_matrix once per (propagator_data, num_params)  [memo keyed by ids]
  modes  : _build_edge_mode_sums from a per-model complex table (poles, residues[k,pi,ri]) built once
  autmemo: memoize _automorphism_order per (id(td), fix_external)  (shares Aut_fixed between classify + wick)
  nodisp : skip the per-diagram debug-only display_stripped build+expand (exec-modified source)
  solve  : replace sage_solve on dt_e==0 by coefficient extraction + remove-by-identity (exec-modified source)
Prints wall, per-ell C at all taus (for bit-comparison), and phase walls."""
import sys, os, time, warnings, types, json
warnings.simplefilter('ignore'); sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
which = sys.argv[1]; MAX_ELL = int(sys.argv[2]); LEV = set(sys.argv[3].split(',')) - {'none'}
K = int(sys.argv[4]) if len(sys.argv) > 4 else 2
import daedalus as dd
from sage.all import SR, CDF
import engine.integration.time_domain.final_integral as FI
import engine.integration.time_domain.propagator_td as PT
import engine.integration.time_domain.pipeline as PL
import engine.diagrams.symmetry as SY
import api.compute as AC
import api._diagrams as AD

# ---- exec-modified source levers (must run before rebinding) ----
if LEV & {'nodisp', 'solve'}:
    src = open(FI.__file__).read()
    if 'nodisp' in LEV:
        old = """    display_stripped = cp
    for ei in edge_info:
        display_stripped = display_stripped * ei['smooth_factor']
    try:
        display_stripped = display_stripped.expand()
    except Exception:
        pass
"""
        assert old in src; src = src.replace(old, "    display_stripped = cp\n")
    if 'solve' in LEV:
        old = """            if int_var_to_eliminate is not None:
                try:
                    sol = sage_solve(
                        eq_expr == 0, int_var_to_eliminate,
                        solution_dict=True,
                    )
                except Exception:
                    sol = []
                if not sol:
                    subset_infeasible = True
                    break
                new_rhs = sol[0][int_var_to_eliminate]
                substitutions[int_var_to_eliminate] = new_rhs
                remaining_int_vars.remove(int_var_to_eliminate)"""
        new = """            if int_var_to_eliminate is not None:
                # linear in each var: eq = a*iv + rest  ->  iv = -rest/a
                _a = eq_expr.coefficient(int_var_to_eliminate, 1)
                _rest = (eq_expr - _a * int_var_to_eliminate).expand()
                if _a.is_zero() or int_var_to_eliminate in set(_rest.variables()):
                    try:
                        sol = sage_solve(eq_expr == 0, int_var_to_eliminate, solution_dict=True)
                    except Exception:
                        sol = []
                    if not sol:
                        subset_infeasible = True
                        break
                    new_rhs = sol[0][int_var_to_eliminate]
                else:
                    new_rhs = (-_rest / _a).expand()
                    if _VALSOLVE:
                        _sol = sage_solve(eq_expr == 0, int_var_to_eliminate, solution_dict=True)
                        _SOLVE_CHK[0] += 1
                        if not _sol or not (SR(_sol[0][int_var_to_eliminate]) - new_rhs).expand().is_zero():
                            _SOLVE_CHK[1] += 1
                substitutions[int_var_to_eliminate] = new_rhs
                remaining_int_vars = [_iv for _iv in remaining_int_vars if _iv is not int_var_to_eliminate]"""
        assert old in src; src = src.replace(old, new)
        old2 = """            for iv in remaining_int_vars:
                if iv in eq_vars:"""
        new2 = """            _eq_ids = {id(_x) for _x in eq_vars}
            _eq_names = {str(_x) for _x in eq_vars}
            for iv in remaining_int_vars:
                if str(iv) in _eq_names:"""
        assert old2 in src; src = src.replace(old2, new2)
    FI.__dict__['_VALSOLVE'] = bool(os.environ.get('VAL')); FI.__dict__['_SOLVE_CHK'] = _SOLVE_CHK = [0, 0]
    import atexit; atexit.register(lambda: print(f"   [solve] validated {_SOLVE_CHK[0]} eliminations, mismatches {_SOLVE_CHK[1]}"))
    exec(compile(src, FI.__file__, 'exec'), FI.__dict__)
    for mod in list(sys.modules.values()):
        if mod is None or mod is FI: continue
        for n_, v in list(getattr(mod, '__dict__', {}).items()):
            if isinstance(v, types.FunctionType) and getattr(v, '__module__', '') == FI.__name__ and n_ in FI.__dict__ and FI.__dict__[n_] is not v:
                setattr(mod, n_, FI.__dict__[n_])

CAP = {}
if 'gtmemo' in LEV:
    _o_bg = FI.build_G_t_matrix; _memo = {}
    def bg(propagator_data, t_sym, num_params=None):
        key = (id(propagator_data), id(num_params), str(t_sym))
        if key not in _memo: _memo[key] = _o_bg(propagator_data, t_sym, num_params=num_params)
        return _memo[key]
    FI.build_G_t_matrix = bg
if 'modes' in LEV:
    _tab = {}
    def bems(edge_info, propagator_data):
        key = id(propagator_data)
        if key not in _tab:
            pv = propagator_data.get('pole_vals'); Cm = propagator_data.get('C_mats')
            if pv is None or Cm is None: _tab[key] = None
            else:
                try:
                    lam = tuple(complex(CDF(SR(p))) * 1j for p in pv)
                    n = Cm[0].nrows() if Cm else 0
                    R = [[[complex(CDF(SR(Cm[kk][i, j]))) for j in range(n)] for i in range(n)] for kk in range(len(pv))]
                    _tab[key] = (lam, R, {})
                except Exception:
                    _tab[key] = None
        t = _tab[key]
        if t is None: return None
        lam, R, mcache = t; out = []
        for ei in edge_info:
            ri, pi = ei['ri'], ei['pi']
            modes = mcache.get((pi, ri))
            if modes is None:
                modes = tuple(zip(tuple(R[kk][pi][ri] for kk in range(len(lam))), lam)); mcache[(pi, ri)] = modes
            try: d_c = complex(ei['delta_coeff'])
            except Exception:
                try: d_c = complex(CDF(SR(ei['delta_coeff'])))
                except Exception: return None
            out.append(FI.EdgeModeSum(ri=ri, pi=pi, delta_coeff=d_c, modes=modes))
        return out
    FI._build_edge_mode_sums = bems
if 'autmemo' in LEV:
    _o_ao = SY._automorphism_order; _am = {}
    def ao(td, fix_external=True):
        key = (id(td), bool(fix_external))
        if key not in _am: _am[key] = (_o_ao(td, fix_external=fix_external), td)
        return _am[key][0]
    SY._automorphism_order = ao
if 'autfast' in LEV:
    # order only, no PermutationGroup object; validated against the original on every call when VAL=1
    _o_ao2 = SY._automorphism_order; _VAL = bool(os.environ.get('VAL')); _nv = [0, 0]
    def ao2(td, fix_external=True):
        D, partition, _, _ = SY._colored_incidence_digraph(td, fix_external=fix_external)
        o = int(D.automorphism_group(partition=partition, order=True, return_group=False))
        if _VAL:
            _nv[0] += 1
            if o != _o_ao2(td, fix_external=fix_external): _nv[1] += 1
        return o
    SY._automorphism_order = ao2
    import atexit; atexit.register(lambda: print(f"   [autfast] validated {_nv[0]} calls, mismatches {_nv[1]}") if _VAL else None)
if 'prune' in LEV:
    _o_cpr = AC.compute_poles_and_residues
    def cpr(prop, num_params, **kw):
        CAP['num_params'] = num_params; return _o_cpr(prop, num_params, **kw)
    AC.compute_poles_and_residues = cpr
    _o_eud = AC.enumerate_unique_diagrams
    def eud(ft, model, **kw):
        npar = CAP['num_params']; vt = kw['vtypes']; keep = []
        for v in vt:
            try: z = complex(CDF(SR(v.coefficient).subs(npar))) == 0
            except Exception: z = False
            if not z: keep.append(v)
        print(f"   [prune] vtypes {len(vt)} -> {len(keep)}")
        kw['vtypes'] = keep
        return _o_eud(ft, model, **kw)
    AC.enumerate_unique_diagrams = eud
if 'zskip' in LEV:
    _o_cct = AC.compute_correction_td
    def cct(*a, **kw):
        npar = kw.get('num_params'); tds, pfs = [], []
        for td, pf in zip(kw['typed_diagrams'], kw['prefactors']):
            try: z = complex(CDF(SR(pf).subs(npar) if npar else SR(pf))) == 0
            except Exception: z = False
            if not z: tds.append(td); pfs.append(pf)
        print(f"   [zskip] {len(kw['typed_diagrams'])} -> {len(tds)} diagrams")
        kw['typed_diagrams'] = tds; kw['prefactors'] = pfs
        return _o_cct(*a, **kw)
    AC.compute_correction_td = cct
if os.environ.get('PROFCCT'):
    import cProfile as _cP, pstats as _ps, io as _io, atexit as _ax
    _PRC = _cP.Profile(); _o_cct3 = AC.compute_correction_td
    def cct3(*a, **kw):
        _PRC.enable(); r = _o_cct3(*a, **kw); _PRC.disable(); return r
    AC.compute_correction_td = cct3
    def _dump():
        _s = _io.StringIO(); st = _ps.Stats(_PRC, stream=_s).sort_stats(os.environ['PROFCCT'])
        st.print_stats(int(os.environ.get('PN', '30')))
        if os.environ.get('CALLERS'): st.print_callers(os.environ['CALLERS'])
        print(_s.getvalue().replace(os.getcwd() + '/', '').replace('/var/tmp/sage-10.8-current/local/lib/python3.13/site-packages/', ''))
    _ax.register(_dump)
if os.environ.get('S'):
    _o_cct2 = AC.compute_correction_td
    def cct2(*a, **kw):
        r = _o_cct2(*a, **kw); r['total_C_batch'] = lambda pts, **k_: [0j] * len(pts); r['total_C'] = lambda *t: 0j; return r
    AC.compute_correction_td = cct2

if which == 'ou2':
    model = dd.load_model('ou_quartic_two_dim')[0]; ext = [('dx',1)]*K; taus = np.linspace(0, 3, 7)
elif which == 'ou1':
    model = dd.load_model('ou_quartic')[0]; ext = [('dx',1)]*K; taus = np.linspace(0, 3, 7)
else:
    raise SystemExit('unknown model key: use ou2 or ou1')
kw = dict(k=K, max_ell=MAX_ELL, tau_grid=taus, use_cache=bool(int(os.environ.get('UC', '1'))), parallel=False, verbose=True, external_fields=ext)
import io, contextlib
buf = io.StringIO()
t0 = time.time()
import cProfile, pstats
_pr = cProfile.Profile() if os.environ.get('PROF') else None
if _pr: _pr.enable()
with contextlib.redirect_stdout(buf): res = AC.compute_cumulants(model, **kw)
if _pr:
    _pr.disable(); _s = io.StringIO(); pstats.Stats(_pr, stream=_s).sort_stats(os.environ.get('PROF')).print_stats(int(os.environ.get('PN', '30')))
    print(_s.getvalue().replace(os.getcwd() + '/', '').replace('/var/tmp/sage-10.8-current/local/lib/python3.13/site-packages/', ''))
wall = time.time() - t0
lines = buf.getvalue().splitlines()
print(f"{which} k={K} max_ell={MAX_ELL} levers={sorted(LEV) or ['none']}: wall {wall:.2f}s")
for l in lines:
    if '] done in' in l or '[prune]' in l or '[zskip]' in l or 'unique' in l: print('  ' + l.strip())
out = {}
if K == 2:
    for ell, arr in res['C_tau_by_ell'].items():
        out[str(ell)] = [[float(np.real(x)), float(np.imag(x))] for x in arr]
        print(f"  ell={ell}: " + " ".join(f"{np.real(x):.14g}" for x in arr))
fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"proptd_res_{which}_k{K}_ell{MAX_ELL}_{'-'.join(sorted(LEV)) or 'none'}{'_S' if os.environ.get('S') else ''}.json")
json.dump({'wall': wall, 'C': out}, open(fn, 'w'))
