"""
FUNCTIONAL AUDIT #2 — serializer round-trip for the NEW spatial features.

For each spec covering a new feature, this script runs the full UI persistence
path:
    spec (form-state dict)
      -> render_theory_file(spec)               (forward render)
      -> save_theory_to_file(spec, tmp)         (write .theory.py)
      -> importlib-load the file and call build()  (file must build)
      -> load_spec_from_file(tmp)               (reverse parse -> spec')
      -> render_theory_file(spec')              (re-render)
      -> assert render(spec) == render(spec')   (idempotence modulo boilerplate)

It also inspects spec' field-by-field for the specific new properties the audit
must certify round-trip: spatial_dim per field, boundary mode+params, initial
mode+params, operator_ir flag, dyson policy {mode,order}, reference_diffusion,
multiple equations, multiple physical_fields, full action text (Dx/Lap/phi~^3).

Run:  timeout 900 sage -python scratch/serializer_roundtrip_audit.py
"""
import importlib.util
import json
import os
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.theory_serialize import (
    render_theory_file, save_theory_to_file, load_spec_from_file)

RESULTS = []  # list of (trial_name, status, note)
FINDINGS = []  # list of dicts


def record(name, status, note):
    RESULTS.append((name, status, note))
    print(f"[{status.upper():4}] {name}: {note}")


def finding(severity, title, detail, where, fix):
    FINDINGS.append(dict(severity=severity, title=title, detail=detail,
                         where=where, fix=fix))
    print(f"  !! ({severity}) {title}: {detail}")


def _load_and_build(path):
    """importlib-load a .theory.py and call build(); return (model, err)."""
    spec_ = importlib.util.spec_from_file_location("audit_theory_mod", path)
    mod = importlib.util.module_from_spec(spec_)
    try:
        spec_.loader.exec_module(mod)
        model = mod.build()
        return model, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def run_trial(name, spec, expectations):
    """Full render->save->build->load->re-render->idempotence trial.

    expectations: dict of checks to assert on the RELOADED spec', each a
    callable(spec_reloaded)->(ok:bool, msg:str).
    """
    notes = []
    ok_overall = True
    try:
        src1 = render_theory_file(spec)
    except Exception as e:
        record(name, "fail", f"render_theory_file raised: {e}")
        return

    # Save to a temp .theory.py
    tmpdir = tempfile.mkdtemp(prefix="audit_")
    path = os.path.join(tmpdir, f"{name}.theory.py")
    try:
        save_theory_to_file(spec, path)
    except Exception as e:
        record(name, "fail", f"save_theory_to_file raised: {e}")
        return

    # The generated file must build()
    model, err = _load_and_build(path)
    if err is not None:
        record(name, "fail", f"generated file failed to build(): {err.splitlines()[0]}")
        notes.append("BUILD-FAIL: " + err.splitlines()[0])
        ok_overall = False
    else:
        notes.append("build() OK")

    # Reverse: load_spec_from_file
    try:
        spec2 = load_spec_from_file(path)
    except Exception as e:
        record(name, "fail", f"load_spec_from_file raised: {e}")
        return

    # Re-render and check idempotence
    try:
        src2 = render_theory_file(spec2)
    except Exception as e:
        record(name, "fail", f"re-render of reloaded spec raised: {e}")
        return

    idempotent = (src1 == src2)
    if not idempotent:
        ok_overall = False
        # find first differing line for a compact diff
        l1 = src1.splitlines()
        l2 = src2.splitlines()
        diff_msg = "lengths %d vs %d" % (len(l1), len(l2))
        for i in range(max(len(l1), len(l2))):
            a = l1[i] if i < len(l1) else "<none>"
            b = l2[i] if i < len(l2) else "<none>"
            if a != b:
                diff_msg = f"first diff @line {i}: {a!r} != {b!r}"
                break
        notes.append("NOT-IDEMPOTENT: " + diff_msg)
    else:
        notes.append("idempotent")

    # Per-property expectations on the reloaded spec
    for label, check in expectations.items():
        try:
            ok, msg = check(spec2)
        except Exception as e:
            ok, msg = False, f"check raised {e}"
        if not ok:
            ok_overall = False
            notes.append(f"PROP-FAIL[{label}]: {msg}")
        else:
            notes.append(f"prop[{label}] OK")

    status = "pass" if ok_overall else "fail"
    record(name, status, "; ".join(notes))
    return spec2, src1, src2


# ════════════════════════════════════════════════════════════════════
# SPEC 1 — spatial single-field reaction-diffusion (Allen-Cahn-ish):
#   spatial_dim=1, boundary='infinite', initial='stationary',
#   operator_ir flag, g*p^2 cubic-ish vertex in the action.
# ════════════════════════════════════════════════════════════════════
spec_rd = {
    'name': 'audit_rd_single',
    'description': 'single-field spatial RD audit',
    'populations': [],
    'n_populations': 0,
    'response_fields': [],
    'physical_fields': [
        {'name': 'phi', 'spatial_dim': 1, 'description': 'order param'},
    ],
    'parameters': [
        {'name': 'mu', 'default': 1.0, 'domain': 'positive'},
        {'name': 'D',  'default': 1.0, 'domain': 'positive'},
        {'name': 'g',  'default': 0.5, 'domain': 'real'},
        {'name': 'T',  'default': 1.0, 'domain': 'positive'},
    ],
    'functions': [],
    'kernels': [],
    'cgf_terms': [],
    'action_text': 'pt*(Dt(phi) + mu*phi - D*Lap(phi) + g*phi^2) - T*pt^2',
    'equations': [
        {'lhs': '(Dt + mu - D*Laplacian)*phi', 'rhs': '0', 'population': None},
    ],
    'stability_analysis': False,
    'default_fundamental': {},
    'metadata': {},
    'boundary': {'mode': 'infinite'},
    'initial': {'mode': 'stationary'},
    # The NEW operator-IR flag.  KPZ/RD derivative theories author it.
    'operator_ir': True,
}


def _has_field_dim(name, dim):
    def chk(s):
        for f in s.get('physical_fields', []):
            if f['name'] == name:
                got = int(f.get('spatial_dim') or 0)
                return (got == dim, f"{name}.spatial_dim={got} (want {dim})")
        return False, f"field {name} missing"
    return chk


def _boundary_is(mode, **params):
    def chk(s):
        bc = s.get('boundary')
        if not bc:
            return False, "boundary dropped (no 'boundary' key on reloaded spec)"
        if bc.get('mode') != mode:
            return False, f"boundary mode={bc.get('mode')} want {mode}"
        for k, v in params.items():
            if bc.get(k) != v:
                return False, f"boundary[{k}]={bc.get(k)} want {v}"
        return True, f"boundary={bc}"
    return chk


def _initial_is(mode):
    def chk(s):
        ic = s.get('initial')
        if not ic:
            return False, "initial dropped (no 'initial' key)"
        return (ic.get('mode') == mode, f"initial={ic}")
    return chk


def _operator_ir_is(val):
    def chk(s):
        got = s.get('operator_ir')
        return (bool(got) == bool(val),
                f"operator_ir={got!r} (key present={('operator_ir' in s)}) want {val}")
    return chk


def _action_contains(*substrs):
    def chk(s):
        a = s.get('action_text') or ''
        missing = [x for x in substrs if x not in a]
        return (not missing, f"action missing {missing}" if missing
                else "action substrings present")
    return chk


def _n_equations(n):
    def chk(s):
        got = len(s.get('equations') or [])
        return (got == n, f"#equations={got} want {n}")
    return chk


def _n_physical(n):
    def chk(s):
        got = len(s.get('physical_fields') or [])
        return (got == n, f"#physical_fields={got} want {n}")
    return chk


def _dyson_is(mode, order):
    def chk(s):
        dy = s.get('dyson')
        if not dy:
            return False, "dyson dropped (no 'dyson' key)"
        return (dy.get('mode') == mode and int(dy.get('order', -999)) == order,
                f"dyson={dy} want mode={mode} order={order}")
    return chk


def _refdiff_is(val):
    def chk(s):
        rd = s.get('reference_diffusion')
        if rd is None:
            return False, "reference_diffusion dropped (key None/absent)"
        return (abs(float(rd) - val) < 1e-12, f"reference_diffusion={rd} want {val}")
    return chk


run_trial('spec1_rd_single', spec_rd, {
    'spatial_dim': _has_field_dim('phi', 1),
    'boundary':    _boundary_is('infinite'),
    'initial':     _initial_is('stationary'),
    'operator_ir': _operator_ir_is(True),
    'action':      _action_contains('Dt(phi)', 'Lap(phi)', 'g*phi^2'),
    'n_eq':        _n_equations(1),
})


# ════════════════════════════════════════════════════════════════════
# SPEC 2 — coupled 2-field spatial with Dyson policy + reference_diffusion.
#   two physical_fields (spatial_dim=1), two equations, dyson_order(2),
#   reference_diffusion(1.0).
# ════════════════════════════════════════════════════════════════════
spec_coupled = {
    'name': 'audit_coupled_dyson',
    'description': 'coupled 2-field spatial, Dyson-dressed',
    'populations': [],
    'n_populations': 0,
    'response_fields': [],
    'physical_fields': [
        {'name': 'a', 'spatial_dim': 1, 'description': 'species A'},
        {'name': 'b', 'spatial_dim': 1, 'description': 'species B'},
    ],
    'parameters': [
        {'name': 'Da', 'default': 1.0, 'domain': 'positive'},
        {'name': 'Db', 'default': 2.0, 'domain': 'positive'},
        {'name': 'ra', 'default': 1.0, 'domain': 'positive'},
        {'name': 'rb', 'default': 1.0, 'domain': 'positive'},
        {'name': 'kappa',  'default': 0.3, 'domain': 'real'},
        {'name': 'Ta', 'default': 1.0, 'domain': 'positive'},
        {'name': 'Tb', 'default': 1.0, 'domain': 'positive'},
    ],
    'functions': [],
    'kernels': [],
    'cgf_terms': [],
    'action_text': ('at*(Dt(a) + ra*a - Da*Lap(a) + kappa*a*b) - Ta*at^2 '
                    '+ bt*(Dt(b) + rb*b - Db*Lap(b) - kappa*a*b) - Tb*bt^2'),
    'equations': [
        {'lhs': '(Dt + ra - Da*Laplacian)*a', 'rhs': '0', 'population': None},
        {'lhs': '(Dt + rb - Db*Laplacian)*b', 'rhs': '0', 'population': None},
    ],
    'stability_analysis': False,
    'default_fundamental': {},
    'metadata': {},
    'boundary': {'mode': 'infinite'},
    'initial': {'mode': 'stationary'},
    'operator_ir': True,
    # NEW Dyson policy + reference diffusion (unequal-D coupled fields).
    'dyson': {'mode': 'fixed', 'order': 2},
    'reference_diffusion': 1.0,
}

run_trial('spec2_coupled_dyson', spec_coupled, {
    'dim_a':       _has_field_dim('a', 1),
    'dim_b':       _has_field_dim('b', 1),
    'n_physical':  _n_physical(2),
    'n_eq':        _n_equations(2),
    'dyson':       _dyson_is('fixed', 2),
    'reference_diffusion': _refdiff_is(1.0),
    'operator_ir': _operator_ir_is(True),
    'action':      _action_contains('kappa*a*b', 'Lap(a)', 'Lap(b)'),
})


# ════════════════════════════════════════════════════════════════════
# SPEC 3 — non-Gaussian white-noise spatial model: a phi-tilde^3 source.
#   action carries  - S3*pt^3   (an internal 3-leg noise vertex).
# ════════════════════════════════════════════════════════════════════
spec_nongauss = {
    'name': 'audit_nongaussian_noise',
    'description': 'spatial RD with phi-tilde^3 non-Gaussian noise source',
    'populations': [],
    'n_populations': 0,
    'response_fields': [],
    'physical_fields': [
        {'name': 'phi', 'spatial_dim': 1, 'description': 'density'},
    ],
    'parameters': [
        {'name': 'mu', 'default': 1.0, 'domain': 'positive'},
        {'name': 'D',  'default': 1.0, 'domain': 'positive'},
        {'name': 'T',  'default': 1.0, 'domain': 'positive'},
        {'name': 'S3', 'default': 0.2, 'domain': 'real'},
    ],
    'functions': [],
    'kernels': [],
    'cgf_terms': [],
    'action_text': 'pt*(Dt(phi) + mu*phi - D*Lap(phi)) - T*pt^2 - S3*pt^3',
    'equations': [
        {'lhs': '(Dt + mu - D*Laplacian)*phi', 'rhs': '0', 'population': None},
    ],
    'stability_analysis': False,
    'default_fundamental': {},
    'metadata': {},
    'boundary': {'mode': 'infinite'},
    'initial': {'mode': 'stationary'},
    'operator_ir': True,
}

run_trial('spec3_nongaussian', spec_nongauss, {
    'spatial_dim': _has_field_dim('phi', 1),
    'action_pt3':  _action_contains('S3*pt^3', 'Dt(phi)', 'Lap(phi)'),
    'operator_ir': _operator_ir_is(True),
    'boundary':    _boundary_is('infinite'),
})


# ════════════════════════════════════════════════════════════════════
# SPEC 4 — KPZ derivative-vertex model (Dx per-leg gradient).
#   mirrors theories/kpz_1d.theory.py; the load-spec test ALSO loads that
#   on-disk file directly to verify the hand-written file round-trips.
# ════════════════════════════════════════════════════════════════════
spec_kpz = {
    'name': 'audit_kpz_1d',
    'description': '1D KPZ per-leg gradient vertex',
    'populations': [],
    'n_populations': 0,
    'response_fields': [],
    'physical_fields': [
        {'name': 'h', 'spatial_dim': 1, 'description': 'interface height'},
    ],
    'parameters': [
        {'name': 'mu',  'default': 1.0, 'domain': 'positive'},
        {'name': 'D',   'default': 1.0, 'domain': 'positive'},
        {'name': 'lam', 'default': 0.3, 'domain': 'real'},
        {'name': 'T',   'default': 1.0, 'domain': 'positive'},
    ],
    'functions': [],
    'kernels': [],
    'cgf_terms': [],
    'action_text': 'ht*(Dt(h) + mu*h - D*Lap(h) - (lam/2)*Dx(h, 0)^2) - T*ht^2',
    'equations': [
        {'lhs': '(Dt + mu - D*Laplacian)*h', 'rhs': '0', 'population': None},
    ],
    'stability_analysis': False,
    'default_fundamental': {},
    'metadata': {},
    'boundary': {'mode': 'infinite'},
    'initial': {'mode': 'stationary'},
    'operator_ir': True,
}

run_trial('spec4_kpz', spec_kpz, {
    'spatial_dim': _has_field_dim('h', 1),
    'action_dx':   _action_contains('Dx(h, 0)^2', 'Lap(h)', 'Dt(h)'),
    'operator_ir': _operator_ir_is(True),
    'boundary':    _boundary_is('infinite'),
    'initial':     _initial_is('stationary'),
})


# ════════════════════════════════════════════════════════════════════
# SPEC 5 — periodic boundary with a length param (BC params round-trip).
# ════════════════════════════════════════════════════════════════════
spec_pbc = {
    'name': 'audit_pbc_length',
    'description': 'periodic BC with length param',
    'populations': [],
    'n_populations': 0,
    'response_fields': [],
    'physical_fields': [
        {'name': 'phi', 'spatial_dim': 1},
    ],
    'parameters': [
        {'name': 'mu', 'default': 1.0, 'domain': 'positive'},
        {'name': 'D',  'default': 1.0, 'domain': 'positive'},
        {'name': 'T',  'default': 1.0, 'domain': 'positive'},
        {'name': 'L',  'default': 20.0, 'domain': 'positive'},
    ],
    'functions': [],
    'kernels': [],
    'cgf_terms': [],
    'action_text': 'pt*(Dt(phi) + mu*phi - D*Lap(phi)) - T*pt^2',
    'equations': [
        {'lhs': '(Dt + mu - D*Laplacian)*phi', 'rhs': '0', 'population': None},
    ],
    'stability_analysis': False,
    'default_fundamental': {},
    'metadata': {},
    'boundary': {'mode': 'periodic', 'length': 'L'},
    'initial': {'mode': 'stationary'},
    'operator_ir': True,
}

run_trial('spec5_pbc_length', spec_pbc, {
    'boundary_len': _boundary_is('periodic', length='L'),
})


# ════════════════════════════════════════════════════════════════════
# DIRECT FILE TEST — load the hand-written KPZ theory file from disk and
# verify .operator_ir() survives the load->render round-trip.
# ════════════════════════════════════════════════════════════════════
kpz_path = os.path.join(ROOT, 'theories', 'kpz_1d.theory.py')
if os.path.exists(kpz_path):
    try:
        kspec = load_spec_from_file(kpz_path)
        has_opir = kspec.get('operator_ir')
        # Re-render and check whether .operator_ir() reappears in the output.
        rendered = render_theory_file(kspec)
        opir_in_render = '.operator_ir(' in rendered
        msg = (f"loaded spec operator_ir key={'operator_ir' in kspec} "
               f"value={has_opir!r}; '.operator_ir(' in re-render={opir_in_render}")
        if opir_in_render and has_opir:
            record('direct_kpz_file_operator_ir', 'pass', msg)
        else:
            record('direct_kpz_file_operator_ir', 'fail', msg)
    except Exception as e:
        record('direct_kpz_file_operator_ir', 'fail', f"raised: {e}")
else:
    record('direct_kpz_file_operator_ir', 'warn', 'kpz_1d.theory.py not found')


# ════════════════════════════════════════════════════════════════════
# Dump machine-readable summary for the harness.
# ════════════════════════════════════════════════════════════════════
print("\n==== SUMMARY JSON ====")
print(json.dumps({'results': RESULTS, 'findings': FINDINGS}, indent=2))
