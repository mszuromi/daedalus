"""api/config_run.py — run Daedalus from a YAML config file (``dd run config.yaml``).

A thin entry point over ``daedalus.run`` / ``api.compute_cumulants``: it parses
and strictly validates a config, loads the model, resolves every default, calls
the existing pipeline, and writes the requested outputs with the existing
writers (``save_npz`` / ``save_csv`` / ``plot_cumulant`` / ``generate_report``).

    sage -python -m daedalus run config.yaml [--dry-run]

Schema (all sections but ``model`` are optional; unknown keys are errors)::

    model: ou_quartic                 # name of models/<name>.model.py, or a path
    parameters: {mu: 1.0, eps: 0.02}  # numbers, or (nested) lists of numbers
    question:
      k: 2
      max_ell: 1
      external_fields: [[x, 1], [x, 1]]          # [field, population index]
      tau_grid: {start: -4, stop: 4, num: 17}    # or an explicit list
      points: [...]                              # spatial k >= 3 only
      taylor_order: 4
    policy: {use_cache: true, parallel: false, n_workers: 4, verbose: false}
    outputs: {npz: out/c.npz, csv: out/c.csv, plot: out/c.png, report_pdf: out/r.pdf}

Every error names the config line it comes from.  Relative paths (model file,
outputs) resolve against the directory of the config file.  The YAML is
composed to a node tree (``yaml.compose``) so the marks are available; values
are read off that tree, with one leniency: PyYAML reads ``1e-3`` (no dot) as a
string, so a string that is a plain decimal literal is accepted as a number.
"""
from __future__ import annotations

import contextlib
import difflib
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

TOP_KEYS = ('model', 'parameters', 'question', 'policy', 'outputs')
QUESTION_KEYS = ('k', 'max_ell', 'external_fields', 'tau_grid', 'points',
                 'taylor_order')
POLICY_KEYS = ('use_cache', 'parallel', 'n_workers', 'verbose')
OUTPUT_KEYS = ('npz', 'csv', 'plot', 'report_pdf')
TAU_GRID_KEYS = ('start', 'stop', 'num')

_DECIMAL = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')


class ConfigError(ValueError):
    """A config problem; ``str(e)`` is ``<file>: line <n>: <message>``."""

    def __init__(self, source, line, message):
        self.source, self.line, self.message = source, line, message
        where = f'{source}: line {line}' if line else f'{source}'
        super().__init__(f'{where}: {message}')


# ── YAML → validated raw config (no model needed) ─────────────────────────────

class _Parser:
    """Walks the composed node tree; every check raises a line-numbered error."""

    def __init__(self, text, source):
        import yaml
        self.yaml, self.source = yaml, source
        try:
            self.root = yaml.compose(text, Loader=yaml.SafeLoader)
        except yaml.YAMLError as e:
            mark = getattr(e, 'problem_mark', None)
            problem = getattr(e, 'problem', None) or str(e)
            raise ConfigError(source, mark.line + 1 if mark else None,
                              f'invalid YAML: {problem}') from None
        if self.root is None:
            raise ConfigError(source, 1, 'the config file is empty')

    # -- helpers --
    def fail(self, node, msg):
        raise ConfigError(self.source, node.start_mark.line + 1, msg)

    def py(self, node):
        loader = self.yaml.SafeLoader('')
        try:
            return loader.construct_object(node, deep=True)
        finally:
            loader.dispose()

    def mapping(self, node, allowed, what):
        """{key: (key_node, value_node)} of a mapping node; strict keys."""
        if not isinstance(node, self.yaml.MappingNode):
            self.fail(node, f'{what} must be a mapping (key: value)')
        out = {}
        for knode, vnode in node.value:
            if not isinstance(knode, self.yaml.ScalarNode):
                self.fail(knode, f'{what}: keys must be plain names')
            key = knode.value
            if key not in allowed:
                near = difflib.get_close_matches(key, allowed, n=1)
                hint = f' (did you mean {near[0]!r}?)' if near else ''
                self.fail(knode, f'unknown key {key!r} in {what}{hint}; '
                                 f'allowed: {", ".join(allowed)}')
            if key in out:
                self.fail(knode, f'duplicate key {key!r} in {what}')
            out[key] = (knode, vnode)
        return out

    def section(self, items, key):
        """A sub-mapping's items; ``key:`` with no value is an empty section."""
        if key not in items:
            return None
        node = items[key][1]
        if isinstance(node, self.yaml.ScalarNode) and self.py(node) is None:
            return {}
        return node

    def integer(self, node, what, minimum):
        v = self.py(node)
        if isinstance(v, bool) or not isinstance(v, int):
            self.fail(node, f'{what} must be an integer; got {node.value!r}')
        if v < minimum:
            self.fail(node, f'{what} must be >= {minimum}; got {v}')
        return v

    def boolean(self, node, what):
        v = self.py(node)
        if not isinstance(v, bool):
            self.fail(node, f'{what} must be true or false; got {node.value!r}')
        return v

    def string(self, node, what):
        v = self.py(node)
        if not isinstance(v, str) or not v.strip():
            self.fail(node, f'{what} must be a non-empty string')
        return v

    def number(self, node, what):
        if not isinstance(node, self.yaml.ScalarNode):
            self.fail(node, f'{what} must be a number')
        v = self.py(node)
        if isinstance(v, str) and _DECIMAL.match(v.strip()):
            v = float(v)                       # PyYAML: '1e-3' is a string
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            self.fail(node, f'{what} must be a number; got {node.value!r}')
        if not math.isfinite(v):
            self.fail(node, f'{what} must be finite; got {node.value!r}')
        return v

    def numeric_tree(self, node, what):
        """A number, or a non-empty (nested) list of numbers."""
        if isinstance(node, self.yaml.SequenceNode):
            if not node.value:
                self.fail(node, f'{what}: an empty list is not a valid value')
            return [self.numeric_tree(n, what) for n in node.value]
        return self.number(node, what)

    # -- sections --
    def parse(self):
        top = self.mapping(self.root, TOP_KEYS, 'the config')
        if 'model' not in top:
            self.fail(self.root, "missing required key 'model'")
        raw = {'_lines': {}}
        raw['model'] = self.string(top['model'][1], "'model'")
        raw['_lines']['model'] = top['model'][1].start_mark.line + 1

        raw['parameters'] = {}
        node = self.section(top, 'parameters')
        if node:
            raw['parameters'] = {
                name: (self.numeric_tree(vnode, f'parameter {name!r}'),
                       knode.start_mark.line + 1)
                for name, (knode, vnode) in
                self._free_mapping(node, 'parameters').items()}

        raw['question'], raw['policy'], raw['outputs'] = {}, {}, {}
        node = self.section(top, 'question')
        if node:
            raw['question'] = self.question(node)
        node = self.section(top, 'policy')
        if node:
            raw['policy'] = self.policy(node)
        node = self.section(top, 'outputs')
        if node:
            items = self.mapping(node, OUTPUT_KEYS, "'outputs'")
            raw['outputs'] = {k: self.string(v[1], f'outputs.{k}')
                              for k, v in items.items()}
        return raw

    def _free_mapping(self, node, what):
        """Mapping with arbitrary (parameter) names; duplicates are errors."""
        if not isinstance(node, self.yaml.MappingNode):
            self.fail(node, f'{what} must be a mapping (name: value)')
        out = {}
        for knode, vnode in node.value:
            if not isinstance(knode, self.yaml.ScalarNode):
                self.fail(knode, f'{what}: names must be plain strings')
            name = str(knode.value)
            if name in out:
                self.fail(knode, f'duplicate parameter {name!r}')
            out[name] = (knode, vnode)
        return out

    def question(self, node):
        items = self.mapping(node, QUESTION_KEYS, "'question'")
        q, lines = {}, {}
        for key, (knode, vnode) in items.items():
            lines[key] = knode.start_mark.line + 1
        if 'k' in items:
            q['k'] = self.integer(items['k'][1], 'question.k', 1)
        if 'max_ell' in items:
            q['max_ell'] = self.integer(items['max_ell'][1],
                                        'question.max_ell', 0)
        if 'taylor_order' in items:
            q['taylor_order'] = self.integer(items['taylor_order'][1],
                                             'question.taylor_order', 1)
        if 'external_fields' in items:
            q['external_fields'] = self.external_fields(
                items['external_fields'][1])
        if 'tau_grid' in items:
            q['tau_grid'] = self.tau_grid(items['tau_grid'][1])
        if 'points' in items:
            q['points'] = (self.numeric_tree(items['points'][1],
                                             'question.points'),
                           items['points'][1])
        q['_lines'] = lines
        return q

    def external_fields(self, node):
        if (not isinstance(node, self.yaml.SequenceNode) or not node.value):
            self.fail(node, 'question.external_fields must be a non-empty '
                            'list of [field, index] pairs, e.g. [[x, 1], [x, 1]]')
        out = []
        for entry in node.value:
            if (not isinstance(entry, self.yaml.SequenceNode)
                    or len(entry.value) != 2):
                self.fail(entry, 'each external field must be a '
                                 '[field, index] pair, e.g. [x, 1]')
            name = self.string(entry.value[0], 'external field name')
            idx = self.integer(entry.value[1], 'external field index', 1)
            out.append((name, idx, entry.start_mark.line + 1))
        return out

    def tau_grid(self, node):
        """-> list of floats (an explicit list or a {start, stop, num} range)."""
        if isinstance(node, self.yaml.MappingNode):
            items = self.mapping(node, TAU_GRID_KEYS, 'question.tau_grid')
            missing = [k for k in TAU_GRID_KEYS if k not in items]
            if missing:
                self.fail(node, 'question.tau_grid needs start, stop and num; '
                                f'missing: {", ".join(missing)}')
            start = self.number(items['start'][1], 'tau_grid.start')
            stop = self.number(items['stop'][1], 'tau_grid.stop')
            num = self.integer(items['num'][1], 'tau_grid.num', 2)
            if not start < stop:
                self.fail(node, 'tau_grid is inconsistent: need start < stop; '
                                f'got start={start}, stop={stop}')
            return [float(t) for t in np.linspace(start, stop, num)]
        if isinstance(node, self.yaml.SequenceNode) and node.value:
            vals = [float(self.number(n, 'tau_grid entry')) for n in node.value]
            for i in range(1, len(vals)):
                if not vals[i] > vals[i - 1]:
                    self.fail(node.value[i],
                              'tau_grid is inconsistent: the list must be '
                              f'strictly increasing, but {vals[i]} follows '
                              f'{vals[i - 1]}')
            return vals
        self.fail(node, 'question.tau_grid must be a non-empty list of '
                        'numbers or a mapping {start, stop, num}')

    def policy(self, node):
        items = self.mapping(node, POLICY_KEYS, "'policy'")
        out = {}
        for key in ('use_cache', 'parallel', 'verbose'):
            if key in items:
                out[key] = self.boolean(items[key][1], f'policy.{key}')
        if 'n_workers' in items:
            out['n_workers'] = self.integer(items['n_workers'][1],
                                            'policy.n_workers', 1)
        return out


def parse_config(path):
    """Read and validate the YAML structure of ``path`` -> raw config dict."""
    path = os.path.abspath(path)
    try:
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
    except OSError as e:
        raise ConfigError(os.path.basename(path), None,
                          f'cannot read the config file: {e}') from None
    try:
        import yaml                                         # noqa: F401
    except ImportError:
        raise RuntimeError('dd run needs PyYAML (import yaml); it ships with '
                           'the Daedalus conda environment') from None
    raw = _Parser(text, os.path.basename(path)).parse()
    raw['_source'] = os.path.basename(path)
    raw['_dir'] = os.path.dirname(path)
    raw['_path'] = path
    return raw


# ── model loading and the checks that need the model ──────────────────────────

def _load_model(raw):
    """(model, module) exactly as ``dd.load_model`` loads a model file."""
    import daedalus as dd
    spec, src, line = raw['model'], raw['_source'], raw['_lines']['model']
    looks_like_path = (spec.endswith('.py') or '/' in spec or os.sep in spec)
    try:
        if not looks_like_path:
            return dd.load_model(spec)
        path = spec if os.path.isabs(spec) else os.path.join(raw['_dir'], spec)
        path = os.path.normpath(path)
        if not path.endswith('.model.py'):
            raise ConfigError(src, line, f'model file {spec!r} must be named '
                                         '<name>.model.py')
        if not os.path.isfile(path):
            raise ConfigError(src, line, f'no model file at {path}')
        if os.path.dirname(path) == os.path.normpath(dd.MODELS_DIR):
            return dd.load_model(os.path.basename(path)[:-len('.model.py')])
        import importlib.util
        name = os.path.basename(path)[:-len('.model.py')]
        mspec = importlib.util.spec_from_file_location(f'models.{name}', path)
        mod = importlib.util.module_from_spec(mspec)
        try:
            mspec.loader.exec_module(mod)
            return mod.build(), mod
        except Exception as e:
            raise RuntimeError(
                f'Failed to load model {name!r} from {path}: {e}') from e
    except FileNotFoundError as e:
        near = difflib.get_close_matches(spec, dd.list_models(), n=3)
        hint = f' Did you mean: {", ".join(near)}?' if near else ''
        raise ConfigError(src, line, f'unknown model {spec!r}.{hint} '
                                     f'Available: {dd.list_models()}') from e
    except RuntimeError as e:
        raise ConfigError(src, line, str(e)) from e


def _field_entry(model, name):
    """The physical-field dict whose natural or internal name is ``name``."""
    for f in model.get('physical_fields') or []:
        if name in (f.get('natural_name'), f.get('name')):
            return f
    return None


def _plain(v):
    """JSON/YAML-safe copy of a parameter value (numpy / tuples -> lists)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_plain(x) for x in (v.tolist() if isinstance(v, np.ndarray)
                                    else v)]
    if isinstance(v, (bool, str)) or v is None:
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def resolve(raw, model, mod):
    """Validate the model-dependent parts and fill in every default.

    Returns the resolved config: a plain dict that is itself a valid config
    (it is what ``config_resolved.yaml`` holds).  Raises ``ConfigError``."""
    import daedalus as dd
    src, base = raw['_source'], raw['_dir']
    q, qlines = raw['question'], raw['question'].get('_lines', {})
    spatial = dd.is_spatial(model)

    # parameters: declared names only (same rule as dd.run)
    valid = set(dd.param_names(model))
    overrides = {}
    for name, (value, line) in raw['parameters'].items():
        if valid and name not in valid:
            near = difflib.get_close_matches(name, sorted(valid), n=1)
            hint = f' (did you mean {near[0]!r}?)' if near else ''
            raise ConfigError(src, line, f'unknown parameter {name!r}{hint}; '
                                         f'declared: {sorted(valid)}')
        overrides[name] = value

    # external fields: real physical field, population index in range
    ext = None
    if 'external_fields' in q:
        ext = []
        for name, idx, line in q['external_fields']:
            f = _field_entry(model, name)
            if f is None:
                raise ConfigError(
                    src, line, f'external field {name!r} is not a physical '
                               f'field of the model; fields: '
                               f'{dd.field_names(model)}')
            npop = dd._field_pop_size(model, f.get('natural_name')
                                      or f.get('name'))
            if idx > npop:
                raise ConfigError(
                    src, line, f'external field {name!r} has {npop} '
                               f'population(s); index {idx} is out of range')
            ext.append((name, idx))

    # k, max_ell, external_fields: the same defaults dd.run applies
    k = q.get('k')
    if k is not None and ext is not None and len(ext) != k:
        raise ConfigError(
            src, qlines['external_fields'],
            f'k={k} (line {qlines["k"]}) but external_fields lists {len(ext)} '
            f'leg(s); a k-point correlator needs exactly k legs')
    if k is None:
        k = len(ext) if ext is not None else dd._meta(mod, 'k_default', 2)
    max_ell = q.get('max_ell')
    if max_ell is None:
        max_ell = dd._meta(mod, 'ell_default', 0)
    if ext is None:
        rec = dd._meta(mod, 'recommended_external_fields')
        ext = (list(rec) if dd._ext_is_valid(model, rec, k)
               else dd._auto_external_fields(model, k))

    # tau grid: explicit, else the grid compute_cumulants builds from the
    # model's tau_max / tau_step (temporal models)
    tau = q.get('tau_grid')
    if tau is None and not spatial:
        tmax = dd._meta(mod, 'tau_max', 10.0)
        tstep = dd._meta(mod, 'tau_step', 0.5)
        tau = [float(t) for t in np.arange(-tmax, tmax + tstep * 0.5, tstep)]

    # points: spatial k >= 3 only, shape (n_pts, k-1, 2)
    points = None
    if 'points' in q:
        value, node = q['points']
        line = node.start_mark.line + 1
        if not (spatial and k >= 3):
            raise ConfigError(src, qlines['points'],
                              'question.points applies only to a spatial '
                              f'model with k >= 3 (this one is '
                              f'{"spatial" if spatial else "temporal"}, '
                              f'k={k})')
        try:
            arr = np.asarray(value, dtype=float)
        except ValueError:
            arr = None
        if arr is None or arr.shape[1:] != (k - 1, 2):
            raise ConfigError(src, line, 'question.points must have shape '
                                         f'(n_points, {k - 1}, 2): '
                                         '(x_j, tau_j) per non-anchor leg')
        points = arr.tolist()

    layered = dd.fundamental_from_model(model)
    layered.update(getattr(mod, 'DEFAULT_FUNDAMENTAL', {}) or {})
    layered.update(overrides)
    params = {n: _plain(v) for n, v in layered.items()
              if not valid or n in valid}

    policy = {'use_cache': True, 'parallel': False, 'verbose': False}
    policy.update(raw['policy'])             # n_workers only when given

    def _abs(p):
        return os.path.normpath(p if os.path.isabs(p) else os.path.join(base, p))

    question = {'k': int(k), 'max_ell': int(max_ell),
                'external_fields': [[n, int(i)] for n, i in ext]}
    if tau is not None:
        question['tau_grid'] = tau
    if points is not None:
        question['points'] = points
    if 'taylor_order' in q:
        question['taylor_order'] = q['taylor_order']
    model_ref = raw['model']
    if model_ref.endswith('.py') or '/' in model_ref or os.sep in model_ref:
        model_ref = _abs(model_ref)
    return {'model': model_ref, 'parameters': params, 'question': question,
            'policy': policy,
            'outputs': {k_: _abs(v) for k_, v in raw['outputs'].items()}}


# ── running ───────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _compute_defaults(**extra):
    """Make ``dd.run`` pass extra keywords (``taylor_order``, ``use_cache``)
    that ``dd.Config`` does not carry on to ``compute_cumulants``."""
    import api
    original = api.compute_cumulants

    def wrapped(*args, **kw):
        for key, value in extra.items():
            kw.setdefault(key, value)
        return original(*args, **kw)

    api.compute_cumulants = wrapped
    try:
        yield
    finally:
        api.compute_cumulants = original


def _git_commit(root):
    try:
        out = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _dump_yaml(obj):
    import yaml
    return yaml.safe_dump(obj, sort_keys=False, default_flow_style=None,
                          width=100)


def load_config(path):
    """Parse, validate and resolve ``path`` ->
    ``(resolved, model, module, parameter_overrides)``."""
    raw = parse_config(path)
    model, mod = _load_model(raw)
    resolved = resolve(raw, model, mod)
    # the parameters the user actually wrote (dd.run layers the defaults itself)
    resolved_overrides = {n: v for n, (v, _) in raw['parameters'].items()}
    return resolved, model, mod, resolved_overrides


def run_config(path, dry_run=False, out=None):
    """Run the config at ``path``.  Returns the resolved config (dry run) or
    ``(result, resolved)``.  A dry run loads and validates the model and prints
    the resolved config; it computes nothing and writes nothing."""
    out = out if out is not None else sys.stdout
    t0 = time.perf_counter()
    resolved, model, mod, overrides = load_config(path)
    if dry_run:
        out.write(_dump_yaml(resolved))
        return resolved

    import daedalus as dd
    from api.save import save_npz, save_csv
    q, pol, outs = resolved['question'], resolved['policy'], resolved['outputs']

    tau = q.get('tau_grid')
    cfg = dd.Config(
        k=q['k'], max_ell=q['max_ell'],
        external_fields=[tuple(e) for e in q['external_fields']],
        parameters=overrides or None,
        tau_grid=None if tau is None else np.asarray(tau, dtype=float),
        spatial_points=q.get('points'),
        parallel=pol['parallel'], n_workers=pol.get('n_workers'),
        verbose=pol['verbose'])
    extra = {}
    if q.get('taylor_order') is not None:
        extra['taylor_order'] = q['taylor_order']
    if not pol['use_cache']:
        extra['use_cache'] = False
    with _compute_defaults(**extra):
        result = dd.run(model, cfg, mod)

    written = {}
    if 'npz' in outs:
        written['npz'] = save_npz(result, outs['npz'])
    if 'csv' in outs:
        written['csv'] = save_csv(result, outs['csv'])
    if 'plot' in outs:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(outs['plot']) or '.', exist_ok=True)
        fig = dd.plot_cumulant(result, cfg, model)
        fig.savefig(outs['plot'], dpi=150, bbox_inches='tight')
        plt.close(fig)
        written['plot'] = outs['plot']
    if 'report_pdf' in outs:
        from api.report import generate_report
        os.makedirs(os.path.dirname(outs['report_pdf']) or '.', exist_ok=True)
        generate_report(model, q['k'], parameters=resolved['parameters'],
                        external_fields=[tuple(e) for e in q['external_fields']],
                        output_pdf=outs['report_pdf'], max_ell=q['max_ell'],
                        result=result, verbose=pol['verbose'])
        written['report_pdf'] = outs['report_pdf']

    # config_resolved.yaml and run_info.json go next to the first output
    first = next((written[k_] for k_ in OUTPUT_KEYS if k_ in written), None)
    run_dir = os.path.dirname(first) if first else os.path.dirname(
        os.path.abspath(path))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config_resolved.yaml'), 'w',
              encoding='utf-8') as fh:
        fh.write(_dump_yaml(resolved))
    info = {
        'daedalus_version': dd.__version__,
        'git_commit': _git_commit(dd.REPO_ROOT),
        'config_file': os.path.abspath(path),
        'resolved': resolved,
        'outputs': written,
        'wall_time_s': time.perf_counter() - t0,
    }
    with open(os.path.join(run_dir, 'run_info.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(info, fh, indent=2)
        fh.write('\n')
    return result, resolved


# ── command line ──────────────────────────────────────────────────────────────

def main(argv=None):
    """``dd run config.yaml [--dry-run]``; returns the exit status."""
    import argparse
    ap = argparse.ArgumentParser(prog='daedalus')
    sub = ap.add_subparsers(dest='command', required=True)
    rp = sub.add_parser('run', help='run a YAML config file')
    rp.add_argument('config', help='path to the config YAML')
    rp.add_argument('--dry-run', action='store_true',
                    help='validate and print the resolved config; compute '
                         'nothing')
    args = ap.parse_args(argv)
    try:
        run_config(args.config, dry_run=args.dry_run)
    except ConfigError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    return 0
