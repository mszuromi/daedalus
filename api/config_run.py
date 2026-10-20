"""api/config_run.py — run Daedalus from a YAML config file (``daedalus run config.yaml``).

A thin entry point over ``daedalus.run`` / ``api.compute_cumulants``: it parses
and strictly validates a config, loads the model, resolves every default, calls
the existing pipeline, and writes the requested outputs with the existing
writers (``save_npz`` / ``save_csv`` / ``plot_cumulant`` / ``generate_report``).

    sage -python -m daedalus run config.yaml [--dry-run]

Schema (all sections but ``model`` are optional; unknown keys are errors)::

    model: ou_quartic                 # name of models/<name>.model.py, or a path
                                      # (or an inline ``sde:`` block instead, below)
    parameters: {mu: 1.0, eps: 0.02}  # numbers, or (nested) lists of numbers
    question:
      k: 2
      max_ell: 1
      external_fields: [[x, 1], [x, 1]]          # [field, population index]
      tau_grid: {start: -4, stop: 4, num: 17}    # or an explicit list
      points: [...]                              # spatial k >= 3 only
      taylor_order: 4
      accuracy: 1e-10                            # target relative accuracy (G6)
      delay: {order: 8}                          # delay models (api.delay options)
    policy: {use_cache: true, parallel: false, n_workers: 4, verbose: false,
             mode: permissive,                   # model analyzer: or strict
             allow_taylor_truncation: true,      # consents (api.analyze.Policy)
             downgrade: [G4_NEGATIVE_VARIANCE],  # errors accepted as warnings
             preflight: true,                    # false: analyzer gates off
             backend: state_space,               # an explicit backend (api.routing)
             discard: auto}                      # true | false | auto: release diagrams after use
    outputs: {npz: out/c.npz, csv: out/c.csv, plot: out/c.png, report_pdf: out/r.pdf}
    output: results/                  # a directory or a file stem: .npz/.csv/.png

An inline SDE (instead of ``model:``; built with ``api.sde.SDE``, so the
noise and the action are derived exactly as for a Python-built model)::

    sde:
      name: OU quartic
      parameters:                     # default, or {default, domain, ...}
        mu: {default: 1.0, domain: positive}
        eps: 0.02
        D: {default: 1.0, domain: positive}
      fields: [x]
      drift: {x: "-mu*x - eps*x^3"}   # or equations: {x: {lhs: ..., rhs: ...}}
      noise:                          # type = a class of api.noise, the rest are its arguments
        xi: {type: Gaussian, var: "2*D", couples: {x: 1}}
      interpretation: ito             # or stratonovich, marcus
      series_order: 8                 # Taylor / Marcus series order of a jump source
      marcus_route: auto              # or series (marcus: exact for affine couplings)
      stability_analysis: true

A point-process model (``SDE.point_process``, kernels, delays) goes through a
model file: ``model: path/to/my.model.py`` whose ``build()`` returns
``SDE(...).build()`` (``dd.run`` also accepts the ``SDE`` object itself).

Every error names the config line it comes from.  Relative paths (model file,
outputs) resolve against the directory of the config file.  The YAML is
composed to a node tree (``yaml.compose``) so the marks are available; values
are read off that tree, with one leniency: PyYAML reads ``1e-3`` (no dot) as a
string, so a string that is a plain decimal literal is accepted as a number.
"""
from __future__ import annotations

import difflib
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

TOP_KEYS = ('model', 'sde', 'parameters', 'question', 'policy', 'output',
            'outputs')
SDE_KEYS = ('name', 'parameters', 'fields', 'drift', 'equations', 'noise',
            'interpretation', 'series_order', 'marcus_route', 'stability_analysis')
SDE_PARAMETER_KEYS = ('default', 'domain', 'description')
#: the ``type:`` of an inline noise source (classes of api.noise); a jump law
#: inside one is any ``api.noise.JumpDistribution`` subclass
from api.noise import SOURCE_TYPES as SDE_NOISE_TYPES  # noqa: E402
from engine.errors import DaedalusError

QUESTION_KEYS = ('k', 'max_ell', 'external_fields', 'tau_grid', 'chi_grid',
                 'spatial_grid', 'points', 'taylor_order', 'accuracy', 'delay')
POLICY_KEYS = ('use_cache', 'parallel', 'n_workers', 'verbose', 'preflight',
               'backend', 'discard',
               'mode', 'allow_taylor_truncation', 'allow_markov_embedding',
               'allow_numeric_fallback', 'interpretation', 'downgrade')
#: the policy keys that go to the model analyzer (``api.analyze.Policy``)
ANALYZER_POLICY_KEYS = ('mode', 'allow_taylor_truncation',
                        'allow_markov_embedding', 'allow_numeric_fallback',
                        'interpretation', 'downgrade')
OUTPUT_KEYS = ('npz', 'csv', 'plot', 'report_pdf')
#: files ``output:`` (a directory or a file stem) expands to
OUTPUT_STEM_KINDS = (('npz', '.npz'), ('csv', '.csv'), ('plot', '.png'))
OUTPUT_STEM_DEFAULT = 'cumulants'
TAU_GRID_KEYS = ('start', 'stop', 'num')

_DECIMAL = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')


class ConfigError(DaedalusError, ValueError):
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

    def expression(self, node, what):
        """An expression of the SDE block: a string, or a plain number
        (``drift: {x: 0}`` is the text ``0``)."""
        v = self.py(node)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return repr(v)
        if not isinstance(v, str) or not v.strip():
            self.fail(node, f'{what} must be an expression (a quoted string '
                            'or a number)')
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
        if 'model' not in top and 'sde' not in top:
            self.fail(self.root, "missing required key 'model' (or an inline "
                                 "'sde' block)")
        if 'model' in top and 'sde' in top:
            self.fail(top['sde'][0], "give either 'model' or 'sde', not both "
                                     f"('model' is on line "
                                     f"{top['model'][0].start_mark.line + 1})")
        raw = {'_lines': {}, 'sde': None}
        if 'sde' in top:
            raw['model'] = None
            raw['sde'] = self.sde_block(top['sde'][1])
            raw['_lines']['model'] = top['sde'][0].start_mark.line + 1
        else:
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

        raw['output'] = None
        if 'output' in top:
            raw['output'] = (self.string(top['output'][1], "'output'"),
                             top['output'][1].start_mark.line + 1)
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

    def sde_block(self, node):
        """The inline ``sde:`` block -> ``{'spec': plain dict, 'lines': {...}}``.
        Structure is checked here; the SDE builder checks the content when
        the model is built (its errors are reported at the line of the
        entry they name)."""
        items = self.mapping(node, SDE_KEYS, "'sde'")
        lines = {k: v[0].start_mark.line + 1 for k, v in items.items()}
        for need in ('fields', 'noise'):
            if need not in items:
                self.fail(node, f"'sde' needs '{need}'")
        if 'drift' not in items and 'equations' not in items:
            self.fail(node, "'sde' needs 'drift' (or 'equations')")
        spec = {}
        if 'name' in items:
            spec['name'] = self.string(items['name'][1], 'sde.name')
        fnode = items['fields'][1]
        if not isinstance(fnode, self.yaml.SequenceNode) or not fnode.value:
            self.fail(fnode, 'sde.fields must be a non-empty list of names, '
                             'e.g. [x]')
        spec['fields'] = [self.string(n, 'sde field name') for n in fnode.value]
        if 'parameters' in items:
            spec['parameters'] = {}
            for name, (knode, vnode) in self._free_mapping(
                    items['parameters'][1], 'sde.parameters').items():
                line = knode.start_mark.line + 1
                if isinstance(vnode, self.yaml.MappingNode):
                    sub = self.mapping(vnode, SDE_PARAMETER_KEYS,
                                       f'sde.parameters.{name}')
                    entry = {k: self.py(v[1]) for k, v in sub.items()}
                    if 'default' in sub:
                        entry['default'] = self.numeric_tree(
                            sub['default'][1], f'parameter {name!r} default')
                else:
                    entry = {'default': self.numeric_tree(
                        vnode, f'parameter {name!r}')}
                spec['parameters'][name] = (entry, line)
        for key in ('drift', 'equations'):
            if key in items:
                spec[key] = {}
                for fname, (knode, vnode) in self._free_mapping(
                        items[key][1], f'sde.{key}').items():
                    line = knode.start_mark.line + 1
                    if key == 'drift':
                        spec[key][fname] = (self.expression(
                            vnode, f'sde.drift.{fname}'), line)
                    else:
                        sub = self.mapping(vnode, ('lhs', 'rhs'),
                                           f'sde.equations.{fname}')
                        if 'lhs' not in sub:
                            self.fail(vnode, f'sde.equations.{fname} needs lhs')
                        spec[key][fname] = ({
                            k: self.expression(v[1], f'sde.equations.{fname}.{k}')
                            for k, v in sub.items()}, line)
        nnode = items['noise'][1]
        spec['noise'] = {}
        for nname, (knode, vnode) in self._free_mapping(
                nnode, 'sde.noise').items():
            if not isinstance(vnode, self.yaml.MappingNode):
                self.fail(vnode, f'sde.noise.{nname} must be a mapping with a '
                                 "'type' and the source's arguments")
            d = self.py(vnode)
            if d.get('type') not in SDE_NOISE_TYPES:
                self.fail(vnode, f"sde.noise.{nname}: 'type' must be one of "
                                 f'{", ".join(SDE_NOISE_TYPES)}; got '
                                 f'{d.get("type")!r}')
            if not isinstance(d.get('couples'), dict) or not d['couples']:
                self.fail(vnode, f"sde.noise.{nname}: 'couples' must be a "
                                 'mapping {field: coefficient}')
            spec['noise'][nname] = (d, knode.start_mark.line + 1)
        if 'interpretation' in items:
            v = self.string(items['interpretation'][1], 'sde.interpretation')
            if v not in ('ito', 'stratonovich', 'marcus'):
                self.fail(items['interpretation'][1],
                          'sde.interpretation must be ito, stratonovich or marcus; '
                          f'got {v!r}')
            spec['interpretation'] = v
        if 'series_order' in items:
            spec['series_order'] = self.integer(items['series_order'][1],
                                                'sde.series_order', 1)
        if 'marcus_route' in items:
            v = self.string(items['marcus_route'][1], 'sde.marcus_route')
            if v not in ('auto', 'series'):
                self.fail(items['marcus_route'][1],
                          f'sde.marcus_route must be auto or series; got {v!r}')
            spec['marcus_route'] = v
        if 'stability_analysis' in items:
            spec['stability_analysis'] = self.boolean(
                items['stability_analysis'][1], 'sde.stability_analysis')
        return {'spec': spec, 'lines': lines}

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
        if 'chi_grid' in items and 'spatial_grid' in items:
            self.fail(items['spatial_grid'][0],
                      "give one of 'chi_grid' and 'spatial_grid' (the alias), "
                      f"not both (chi_grid is on line {lines['chi_grid']})")
        for key in ('chi_grid', 'spatial_grid'):
            if key in items:
                # stored under the canonical name; the alias is spelling only
                q['chi_grid'] = self.chi_grid(items[key][1], f'question.{key}')
                lines['chi_grid'] = lines[key]
        if 'points' in items:
            q['points'] = (self.numeric_tree(items['points'][1],
                                             'question.points'),
                           items['points'][1])
        if 'accuracy' in items:
            v = self.number(items['accuracy'][1], 'question.accuracy')
            if not (math.isfinite(v) and v > 0):
                self.fail(items['accuracy'][1],
                          f'question.accuracy must be a finite number > 0; '
                          f'got {v!r}')
            q['accuracy'] = float(v)
        if 'delay' in items:
            q['delay'] = self.delay_options(items['delay'][1])
        q['_lines'] = lines
        return q

    def delay_options(self, node):
        """``question.delay``: the options of the Pade route for models with
        delays (``api.delay.DEFAULTS`` names; ``compute_cumulants(delay=)``),
        validated by ``api.delay.resolve_options``."""
        from api.delay import DEFAULTS, resolve_options
        items = self.mapping(node, tuple(DEFAULTS), "'question.delay'")
        out = {}
        for key, (_knode, vnode) in items.items():
            v = vnode.value
            if isinstance(vnode, self.yaml.ScalarNode) and _DECIMAL.match(str(v)):
                v = float(v) if any(c in str(v) for c in '.eE') else int(v)
            out[key] = v
        try:
            resolve_options(out)
        except (TypeError, ValueError) as exc:
            self.fail(node, f'question.delay: {exc}')
        return out

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

    def _range_trap(self, node, name):
        """The ``dd.Config`` rule: a 3-element LIST whose last entry is an
        integer >= 2 is the classic "I meant a range" slip (``[-5, 5, 51]``
        is 3 literal points); raise and point at the mapping form."""
        if len(node.value) != 3:
            return
        try:
            last = self.py(node.value[2])
        except Exception:
            return
        if (isinstance(last, (int, float)) and not isinstance(last, bool)
                and last >= 2 and float(last) == int(last)):
            lo, hi = (self.py(n) for n in node.value[:2])
            self.fail(node, f'{name} is a 3-element LIST, so it is read as 3 '
                            f'literal grid points {{{lo}, {hi}, {last}}} (note '
                            f'the {int(last)}!). For a RANGE of {int(last)} '
                            f'points from {lo} to {hi} write {{start: {lo}, '
                            f'stop: {hi}, num: {int(last)}}}; for exactly '
                            f'these 3 literal points write '
                            f'{{values: [{lo}, {hi}, {last}]}}.')

    def _grid(self, node, name, increasing):
        """A grid: ``{start, stop, num}`` (a range, the tuple form of
        ``dd.Config``), ``{values: [...]}`` or a list (literal points; a
        3-element list that looks like a range raises, see ``_range_trap``)."""
        short = name.split('.')[-1]
        if isinstance(node, self.yaml.MappingNode):
            items = self.mapping(node, TAU_GRID_KEYS + ('values',), name)
            if 'values' in items:
                extra = [k for k in items if k != 'values']
                if extra:
                    self.fail(node, f"{name}: 'values' cannot be combined "
                                    f"with {', '.join(extra)}")
                vnode = items['values'][1]
                if not isinstance(vnode, self.yaml.SequenceNode) or not vnode.value:
                    self.fail(vnode, f'{name}.values must be a non-empty list '
                                     'of numbers')
                return self._values(vnode, short, increasing)
            missing = [k for k in TAU_GRID_KEYS if k not in items]
            if missing:
                self.fail(node, f'{name} needs start, stop and num; '
                                f'missing: {", ".join(missing)}')
            start = self.number(items['start'][1], f'{short}.start')
            stop = self.number(items['stop'][1], f'{short}.stop')
            num = self.integer(items['num'][1], f'{short}.num', 2)
            if not start < stop:
                self.fail(node, f'{short} is inconsistent: need start < stop; '
                                f'got start={start}, stop={stop}')
            return [float(x) for x in np.linspace(start, stop, num)]
        if isinstance(node, self.yaml.SequenceNode) and node.value:
            self._range_trap(node, name)
            return self._values(node, short, increasing)
        self.fail(node, f'{name} must be a non-empty list of numbers or a '
                        'mapping {start, stop, num} (or {values: [...]})')

    def _values(self, node, short, increasing):
        vals = [float(self.number(n, f'{short} entry')) for n in node.value]
        if increasing:
            for i in range(1, len(vals)):
                if not vals[i] > vals[i - 1]:
                    self.fail(node.value[i],
                              f'{short} is inconsistent: the list must be '
                              f'strictly increasing, but {vals[i]} follows '
                              f'{vals[i - 1]}')
        return vals

    def chi_grid(self, node, name):
        """The spatial-difference grid chi = x_j - x_k, as ``dd.Config``."""
        return self._grid(node, name, increasing=False)

    def tau_grid(self, node):
        """-> list of floats (an explicit increasing list or a range)."""
        return self._grid(node, 'question.tau_grid', increasing=True)

    def policy(self, node):
        items = self.mapping(node, POLICY_KEYS, "'policy'")
        out = {}
        for key in ('use_cache', 'parallel', 'verbose', 'preflight',
                    'allow_taylor_truncation', 'allow_markov_embedding',
                    'allow_numeric_fallback'):
            if key in items:
                out[key] = self.boolean(items[key][1], f'policy.{key}')
        if 'n_workers' in items:
            out['n_workers'] = self.integer(items['n_workers'][1],
                                            'policy.n_workers', 1)
        if 'backend' in items:
            out['backend'] = self.string(items['backend'][1],
                                         'policy.backend')
        if 'discard' in items:
            node = items['discard'][1]
            v = self.py(node)
            if not (isinstance(v, bool) or v == 'auto'):
                self.fail(node, f'policy.discard must be true, false or '
                                f'auto; got {node.value!r}')
            out['discard'] = v
        choices = {'mode': ('strict', 'permissive'),
                   'interpretation': ('ito', 'stratonovich', 'marcus')}
        for key, allowed in choices.items():
            if key in items:
                v = self.string(items[key][1], f'policy.{key}')
                if v not in allowed:
                    self.fail(items[key][1], f'policy.{key} must be one of '
                                             f'{", ".join(allowed)}; got {v!r}')
                out[key] = v
        if 'downgrade' in items:
            node = items['downgrade'][1]
            if not isinstance(node, self.yaml.SequenceNode):
                self.fail(node, 'policy.downgrade must be a list of finding '
                                'codes, e.g. [G4_NEGATIVE_VARIANCE]')
            from api.analyze import FINDING_CODES, _NEVER_DOWNGRADE
            codes = []
            for n in node.value:
                c = self.string(n, 'policy.downgrade entry')
                if FINDING_CODES.get(c, ('', ''))[1] != 'error':
                    self.fail(n, f'policy.downgrade: {c!r} is not an error '
                                 'code of api.analyze.FINDING_CODES')
                if c in _NEVER_DOWNGRADE:
                    self.fail(n, f'policy.downgrade: {c!r} cannot be '
                                 'downgraded (the run cannot go past it)')
                codes.append(c)
            out['downgrade'] = codes
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
        raise RuntimeError('daedalus run needs PyYAML (import yaml); it ships with '
                           'the Daedalus conda environment') from None
    raw = _Parser(text, os.path.basename(path)).parse()
    raw['_source'] = os.path.basename(path)
    raw['_dir'] = os.path.dirname(path)
    raw['_path'] = path
    return raw


# ── model loading and the checks that need the model ──────────────────────────

def _sde_plain(block):
    """The inline ``sde`` block as plain YAML-able data (no line numbers):
    what the resolved config holds, and what ``_build_sde`` reads."""
    spec = block['spec']
    out = {}
    if 'name' in spec:
        out['name'] = spec['name']
    if 'parameters' in spec:
        out['parameters'] = {n: _plain(e) for n, (e, _) in
                             spec['parameters'].items()}
    out['fields'] = list(spec['fields'])
    for key in ('drift', 'equations'):
        if key in spec:
            out[key] = {f: _plain(v) for f, (v, _) in spec[key].items()}
    out['noise'] = {n: _plain(d) for n, (d, _) in spec['noise'].items()}
    for key in ('interpretation', 'series_order', 'marcus_route', 'stability_analysis'):
        if key in spec:
            out[key] = spec[key]
    return out


def _noise_object(desc):
    """``{type: Gaussian, var: ...}`` -> an ``api.noise`` object; mappings
    with a ``type`` inside it are jump laws."""
    from api import noise as N
    args = {}
    for key, value in desc.items():
        if key in ('type', 'couples'):
            continue
        if isinstance(value, dict) and 'type' in value:
            law = getattr(N, str(value['type']), None)
            if not (isinstance(law, type)
                    and issubclass(law, N.JumpDistribution)):
                laws = sorted(n for n in N.__all__
                              if isinstance(getattr(N, n), type)
                              and issubclass(getattr(N, n), N.JumpDistribution)
                              and n != 'JumpDistribution')
                raise ValueError(f"jump law type must be one of "
                                 f"{', '.join(laws)}; got {value['type']!r}")
            value = law(**{
                k: v for k, v in value.items() if k != 'type'})
        args[key] = value
    return getattr(N, desc['type'])(**args)


def _build_sde(raw):
    """The model dict of an inline ``sde:`` block (``api.sde.SDE``).
    Builder errors become ``ConfigError`` at the line of the entry they
    name."""
    from api.sde import SDE
    src, block = raw['_source'], raw['sde']
    spec, lines = block['spec'], block['lines']
    s = SDE(spec.get('name', 'config SDE'))

    def at(line, fn):
        try:
            return fn()
        except ConfigError:
            raise
        except Exception as e:
            raise ConfigError(src, line, f'{type(e).__name__}: {e}') from None

    for name, (entry, line) in (spec.get('parameters') or {}).items():
        at(line, lambda name=name, entry=entry: s.parameter(name, **entry))
    for f in spec['fields']:
        at(lines['fields'], lambda f=f: s.field(f))
    for f, (expr, line) in (spec.get('drift') or {}).items():
        at(line, lambda f=f, expr=expr: s.drift(f, expr))
    for f, (eq, line) in (spec.get('equations') or {}).items():
        at(line, lambda f=f, eq=eq: s.equation(f, **eq))
    for name, (desc, line) in spec['noise'].items():
        def add(name=name, desc=desc):
            s.noise(name, _noise_object(desc), couples=desc['couples'])
        at(line, add)
    if 'interpretation' in spec:
        s.interpretation(spec['interpretation'])
    if 'series_order' in spec:
        s.series_order(spec['series_order'])
    if 'marcus_route' in spec:
        s.marcus_route(spec['marcus_route'])
    if 'stability_analysis' in spec:
        s.stability_analysis(spec['stability_analysis'])
    return at(lines.get('drift') or lines.get('equations'), s.build)


def _load_model(raw):
    """(model, module) exactly as ``dd.load_model`` loads a model file;
    ``(model, None)`` for an inline ``sde:`` block."""
    import daedalus as dd
    if raw.get('sde'):
        return _build_sde(raw), None
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


def _expected_shape(model, spec):
    """The shape a parameter's value must have: ``()`` for a scalar
    parameter, the population sizes of ``indexed_by`` for an indexed one,
    ``None`` when a size cannot be read off the model."""
    if not spec.get('indexed'):
        return ()
    sizes = {p.get('name'): int(p.get('size') or 1)
             for p in (model.get('populations') or [])}
    for nm, idx in (model.get('index_sets') or {}).items():
        sizes.setdefault(nm, len(idx))
    shape = []
    for nm in spec.get('indexed_by') or ():
        if nm not in sizes:
            return None
        shape.append(sizes[nm])
    return tuple(shape) if shape else None


def _check_parameter_shapes(src, model, parameters):
    """Each written parameter value must be rectangular and, for a list,
    have the shape the model declares (population sizes); a list for a
    scalar parameter is an error.  Line-numbered like every other check."""
    specs = {p.get('name'): p for p in (model.get('parameters') or [])}
    for name, (value, line) in parameters.items():
        spec = specs.get(name)
        if spec is None:
            continue
        try:
            shape = np.asarray(value, dtype=float).shape
        except ValueError:
            raise ConfigError(src, line, f'parameter {name!r}: nested lists '
                                         'must be rectangular') from None
        want = _expected_shape(model, spec)
        if want == () and shape != ():
            raise ConfigError(
                src, line, f'parameter {name!r} is a scalar parameter; got a '
                           f'list of shape {shape}')
        if want and shape != want:
            by = ', '.join(spec.get('indexed_by') or ())
            got = 'a scalar' if shape == () else f'shape {shape}'
            raise ConfigError(
                src, line, f'parameter {name!r} has shape {want} '
                           f'(indexed by {by}); got {got}')


def _output_plan(spec, base):
    """``output:`` -> ({kind: path}, prefix).  A directory (trailing
    separator, or an existing directory) gets ``<dir>/cumulants.*`` and
    ``prefix`` ''; a file stem ``dir/run1`` gets ``dir/run1.*`` and
    ``prefix`` ``'run1.'`` (so two runs sharing a directory keep their own
    ``config_resolved.yaml`` / ``run_info.json``)."""
    path, line = spec
    path = os.path.expanduser(path)
    ab = os.path.normpath(path if os.path.isabs(path)
                          else os.path.join(base, path))
    if os.path.splitext(path.rstrip('/' + os.sep))[1].lower() in (
            '.npz', '.csv', '.png', '.pdf', '.yaml', '.json'):
        raise ValueError(f"'output' is a directory or a file stem without an "
                         f"extension; got {path!r} (use 'outputs' for "
                         'explicit files)')
    if path.endswith(('/', os.sep)) or os.path.isdir(ab):
        stem, prefix = os.path.join(ab, OUTPUT_STEM_DEFAULT), ''
    else:
        stem, prefix = ab, os.path.basename(ab) + '.'
    return {kind: stem + ext for kind, ext in OUTPUT_STEM_KINDS}, prefix


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

    _check_parameter_shapes(src, model, raw['parameters'])

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
            # spatial callers label the legs (phi, 1), (phi, 2): the second
            # entry is a slot, not a population (as in api.analyze G0)
            if idx > npop and not spatial:
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

    chi = q.get('chi_grid')
    if chi is not None and not spatial:
        raise ConfigError(src, qlines['chi_grid'],
                          'question.chi_grid applies only to a spatial model '
                          '(this one is temporal)')

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

    if spatial:
        # what dd.run does with the grids, so no key is silently ignored
        def unused(key, why):
            raise ConfigError(src, qlines[key], f'question.{key} is ignored '
                                                f'({why}); remove it')
        if k == 1:
            for key in ('tau_grid', 'chi_grid'):
                if key in q:
                    unused(key, 'a spatial k=1 run is the mean-field value, '
                                'without a grid')
        elif k >= 3 and points is not None:
            for key in ('tau_grid', 'chi_grid'):
                if key in q:
                    unused(key, 'the events come from question.points')
        elif k >= 3:
            at = qlines.get('k', raw['_lines']['model'])
            if chi is None:
                raise ConfigError(
                    src, at, 'a spatial model with k >= 3 needs question.points '
                             '(the (n_points, k-1, 2) events) or question.chi_grid '
                             '(+ optional tau_grid) to sweep the full grid')
            n_ev = (len(chi) * len(tau or [0.0])) ** (k - 1)
            if n_ev > 20000:
                raise ConfigError(
                    src, qlines['chi_grid'],
                    f'k={k} grid would be {n_ev} events = (len(chi_grid) * '
                    f'len(tau_grid))^(k-1); more than 20000 (dd.run refuses); '
                    'shrink the grids or give question.points')

    layered = dd.fundamental_from_model(model)
    layered.update(getattr(mod, 'DEFAULT_FUNDAMENTAL', {}) or {})
    layered.update(overrides)
    params = {n: _plain(v) for n, v in layered.items()
              if not valid or n in valid}

    policy = {'use_cache': True, 'parallel': False, 'verbose': False}
    policy.update(raw['policy'])             # n_workers only when given

    def _abs(p):
        p = os.path.expanduser(p)
        return os.path.normpath(p if os.path.isabs(p) else os.path.join(base, p))

    question = {'k': int(k), 'max_ell': int(max_ell),
                'external_fields': [[n, int(i)] for n, i in ext]}
    if tau is not None:
        question['tau_grid'] = tau
    if chi is not None:
        question['chi_grid'] = chi
    if points is not None:
        question['points'] = points
    if 'taylor_order' in q:
        question['taylor_order'] = q['taylor_order']
    if 'accuracy' in q:
        question['accuracy'] = q['accuracy']
    if 'delay' in q:
        question['delay'] = dict(q['delay'])
    outputs = {}
    if raw.get('output'):
        try:
            outputs.update(_output_plan(raw['output'], base)[0])
        except ValueError as e:
            raise ConfigError(src, raw['output'][1], str(e)) from None
    outputs.update({k_: _abs(v) for k_, v in raw['outputs'].items()})
    head = {}
    if raw.get('sde'):
        head['sde'] = _sde_plain(raw['sde'])
    else:
        model_ref = raw['model']
        if model_ref.endswith('.py') or '/' in model_ref or os.sep in model_ref:
            model_ref = _abs(model_ref)
        head['model'] = model_ref
    return {**head, 'parameters': params, 'question': question,
            'policy': policy, 'outputs': outputs}


# ── running ───────────────────────────────────────────────────────────────────

def _git_commit(root):
    try:
        out = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _grid_out(vals):
    """A grid for the resolved YAML: a 3-element list that ends in an integer
    >= 2 would be read back as the "range slip" (``dd.Config`` rule), so it is
    written as ``{values: [...]}``."""
    if (len(vals) == 3 and float(vals[2]) >= 2
            and float(vals[2]) == int(vals[2])):
        return {'values': list(vals)}
    return list(vals)


def _dump_yaml(obj):
    import yaml
    if isinstance(obj.get('question'), dict):
        obj = {**obj, 'question': {
            k: (_grid_out(v) if k in ('tau_grid', 'chi_grid') else v)
            for k, v in obj['question'].items()}}
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


def dry_run_report(resolved, model):
    """The analyzer report (gates G0-G5, no diagram enumeration, no Phase
    J) for a resolved config, with the route the router would pick, or
    ``None`` when the config turns the analyzer off (``preflight: false``)."""
    from dataclasses import asdict, replace
    from api import routing
    from api.analyze import Policy, Question, analyze
    pol = resolved['policy']
    if pol.get('preflight') is False:
        return None
    q = resolved['question']
    question = Question(
        k=q['k'], max_ell=q['max_ell'],
        external_fields=tuple(tuple(e) for e in q['external_fields']),
        parameters=dict(resolved['parameters']),
        taylor_order=q.get('taylor_order'))
    policy = Policy(**{key: pol[key] for key in ANALYZER_POLICY_KEYS
                       if key in pol})
    rep = analyze(model, question, policy, use_cache=pol.get('use_cache', True))
    try:
        route = routing.select(routing.RouteRequest(
            model_name=rep.traits.name, spatial=bool(rep.traits.spatial),
            k=q['k'], max_ell=q['max_ell'],
            # dd.run supplies the default chi grid for spatial k = 2 and
            # builds the event points for the other spatial k
            has_spatial_grid=bool(rep.traits.spatial) and (
                q.get('chi_grid') is not None or q['k'] == 2),
            has_spatial_points=bool(rep.traits.spatial) and q['k'] != 2,
            traits=asdict(rep.traits)))
        rep = replace(rep, route=f'{route.name} (predicted)')
    except routing.NoBackendError as e:
        rep = replace(rep, route='none applies: '
                      + str(e).strip().splitlines()[0])
    except Exception:                       # the report is advice, not a gate
        pass
    return rep


def run_config(path, dry_run=False, out=None):
    """Run the config at ``path``.  Returns the resolved config (dry run) or
    ``(result, resolved)``.  A dry run loads and validates the model, prints
    the resolved config and the analyzer report table (gates G0-G5: the
    action is expanded and the mean field solved, but no diagram is
    enumerated and no integral is done); it writes nothing."""
    out = out if out is not None else sys.stdout
    t0 = time.perf_counter()
    resolved, model, mod, overrides = load_config(path)
    # the output plan is fixed now, before any output exists (a directory
    # that the run itself creates must not change the stem/dir decision)
    out_spec = parse_config(path).get('output')
    prefix = (_output_plan(out_spec, os.path.dirname(os.path.abspath(path)))[1]
              if out_spec else '')
    if dry_run:
        out.write(_dump_yaml(resolved))
        rep = dry_run_report(resolved, model)
        # every report line is a YAML comment, so the whole output stays a
        # valid config (scripts that parse the dry run keep working)
        out.write('\n# analyzer report (gates G0-G5; no cumulant is computed)\n')
        body = ('no analyzer (policy.preflight is false)' if rep is None
                else rep.table())
        out.write(''.join(f'# {ln}\n' for ln in body.splitlines()))
        return resolved

    import daedalus as dd
    from api.save import save_npz, save_csv
    q, pol, outs = resolved['question'], resolved['policy'], resolved['outputs']

    tau = q.get('tau_grid')
    chi = q.get('chi_grid')
    cfg = dd.Config(
        k=q['k'], max_ell=q['max_ell'],
        external_fields=[tuple(e) for e in q['external_fields']],
        parameters=overrides or None,
        tau_grid=None if tau is None else np.asarray(tau, dtype=float),
        chi_grid=None if chi is None else np.asarray(chi, dtype=float),
        spatial_points=q.get('points'),
        parallel=pol['parallel'], n_workers=pol.get('n_workers'),
        verbose=pol['verbose'],
        policy={key: pol[key] for key in ANALYZER_POLICY_KEYS if key in pol}
        or None,
        preflight=pol.get('preflight'), delay=q.get('delay'),
        taylor_order=q.get('taylor_order'), accuracy=q.get('accuracy'),
        use_cache=None if pol['use_cache'] else False,
        backend=pol.get('backend'), discard=pol.get('discard'))
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
    with open(os.path.join(run_dir, prefix + 'config_resolved.yaml'), 'w',
              encoding='utf-8') as fh:
        fh.write(_dump_yaml(resolved))
    info = {
        'daedalus_version': dd.__version__,
        'git_commit': _git_commit(dd.REPO_ROOT),
        'config_file': os.path.abspath(path),
        'resolved': resolved,
        'outputs': written,
        'wall_time_s': time.perf_counter() - t0,
        # D2: the backend that computed the numbers and the assumptions
        # ledger of the model analyzer (None with preflight: false)
        'route': (result.get('route') or {}).get('backend'),
        'assumptions': (result.get('config') or {}).get('assumptions'),
    }
    with open(os.path.join(run_dir, prefix + 'run_info.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(info, fh, indent=2)
        fh.write('\n')
    return result, resolved


# ── command line ──────────────────────────────────────────────────────────────

def main(argv=None):
    """``daedalus run config.yaml [--dry-run]``; returns the exit status."""
    import argparse
    ap = argparse.ArgumentParser(prog='daedalus')
    sub = ap.add_subparsers(dest='command', required=True)
    rp = sub.add_parser('run', help='run a YAML config file')
    rp.add_argument('config', help='path to the config YAML')
    rp.add_argument('--dry-run', action='store_true',
                    help='validate and print the resolved config and the '
                         'analyzer report (G0-G5); computes no cumulant and '
                         'writes nothing')
    args = ap.parse_args(argv)
    try:
        run_config(args.config, dry_run=args.dry_run)
    except ConfigError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    return 0
