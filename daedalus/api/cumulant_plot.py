"""
daedalus.api.cumulant_plot
==========================
Plots of a cumulant with its shot-noise δ components (Feature B):
``dd.plot_cumulant(result, show_delta=True, delta_style='arrow'|'bin')``.

* k = 2: the smooth part per loop order (cumulative: tree, tree+1loop, ...)
  and the spikes, either as arrows at their location with the weight in the
  label ('arrow'), or in the bin convention of a simulation estimator
  ('bin': ``weight / Δτ`` added to the grid bin that holds the spike, the
  curve then drawn through it).  A ``sim`` overlay ({'tau', 'C', 'C_err'})
  is a binned estimate, so it includes the spike in its τ = 0 bin; compare
  it with the 'bin' style.
* k = 3: the δ hyperplanes as lines in the plane of the relative times
  (τ1, τ2) = (t_2 − t_1, t_3 − t_1), point spikes as markers, and a second
  panel with each line's weight density along it.

Serial, Matplotlib only (no multiprocessing).
"""

from __future__ import annotations

import numpy as np

_ORDER_COLORS = ['#3F00FF', '#1F9FCC', '#16A085', '#E67E22', '#E74C3C',
                 '#8E44AD', '#7F8C8D']


def _order_label(ell):
    return 'tree' if ell == 0 else 'tree+' + '+'.join(
        f'{e}loop' for e in range(1, ell + 1))


def plot_cumulant_delta(result, cfg=None, model=None, sim=None,
                        delta_style='arrow', ax=None):
    """Plot ``result`` (a ``compute_cumulants`` / ``dd.run`` result) with its
    δ components; returns the Matplotlib ``Figure``.  See the module
    docstring for the two styles and the k = 3 layout."""
    import matplotlib.pyplot as plt
    from daedalus.api.cumulant import Cumulant
    if delta_style not in ('arrow', 'bin'):
        raise ValueError(f"delta_style must be 'arrow' or 'bin', not "
                         f"{delta_style!r}")
    c = Cumulant.from_result(result)
    title = getattr(cfg, 'title', None) if cfg is not None else None
    name = ((model or result.get('_model') or {}).get('name')
            or (result.get('config') or {}).get('model_name', ''))
    if c.k == 2:
        return _plot_k2(c, sim, delta_style, ax, title or f'{name}: C(τ)',
                        cfg)
    if c.k == 3:
        return _plot_k3(c, title or f'{name}: δ components of the k=3 '
                        f'cumulant', cfg)
    raise NotImplementedError(f'show_delta is implemented for k = 2 and 3, '
                              f'not k = {c.k}')


def _plot_k2(c, sim, style, ax, title, cfg):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=getattr(cfg, 'figsize', None)
                               or (7.5, 4.6))
    else:
        fig = ax.figure
    tau = c.tau
    order = np.argsort(tau)
    ts = tau[order]
    ells = sorted(c.smooth_by_ell)
    wts = c.delta_weights_by_ell()
    cum_smooth = np.zeros(tau.shape, dtype=complex)
    cum_w = {}
    for i, ell in enumerate(ells):
        col = _ORDER_COLORS[i % len(_ORDER_COLORS)]
        v = c.smooth_by_ell.get(ell)
        if v is not None:
            cum_smooth = cum_smooth + v
        for loc, w in (wts.get(ell) or {}).items():
            cum_w[loc] = cum_w.get(loc, 0j) + w
        lab = _order_label(ell)
        if style == 'bin':
            y = cum_smooth + _bin_deposit(tau, cum_w)
            ax.plot(ts, np.real(y)[order], '-', lw=1.6, color=col,
                    label=f'theory: {lab} (bin)')
        else:
            ax.plot(ts, np.real(cum_smooth)[order], '-', lw=1.8, color=col,
                    label=f'theory: {lab}')
    if style == 'arrow' and cum_w:
        _draw_arrows(ax, c, ells, wts, tau, cum_smooth)
    if sim is not None:
        st = np.asarray(sim['tau'])
        sc = np.real(np.asarray(sim['C']))
        se = sim.get('C_err')
        if se is not None:
            ax.errorbar(st, sc, yerr=np.asarray(se), fmt='o', ms=3,
                        color='#222', alpha=0.6, capsize=2,
                        label='sim (binned: spike in its bin)')
        else:
            ax.plot(st, sc, 'o', ms=3, color='#222', alpha=0.6,
                    label='sim (binned: spike in its bin)')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$C(\tau)$')
    ax.set_title(title + ('' if c.complete else '  [δ list incomplete]'))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save = getattr(cfg, 'save', None)
    if save:
        fig.savefig(save, dpi=130, bbox_inches='tight')
    return fig


def _bin_deposit(tau, weights):
    from daedalus.api.cumulant import _bin_deposits
    out = np.zeros(tau.shape, dtype=complex)
    for loc, w in weights.items():
        for i, f in _bin_deposits(tau, loc):
            out[i] += w * f
    return out


def _draw_arrows(ax, c, ells, wts, tau, smooth):
    """One arrow per spike location and loop order (cumulative weight in
    the label), stacked so the orders are distinguishable."""
    y = np.real(smooth)
    span = (np.max(y) - np.min(y)) if y.size else 1.0
    span = span if span > 0 else 1.0
    base = float(np.max(y)) if y.size else 0.0
    locs = sorted({loc for e in wts for loc in wts[e]})
    for loc in locs:
        y0 = float(np.interp(loc, np.sort(tau), y[np.argsort(tau)]))
        tot = 0j
        for i, ell in enumerate(ells):
            w = (wts.get(ell) or {}).get(loc)
            if w is None:
                continue
            tot += w
            col = _ORDER_COLORS[i % len(_ORDER_COLORS)]
            top = base + span * (0.35 + 0.12 * i)
            ax.annotate('', xy=(loc, top), xytext=(loc, y0),
                        arrowprops=dict(arrowstyle='-|>', color=col, lw=1.6))
            ax.plot([], [], color=col, marker=r'$\uparrow$', ls='none',
                    ms=10, label=f'δ(τ−{loc:g}) {_order_label(ell)}: '
                                 f'w = {tot.real:.6g}')
    ax.set_ylim(top=base + span * (0.45 + 0.12 * len(ells)))


def _plot_k3(c, title, cfg, extent=None, n=101):
    """k = 3: δ hyperplanes as lines in (τ1, τ2) = (t_2 − t_1, t_3 − t_1),
    point spikes as markers; right panel: weight density along each line
    (parametrised by its τ1, or τ2 for a line τ1 = const)."""
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=getattr(cfg, 'figsize', None)
                                  or (12.0, 4.8))
    if extent is None:
        tmax = float(np.max(np.abs(c.tau))) if c.tau is not None and \
            c.tau.size else 0.0
        tmax = tmax if tmax > 0 else 5.0
        extent = (-tmax, tmax)
    s = np.linspace(extent[0], extent[1], n)
    # one entry per (loop order, surface): the diagrams are summed
    merged = []
    for d in c.deltas:
        for m in merged:
            if m[0].loop_order == d.loop_order and m[0].codim == d.codim \
                    and np.allclose(m[0]._M, d._M, atol=1e-12, rtol=0) \
                    and np.allclose(m[0]._c, d._c, atol=1e-12, rtol=0):
                m.append(d)
                break
        else:
            merged.append([d])
    for i, ds in enumerate(merged):
        d = ds[0]
        col = _ORDER_COLORS[i % len(_ORDER_COLORS)]
        lab = f'{d.equality_str} ({_order_label(d.loop_order)} order, ' \
              f'{len(ds)} diagram(s))'
        if d.codim == 2:
            loc = d.location
            if loc is not None:
                w = sum(x.weight for x in ds)
                ax.plot([loc[0]], [loc[1]], 'o', color=col, ms=8,
                        label=f'{lab}: w = {w.real:.4g}')
            continue
        A, cc = d.equality
        a1, a2, a3 = A
        # with t_1 = 0: a2 τ1 + a3 τ2 + cc = 0
        if abs(a3) > 1e-15:
            t1 = s
            t2 = -(a2 * t1 + cc) / a3
            par = t1
        elif abs(a2) > 1e-15:
            t2 = s
            t1 = np.full_like(s, -cc / a2)
            par = t2
        else:
            continue
        ax.plot(t1, t2, '-', color=col, lw=1.5, label=lab)
        w = np.array([sum(x.weight_at((0.0, p, q)) for x in ds).real
                      for p, q in zip(t1, t2)])
        ax2.plot(par, w, '-', color=col, lw=1.5, label=lab)
    ax.set_xlabel(r'$\tau_1 = t_2 - t_1$')
    ax.set_ylabel(r'$\tau_2 = t_3 - t_1$')
    ax.set_xlim(*extent)
    ax.set_ylim(*extent)
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    ax.set_title(title + ('' if c.complete else '  [δ list incomplete]'))
    ax2.set_xlabel('position along the line (its τ1, or τ2 if τ1 is fixed)')
    ax2.set_ylabel('weight density')
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    save = getattr(cfg, 'save', None)
    if save:
        fig.savefig(save, dpi=130, bbox_inches='tight')
    return fig
