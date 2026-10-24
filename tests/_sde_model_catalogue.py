"""A catalogue of SDE front-end models (api/sde.py) written with the default ``Dt``.

The declarations are taken from the SDE tests (test_sde_frontend, test_marcus, test_point_process_front_end,
test_delay_kernels, test_delay_nonlinear) and the README.  tests/test_declared_operators.py checks that their derived
action text, mean-field equations and built model dicts stay byte-identical to the snapshot in
tests/fixtures/declared_operators_sde_golden.json (written before declared operators existed), and
``digest`` is the snapshot's form of a built model dict.
"""
from api.delay import Kernel
from api.noise import (CompoundPoisson, Cumulants, Dirac, Exponential, Gaussian, GammaProcess, Poisson,
                       Uniform)
from api.sde import SDE


def _ou(name, rhs, defaults, drift_form=True, stability=None):
    s = SDE(name).population('pop', size=1).field('x', population='pop', description='variable')
    for p, (d, dom) in defaults.items():
        s.parameter(p, default=d, domain=dom)
    if drift_form:
        s.drift('x', f'-mu*x + {rhs}')
    else:
        s.equation('x', lhs='(Dt+mu)*x[i]', rhs=rhs)
    s.noise('xi', Gaussian(var='2*D'), couples={'x': 1})
    if stability is not None:
        s.stability_analysis(stability)
    return s


def ou_quartic(drift_form=True):
    return _ou('OU Quartic (white noise)', '-eps*x^3',
               {'mu': (1.0, 'positive'), 'eps': (0.02, 'positive'), 'D': (1.0, 'positive')}, drift_form)


def ou_sextic(drift_form=True):
    return _ou('OU Sextic', '-eps*x^3 - gamma*x^5',
               {'mu': (1.0, 'positive'), 'eps': (0.05, 'positive'), 'D': (1.0, 'positive'),
                'gamma': (0.05, 'positive')}, drift_form, stability=True)


def linear_hawkes():
    return (SDE('Linear Hawkes 2-pop', n_populations=2)
            .parameter('E', indexed=True, default=[0.78, 0.81])
            .parameter('tau', default=10.0, domain='positive')
            .parameter('a', default=1.0)
            .parameter('tau_g', default=2.5, domain='positive')
            .parameter('w', indexed='matrix', default=[[0.30, 0.25], [0.30, 0.35]])
            .define_function('phi', args=['v'], expression='a * v', latex=r'\varphi')
            .define_kernel('g', freq_image='1 / (1 + I*omega*tau_g)', latex_name='g')
            .noise('spikes', Poisson(rate='phi(v)'), expose='n')
            .field('v')
            .equation('v', lhs='(tau*Dt + 1)*v', rhs='E[i] + sum(w[i, j]*g*n[j] for j in pop)'))


def exp_hawkes():
    return (SDE('exp Hawkes 1-pop (SDE)')
            .parameter('Em', default=-1.5).parameter('tau', default=2.0, domain='positive')
            .parameter('taug', default=1.0, domain='positive').parameter('a', default=1.0, domain='positive')
            .parameter('w', default=0.4)
            .define_kernel('g', time_expr='exp(-t/taug)/taug*heaviside(t)')
            .noise('spikes', Poisson(rate='a*exp(v)'), expose='n')
            .field('v').equation('v', lhs='(tau*Dt + 1)*v', rhs='Em + w*g*n').stability_analysis(True))


def two_field_cross():
    s = (SDE('two-field').field('x').field('y')
         .equation('x', lhs='(Dt+mu1)*x', rhs='J1*y')
         .equation('y', lhs='(Dt+mu2)*y', rhs='J2*x - eps*x^2')
         .noise('own_x', Gaussian(var='2*D1'), couples={'x': 1})
         .noise('own_y', Gaussian(var='2*D2'), couples={'y': 1})
         .noise('common', Gaussian(var='2*r'), couples={'x': 1, 'y': 1}))
    for p in ('mu1', 'mu2', 'J1', 'J2', 'eps', 'D1', 'D2', 'r'):
        s.parameter(p, default=0.5)
    return s


def stratonovich_multiplicative():
    return (SDE('Stratonovich multiplicative').parameter('mu', default=1.0, domain='positive')
            .parameter('s0', default=0.6, domain='positive')
            .parameter('s1', default=0.2, domain='positive').field('x')
            .drift('x', '-mu*x').noise('W', Gaussian(var=1), couples={'x': 's0 + s1*x'})
            .interpretation('stratonovich').stability_analysis(True))


def stratonovich_tau_lhs():
    return (SDE('strat tau').parameter('tau', default=2.0).parameter('c0', default=1.0)
            .parameter('c1', default=0.1).field('x')
            .equation('x', lhs='(tau*Dt + 1)*x', rhs='0')
            .noise('W', Gaussian(var=1), couples={'x': 'c0 + c1*x'}).interpretation('stratonovich'))


def nonpolynomial_drift():
    return (SDE('nonpolynomial').parameter('eps', default=0.05, domain='positive').field('x')
            .drift('x', '-x/(1+eps*x^2)')
            .noise('xix', Gaussian(var='2'), couples={'x': '1 + x/5'}).stability_analysis(True))


def gaussian_mean_and_var():
    return (SDE('mv').parameter('mu', default=1.0).parameter('c', default=0.3)
            .parameter('s', default=0.4).field('x').drift('x', '-mu*x')
            .noise('W', Gaussian(mean='c*x', var='s^2*(1 + x^2)'), couples={'x': 'x'}))


def _additive(name):
    return (SDE(name).parameter('mu', default=1.0, domain='positive').parameter('lam', default=1.5)
            .parameter('w', default=0.3).parameter('m', default=0.5).field('x').drift('x', '-mu*x'))


def poisson_driven():
    return _additive('p').noise('N', Poisson('lam'), couples={'x': 'w'})


def gamma_process():
    return _additive('g').noise('G', GammaProcess('lam', 'm'), couples={'x': '-w'})


def uniform_jumps():
    return _additive('u').noise('U', CompoundPoisson('lam', Uniform(0, 'm')), couples={'x': 1}).series_order(5)


def cumulants_source():
    return _additive('cum').noise('C', Cumulants(['0', 'w', 'm']), couples={'x': 1})


def marcus_affine():
    return (SDE('affine compound Poisson').parameter('mu', default=1.0, domain='positive')
            .parameter('lam', default=1.5, domain='positive').parameter('a', default=0.4)
            .parameter('b', default=0.15).parameter('mJ', default=0.2, domain='positive')
            .field('x').drift('x', '-mu*x')
            .noise('L', CompoundPoisson('lam', Exponential('mJ')), couples={'x': 'a + b*x'})
            .interpretation('marcus').series_order(16).stability_analysis(True))


def marcus_rotation():
    return (SDE('rotation').parameter('mu', default=1.0).parameter('lam', default=1.0)
            .parameter('z', default=0.7).field('x').field('y')
            .drift('x', '-mu*x').drift('y', '-mu*y')
            .noise('L', CompoundPoisson('lam', Dirac('z')), couples={'x': 'y', 'y': '-x'})
            .interpretation('marcus').series_order(7))


def point_process_voltage():
    return (SDE('pp').parameter('tau', default=1.0, domain='positive').parameter('w', default=0.2)
            .parameter('a', default=1.0, domain='positive').field('v')
            .point_process('n', 'a')
            .equation('v', lhs='(tau*Dt + 1)*v', rhs='w*n'))


def readme_hawkes():
    return (SDE('Hawkes').population('E', size=2)
            .parameter('a', default=[1.0, 1.0], indexed_by=['E'])
            .parameter('Em', default=[-1.5, -1.5], indexed_by=['E'])
            .parameter('tau', default=2.0, domain='positive')
            .parameter('w', default=[[0.4, 0.4], [0.4, 0.4]], indexed_by=['E', 'E'])
            .define_kernel('r', kernel=Kernel.exponential('tau'))
            .point_process('n', 'a*exp(Em + sum(w[i,j]*Conv(r, n[j]) for j in E))', population='E'))


def delayed_hawkes_voltage():
    n_pop = 2
    return (SDE('Hawkes delayed alpha').population('E', size=n_pop)
            .parameter('Em', default=[-1.5] * n_pop, indexed_by=['E'])
            .parameter('tau', default=[2.0] * n_pop, indexed_by=['E'], domain='positive')
            .parameter('taug', default=[[1.0] * n_pop] * n_pop, indexed_by=['E', 'E'], domain='positive')
            .parameter('a', default=[1.0] * n_pop, indexed_by=['E'], domain='positive')
            .parameter('w', default=[[0.0, 0.5], [0.5, 0.0]], indexed_by=['E', 'E'])
            .parameter('d', default=1.5, domain='positive')
            .define_kernel('g', kernel=Kernel.delay('d') * Kernel.alpha('taug[i,j]'), indexed_by=['E', 'E'])
            .point_process('n', 'phi(v)', population='E').field('v', population='E')
            .define_function('phi', args=['v'], expression='a[i]*exp(v)', population='E')
            .equation('v', lhs='(tau*Dt + 1)*v', rhs='Em + sum(w[i,j]*Conv(g[i,j], n[j]) for j in E)'))


def delay_ou():
    return (SDE('delay OU').parameter('a', default=1.0).parameter('c', default=0.4)
            .parameter('d', default=0.5, domain='positive').parameter('eps', default=0.05)
            .parameter('D', default=0.2).field('x')
            .drift('x', '-a*x + c*x(t - d) - eps*x^3')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}))


def delay_product():
    return (SDE('delay product').parameter('a', default=1.0).parameter('b', default=1.0)
            .parameter('c', default=0.5).parameter('d', default=0.5, domain='positive')
            .parameter('D', default=0.1).field('x')
            .drift('x', '-a*x + b - c*x*x(t - d)')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}))


def kernel_conductance():
    return (SDE('kernel conductance').parameter('a', default=1.0).parameter('b', default=1.0)
            .parameter('c', default=0.3).parameter('d', default=0.5, domain='positive')
            .parameter('tau', default=0.4, domain='positive').parameter('D', default=0.05)
            .field('x').define_kernel('g', kernel=Kernel.delay('d') * Kernel.exponential('tau'))
            .drift('x', '-a*x + b - c*x*Conv(g, x)')
            .noise('xi', Gaussian(var='2*D'), couples={'x': 1}))


#: name -> factory of a fresh SDE (declared with the default ``Dt``)
CATALOGUE = {
    'ou_quartic_drift': lambda: ou_quartic(True),
    'ou_quartic_equation': lambda: ou_quartic(False),
    'ou_sextic_drift': lambda: ou_sextic(True),
    'ou_sextic_equation': lambda: ou_sextic(False),
    'linear_hawkes': linear_hawkes,
    'exp_hawkes': exp_hawkes,
    'two_field_cross': two_field_cross,
    'stratonovich_multiplicative': stratonovich_multiplicative,
    'stratonovich_tau_lhs': stratonovich_tau_lhs,
    'nonpolynomial_drift': nonpolynomial_drift,
    'gaussian_mean_and_var': gaussian_mean_and_var,
    'poisson_driven': poisson_driven,
    'gamma_process': gamma_process,
    'uniform_jumps': uniform_jumps,
    'cumulants_source': cumulants_source,
    'marcus_affine': marcus_affine,
    'marcus_rotation': marcus_rotation,
    'point_process_voltage': point_process_voltage,
    'readme_hawkes': readme_hawkes,
    'delayed_hawkes_voltage': delayed_hawkes_voltage,
    'delay_ou': delay_ou,
    'delay_product': delay_product,
    'kernel_conductance': kernel_conductance,
}


def _canon(v):
    if callable(v):
        return '<callable>'
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return repr(v)


def digest(sde):
    """The snapshot of one SDE: its action text, mean-field equations, notes and built model dict (callables as
    ``'<callable>'``; ``spec_signature`` hashes every text declaration the expansion compiles)."""
    return {'action_text': sde.action_text(), 'equations': _canon(sde.equations()),
            'notes': list(sde.notes), 'model': _canon(sde.build())}
