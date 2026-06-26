"""
theories/quadratic_hawkes_alpha.theory.py
=========================================
Nonlinear (quadratic) Hawkes process with an ALPHA-FUNCTION synaptic kernel.

A population of ``E`` units carries a spike train ``n`` driven through a
**quadratic** transfer of the synaptic voltage ``v``:

    n_i(t)  = φ(v_i)        with   φ(v) = a_i · v²        (quadratic Hawkes)
    τ_i v̇_i = −(v_i − E_i) + Σ_j w_ij (g_ij * n_j)        (synaptic drive)

and the coupling is filtered by the **alpha function**

    g_ij(t) = (t / τg_ij²) · e^{−t/τg_ij} · Θ(t)            (∫g = 1, peak at t=τg)

— a kernel that *rises then decays* (unlike a bare exponential), the standard
shape for a conductance-based synapse.  This is the quadratic-transfer sibling
of :mod:`single_population_cubic_alpha_test` (same α-kernel, cubic → quadratic
transfer), so it exercises the same colored-in-time synaptic-filter machinery
with a softer (quadratic) nonlinearity.

MSR–JD response fields ``nt``/``vt`` are added automatically; the spike
nonlinearity enters through ``(exp(nt)-1)·φ(v)`` (the Hawkes/​point-process
generating term), and the alpha kernel is a genuine non-Markovian convolution
``g * n`` handled by the pipeline's kernel machinery.
"""
from api.theory import TemporalTheoryBuilder


def build():
    return (
        TemporalTheoryBuilder('Quadratic Hawkes (alpha-kernel)')
        .population('E', size=2, description='Excitatory')
        .physical_field('n', population='E', description='spike train')
        .physical_field('v', population='E', description='synaptic voltage')
        .parameter('Em', default=[0.8, 0.78], indexed_by=['E'], domain='positive')
        .parameter('tau', default=[10, 9], indexed_by=['E'], domain='positive')
        .parameter('taug', default=[
            [2, 3],
            [1, 3],
        ], indexed_by=['E', 'E'], domain='positive')
        .parameter('a', default=[0.44, 0.44], indexed_by=['E'], domain='positive')
        .parameter('w', default=[
            [0.25, 0.25],
            [0.2, 0.3],
        ], indexed_by=['E', 'E'], domain='positive')
        .define_function('phi', args=['v'], expression='a[i]*v^2', population='E')
        .define_kernel('g', time_expr='(t/taug[i,j]^2)*exp(-t/taug[i,j])*heaviside(t)',
                       latex_name='g', indexed_by=['E', 'E'])
        .set_action_text('''
            sum( nt[i]*n[i]
            - (exp(nt[i])-1)*phi[i](v[i])
            + vt[i]*((tau[i]*Dt + 1)*v[i] - Em[i]
            - sum(w[i,j]*g[i,j]*n[j] for j in E))
            for i in E)
        ''')
        .set_mf_equation('vstar', '(Em[i] + sum(w[i, j]*g[i,j]*nstar[j] for j in E))')
        .set_mf_equation('nstar', 'phi[i](vstar[i])')
        .build()
    )


DEFAULT_FUNDAMENTAL = {}


METADATA = {
    'k_default': 2,
    'ell_default': 0,
    'recommended_external_fields': [('n', 1), ('n', 2)],
    'tau_max': 20.0,
    'tau_step': 2.5,
    'description': 'Quadratic Hawkes with alpha-function synaptic kernel.',
}
