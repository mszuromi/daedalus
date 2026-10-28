"""Two-population exponential Hawkes network with delayed alpha synapses, typed in the SDE front end.

    n_i ~ Poisson process with intensity a exp(v_i),
    (tau d/dt + 1) v_i = Em_i + sum_j w_ij (g * n_j)(t),     g(t) = ((t - D)/taug^2) exp(-(t - D)/taug) Theta(t - D).

The delay D enters as one keyword on the kernel. Paper parameters: W = [[-1, 1.2], [1, -1.5]], Em = -1, tau = 2,
taug = 1, a = 1. With delay=False the kernel is the plain alpha function (D = 0)."""
from daedalus.api.sde import SDE

W = [[-1.0, 1.2], [1.0, -1.5]]
EM = -1.0


def build(delay=True):
    m = (SDE('exponential Hawkes network, delayed alpha synapses')
         .population('E', size=2)
         .parameter('Em', default=[EM, EM], indexed_by=['E'])
         .parameter('tau', default=2.0, domain='positive')
         .parameter('taug', default=1.0, domain='positive')
         .parameter('a', default=1.0, domain='positive')
         .parameter('w', default=W, indexed_by=['E', 'E']))
    if delay:
        m = m.parameter('D', default=1.0, domain='positive')
    return (m.define_kernel('g', freq_image='1/(1 + I*omega*taug)^2', **({'delay': 'D'} if delay else {}))
             .field('v', population='E')
             .point_process('n', rate='a*exp(v[i])', population='E')
             .equation('v', lhs='(tau*Dt + 1)*v[i]', rhs='Em[i] + sum(w[i,j]*Conv(g, n[j]) for j in E)'))


def params(D):
    p = {'Em': [EM, EM], 'w': W}
    if D > 0:
        p['D'] = float(D)
    return p
