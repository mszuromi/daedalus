"""NEAT-RN (Freitas et al., arXiv:2605.26267): two subthreshold CMOS inverters joined by a capacitance matrix,
typed in the SDE front end as the exact Markov jump process of the node charges.

Fields u = v_1 - v_1*, w = v_2 - v_2* (units of V_T; time in C V_T / I0p). Node i gains or loses one electron
charge at the EKV subthreshold rates; each jump moves both voltages by +-s b_i, with s = C_T/C and b_i the columns of
C C_Max^{-1} = [[b1, b2], [b2, b1]]. Four state-dependent Poisson sources, no Gaussian noise, zero drift: the drift
comes out of the jump means. Rates, with cp_i = (ga_i + al_i - be_i)/2 and cn_i = (ga_i - al_i + be_i)/2:
    up_i: (cp_i + be_i e^{-x_i})/s,     down_i: (al_i e^{x_i} + cn_i)/s."""
from daedalus.api.sde import SDE
from daedalus.api.noise import Poisson

PARAMS = ['al1', 'be1', 'ga1', 'al2', 'be2', 'ga2', 's', 'b1', 'b2']


def build():
    m = SDE('NEAT-RN CMOS inverter pair')
    for p in PARAMS:
        m = m.parameter(p, default=1.0, domain='positive')
    cp1, cn1 = '((ga1+al1-be1)/2)', '((ga1-al1+be1)/2)'
    cp2, cn2 = '((ga2+al2-be2)/2)', '((ga2-al2+be2)/2)'
    return (m.field('u').field('w').drift('u', '0').drift('w', '0')
             .noise('up1', Poisson(rate=f'({cp1} + be1*exp(-u))/s', size='s'), couples={'u': 'b1', 'w': 'b2'})
             .noise('dn1', Poisson(rate=f'(al1*exp(u) + {cn1})/s', size='-s'), couples={'u': 'b1', 'w': 'b2'})
             .noise('up2', Poisson(rate=f'({cp2} + be2*exp(-w))/s', size='s'), couples={'u': 'b2', 'w': 'b1'})
             .noise('dn2', Poisson(rate=f'(al2*exp(w) + {cn2})/s', size='-s'), couples={'u': 'b2', 'w': 'b1'}))


def params_of(P):
    """Package parameter values for a neat.point() operating point."""
    return dict(al1=float(P['alpha'][0]), be1=float(P['beta'][0]), ga1=float(P['cp'][0] + P['cn'][0]),
                al2=float(P['alpha'][1]), be2=float(P['beta'][1]), ga2=float(P['cp'][1] + P['cn'][1]),
                s=float(P['s']), b1=float(P['B'][0, 0]), b2=float(P['B'][0, 1]))
