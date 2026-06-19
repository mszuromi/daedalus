"""
theories/dendritic_quad_soma_sigmoid.theory.py
==============================================
Two-neuron, two-compartment (soma + dendrite) point-process model with a
QUADRATIC somatic transfer and a SIGMOIDAL dendritic transfer whose output is a
genuine Bernoulli PROBABILITY in (0,1).

Voltage formulation (Markovian).  Each compartment carries a voltage that
exponentially low-pass-filters its synaptic input; the nonlinear transfers act
on the *local* voltage (as in ``quadratic_hawkes_alpha``), which keeps the
symbolic Taylor expansion tractable -- applying the sigmoid directly to a spike
convolution instead makes the expansion blow up (the sigmoid raises the
convolution to all powers).

  Somatic voltage   tauS_i vS_i' = -(vS_i - ES_i) + sum_j wSD_ij nD_j
  Dendritic voltage tauD_i vD_i' = -(vD_i - ED_i) + sum_j wDS_ij nS_j

  Soma  -- QUADRATIC rate, Poisson spikes:
        nS_i ~ Poisson( phiS(vS_i) dt ),     phiS(x) = aS_i x^2
  Dendrite -- SIGMOID probability, Bernoulli gate (one trial per somatic spike):
        nD_i ~ Binomial( nS_i, p_D_i ),      p_D_i = sigma(vD_i) = 1/(1+e^{-vD_i})

The sigmoid is what makes the dendritic output a true probability in (0,1).
Cross-neuron coupling is carried by the off-diagonal weights wSD, wDS.
"""
from pipeline.theory import TemporalTheoryBuilder


def build():
    return (
        TemporalTheoryBuilder('Dendritic (quadratic soma, sigmoid-probability dendrite)')
        .population('E', size=2, description='Excitatory')
        .physical_field('nS', population='E', description='Somatic spike train')
        .physical_field('nD', population='E', description='Dendritic spike train')
        .physical_field('vS', population='E', description='Somatic voltage')
        .physical_field('vD', population='E', description='Dendritic voltage')
        .parameter('aS', default=[0.3, 0.3], indexed_by=['E'], domain='positive',
                   description='Somatic quadratic gain')
        .parameter('ES', default=[0.5, 0.5], indexed_by=['E'], domain='real',
                   description='Somatic voltage baseline')
        .parameter('ED', default=[0.0, 0.0], indexed_by=['E'], domain='real',
                   description='Dendritic voltage baseline (sigmoid bias)')
        .parameter('tauS', default=[1.0, 1.0], indexed_by=['E'], domain='positive',
                   description='Somatic voltage time constant')
        .parameter('tauD', default=[1.0, 1.0], indexed_by=['E'], domain='positive',
                   description='Dendritic voltage time constant')
        .parameter('wSD', default=[[0.1, 0.03], [0.03, 0.1]], indexed_by=['E', 'E'], domain='real',
                   description='dendrite -> soma weights')
        .parameter('wDS', default=[[0.1, 0.03], [0.03, 0.1]], indexed_by=['E', 'E'], domain='real',
                   description='soma -> dendrite weights')
        .set_action_text('''
            sum( nSt[i]*nS[i] + nDt[i]*nD[i]
            + vSt[i]*((tauS[i]*Dt + 1)*vS[i] - ES[i] - sum(wSD[i,j]*nD[j] for j in E))
            + vDt[i]*((tauD[i]*Dt + 1)*vD[i] - ED[i] - sum(wDS[i,j]*nS[j] for j in E))
            - (exp(nSt[i])-1)*aS[i]*vS[i]^2
            - nS[i]*log(1 + (exp(nDt[i])-1)*(1/(1+exp(-vD[i]))))
            for i in E)
        ''')
        .set_mf_equation('vSstar', 'ES[i] + sum(wSD[i,j]*nDstar[j] for j in E)')
        .set_mf_equation('vDstar', 'ED[i] + sum(wDS[i,j]*nSstar[j] for j in E)')
        .set_mf_equation('nSstar', 'aS[i]*vSstar[i]^2')
        .set_mf_equation('nDstar', 'nSstar[i]*(1/(1+exp(-vDstar[i])))')
        .build()
    )


DEFAULT_FUNDAMENTAL = {}

METADATA = {
    'k_default': 2,
    'ell_default': 0,
    'recommended_external_fields': [('nS', 1), ('nS', 2)],
    'tau_max': 20.0,
    'tau_step': 2.5,
    'description': 'Two-neuron dendritic model: quadratic soma, sigmoid-probability dendrite (voltage form).',
}
