"""
Stage C.5a spike — the loop-momentum integral ∫dℓ for a momentum-DEPENDENT
self-energy (the φ̃φ² bubble).

The genuinely new machinery vs the Stage-C tadpole: the loop edge carries
q−ℓ, so the loop value does NOT factor out — we must do
    Σ(q,t) = ∫ dℓ/2π  G_R(ℓ,t) · C(q−ℓ,t)
with  G_R(k,t) = θ(t) e^{-m_k t},  C(k,t) = (T/m_k) e^{-m_k|t|},  m_k = μ+Dk².

This spike validates the loop integral two independent ways:
  (1) GAUSS–HERMITE — the §4c′ analytic-leaning fallback.  The exponent
      m_ℓ + m_{q−ℓ} = (2μ + Dq²/2) + 2D(ℓ−q/2)² is Gaussian in ℓ, so a GH rule
      centred at q/2 with width 1/√(2Dt) integrates it; the Lorentzian
      prefactor 1/m_{q−ℓ} is the smooth GH integrand.  Exponentially convergent.
  (2) ADAPTIVE QUAD (scipy) — the brute-force reference.

Checks: GH → quad to ~1e-10 by modest node count, across q and t; and the
equal-time limit Σ(q,0+) = ∫dℓ/2π · T/(μ+D(q−ℓ)²) = T/(2√(μD)) (q-INDEPENDENT,
= ⟨φ²⟩₀ — the q-dependence only turns on at t>0).
"""
import sys, math
sys.path.insert(0, '.')
import numpy as np
from numpy.polynomial.hermite_e import hermegauss      # ∫ e^{-x²/2} f dx rule
from scipy import integrate

mu = D = T = 1.0


def m_k(k):
    return mu + D * k * k


def integrand(ell, q, t):
    """G_R(ℓ,t) · C(q−ℓ,t) for t>0  =  e^{-m_ℓ t} · (T/m_{q−ℓ}) e^{-m_{q−ℓ} t}."""
    return np.exp(-m_k(ell) * t) * (T / m_k(q - ell)) * np.exp(-m_k(q - ell) * t)


def sigma_quad(q, t):
    val, _ = integrate.quad(lambda l: integrand(l, q, t), -np.inf, np.inf,
                            limit=200)
    return val / (2 * np.pi)


def sigma_gh(q, t, n):
    """Gauss–Hermite (probabilists') centred at ℓ=q/2, width set by the
    Gaussian part e^{-2Dt (ℓ-q/2)²}."""
    x, w = hermegauss(n)                 # ∫ e^{-x²/2} f(x) dx ≈ Σ w_i f(x_i)
    # e^{-2Dt u²} with u=ℓ-q/2  ⇒  x = u·√(4Dt) = 2√(Dt)·u ; ℓ = q/2 + x/√(4Dt)
    s = math.sqrt(4.0 * D * t)
    ell = q / 2.0 + x / s
    # remaining (non-Gaussian) factor at each node:
    #   T/m_{q-ℓ} · e^{-(2μ+Dq²/2) t}   (the Gaussian e^{-2Dt u²}=e^{-x²/2} is the weight)
    pref = math.exp(-(2 * mu + D * q * q / 2.0) * t)
    f = (T / m_k(q - ell)) * pref
    return float(np.sum(w * f) / s / (2 * np.pi))


print('=== (1) Gauss–Hermite ∫dℓ  vs  (2) adaptive quad ===')
print(f'{"q":>4} {"t":>5} {"quad":>14} {"GH n=8":>14} {"GH n=16":>14} '
      f'{"GH n=32":>14} {"rel(n=32)":>10}')
for q in [0.0, 0.7, 1.5]:
    for t in [0.1, 0.5, 1.0, 2.0]:
        ref = sigma_quad(q, t)
        g8, g16, g32 = (sigma_gh(q, t, n) for n in (8, 16, 32))
        rel = abs(g32 - ref) / max(abs(ref), 1e-30)
        print(f'{q:>4} {t:>5} {ref:>14.10f} {g8:>14.10f} {g16:>14.10f} '
              f'{g32:>14.10f} {rel:>10.1e}')

print('\n=== equal-time limit Σ(q,0+) = ⟨φ²⟩₀ = T/(2√(μD)) (q-independent) ===')
phi2_0 = T / (2 * math.sqrt(mu * D))
for q in [0.0, 0.7, 1.5, 3.0]:
    s = sigma_quad(q, 1e-6)
    print(f'  q={q:>4}: Σ(q,0+)={s:.8f}  ⟨φ²⟩₀={phi2_0:.8f}  '
          f'rel={abs(s - phi2_0) / phi2_0:.1e}')

print('\n(GH must converge to quad → the loop-momentum integrator works for a '
      'momentum-dependent\n self-energy; the q-dependence is real at t>0 and '
      'vanishes at t=0.)')
