"""
Stage C.5 (pivot) spike — MOMENTUM-FIRST parametric loop integral.

Validates the core of the pivot: doing ∫dℓ ANALYTICALLY (Gaussian, after a
Schwinger parameter on each correlation edge) gives the SAME bubble self-energy
as the direct numerical ∫dℓ (C.5a) — but with NO momentum-dependent poles, so
the m≥3 close-pair slow path can never fire.

Bubble self-energy (φ̃φ²):  Σ(q,τ) = ∫dℓ/2π G_R(ℓ,τ) C(q−ℓ,τ),  τ>0,
with G_R(k,t)=θ(t)e^{−m_k t}, C(k,t)=(T/m_k)e^{−m_k|t|}, m_k=μ+Dk².

Schwinger the correlation edge: C(q−ℓ,τ)=T∫_τ^∞ ds e^{−m_{q−ℓ} s}.  Then every
edge is e^{−(μ+Dk_e²)w_e} (response w=τ, correlation w=s), the exponent is
quadratic in ℓ, and ∫dℓ/2π e^{−D·U·(ℓ−…)²}=1/√(4πDU).  For this bubble the
Symanzik forms are U=τ+s, W−V²/U=sτ/(τ+s), giving

    Σ(q,τ) = T/√(4πD) ∫_τ^∞ ds  e^{−μ(τ+s) − Dq²·sτ/(τ+s)} / √(τ+s) .

(General routing: k_e = a_e ℓ + b_e q ⇒ U=Σ a_e² w_e, V=Σ a_e b_e w_e,
W=Σ b_e² w_e; ∫dℓ → e^{−Dq²(W−V²/U)}/√(4πDU).)
"""
import sys, math
sys.path.insert(0, '.')
import numpy as np
from scipy import integrate

mu = D = T = 1.0


def m_k(k):
    return mu + D * k * k


# ── direct numerical ∫dℓ (the C.5a reference) ─────────────────────
def sigma_direct(q, tau):
    f = lambda l: math.exp(-m_k(l) * tau) * (T / m_k(q - l)) * math.exp(-m_k(q - l) * tau)
    v, _ = integrate.quad(f, -np.inf, np.inf, limit=200)
    return v / (2 * np.pi)


# ── momentum-FIRST: ∫dℓ done analytically, one Schwinger integral left ──
def sigma_pf(q, tau):
    def f(s):
        U = tau + s                       # Symanzik U
        FF = s * tau / (tau + s)          # W − V²/U
        return math.exp(-mu * (tau + s) - D * q * q * FF) / math.sqrt(U)
    v, _ = integrate.quad(f, tau, np.inf, limit=200)
    return T / math.sqrt(4 * np.pi * D) * v


print('=== momentum-first parametric  vs  direct ∫dℓ  (bubble self-energy) ===')
print(f'{"q":>5} {"τ":>5} {"direct ∫dℓ":>16} {"momentum-first":>16} {"rel":>10}')
worst = 0.0
for q in [0.0, 0.7, 1.5, 3.0]:
    for tau in [0.1, 0.3, 0.7, 1.5, 3.0]:
        a = sigma_direct(q, tau)
        b = sigma_pf(q, tau)
        rel = abs(a - b) / max(abs(a), 1e-30)
        worst = max(worst, rel)
        print(f'{q:>5} {tau:>5} {a:>16.10f} {b:>16.10f} {rel:>10.1e}')
print(f'\nworst rel = {worst:.1e}')

# equal-time limit: Σ(q,0+) = ⟨φ²⟩₀ = T/(2√(μD)), q-independent
print('\n=== equal-time Σ(q,0+) = ⟨φ²⟩₀ (q-independent) ===')
phi2_0 = T / (2 * math.sqrt(mu * D))
for q in [0.0, 1.0, 3.0]:
    s = sigma_pf(q, 1e-7)
    print(f'  q={q}: Σ_pf(q,0+)={s:.8f}  ⟨φ²⟩₀={phi2_0:.8f}  '
          f'rel={abs(s - phi2_0) / phi2_0:.1e}')

print('\n(match → the momentum-first parametric reproduces the loop integral '
      'with NO\n momentum-dependent poles — the close-pair slow path cannot '
      'arise.  This is the\n core of the general integrator.)')
