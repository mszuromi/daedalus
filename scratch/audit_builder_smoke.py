"""
FUNCTIONAL AUDIT #1 — builder API -> .build() smoke across every new-feature path.

Run:  timeout 900 sage -python scratch/audit_builder_smoke.py
"""
import sys, traceback

PASS, FAIL = [], []

def case(name):
    def deco(fn):
        try:
            res = fn()
            PASS.append((name, res))
            print(f"[PASS] {name}  -> {res}")
        except Exception as e:
            tb = traceback.format_exc()
            FAIL.append((name, repr(e), tb))
            print(f"[FAIL] {name}  -> {e!r}")
            print(tb)
        return fn
    return deco


from pipeline.theory import TemporalTheoryBuilder, SpatialTheoryBuilder


# ---------------------------------------------------------------- TEMPORAL
@case("temporal_OU_eps_x3 (k>=3 path)")
def _():
    m = (TemporalTheoryBuilder('OU+eps x^3')
         .population('pop', size=1)
         .physical_field('x', population='pop')
         .parameter('mu', default=1.0, domain='positive')
         .parameter('eps', default=0.05, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .set_action_text('sum(xt[i]*((Dt+mu)*x[i] + eps*x[i]^3) - D*xt[i]^2 for i in pop)')
         .equation(lhs='(Dt+mu)*x[i]', rhs='-eps*x[i]^3', population='pop')
         .build())
    return f"fields={len(m['physical_fields'])} eqs={len(m['equations'])}"

@case("temporal_OU_a_x2_b_x3")
def _():
    m = (TemporalTheoryBuilder('OU+a x^2+b x^3')
         .population('pop', size=1)
         .physical_field('x', population='pop')
         .parameter('mu', default=1.0, domain='positive')
         .parameter('a', default=0.1, domain='real')
         .parameter('b', default=0.05, domain='real')
         .parameter('D', default=1.0, domain='positive')
         .set_action_text('sum(xt[i]*((Dt+mu)*x[i] + a*x[i]^2 + b*x[i]^3) - D*xt[i]^2 for i in pop)')
         .equation(lhs='(Dt+mu)*x[i]', rhs='-a*x[i]^2 - b*x[i]^3', population='pop')
         .build())
    return f"ok fields={len(m['physical_fields'])}"


# ---------------------------------------------------------- SPATIAL single RD
@case("spatial_RD_g_p2_d1 (phi^2 vertex)")
def _():
    m = (SpatialTheoryBuilder('1D RD quadratic')
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('g', default=0.3, domain='real')
         .parameter('T', default=1.0, domain='positive')
         .set_action_text('phit*((Dt + mu - D*Laplacian)*phi + g*phi^2) - T*phit^2')
         .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-g*phi^2')
         .boundary('infinite').initial('stationary')
         .build())
    sp = m.get('spatial', {})
    return f"spatial_dim={sp.get('spatial_dim')} ops={[o['name'] for o in m['operators']]}"

@case("spatial_RD_g_p2_d2 (does d=2 builder build?)")
def _():
    m = (SpatialTheoryBuilder('2D RD quadratic')
         .physical_field('phi', spatial_dim=2)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('g', default=0.3, domain='real')
         .parameter('T', default=1.0, domain='positive')
         .set_action_text('phit*((Dt + mu - D*Laplacian)*phi + g*phi^2) - T*phit^2')
         .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-g*phi^2')
         .boundary('infinite').initial('stationary')
         .build())
    return f"spatial.dim={m['spatial']['dim']} field_sdim={m['physical_fields'][0]['spatial_dim']}"


# -------------------------------------------------- SPATIAL derivative vertices
@case("spatial_KPZ (operator_ir + Dx(h,0)^2)")
def _():
    m = (SpatialTheoryBuilder('1D KPZ')
         .physical_field('h', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('lam', default=0.3, domain='real')
         .parameter('T', default=1.0, domain='positive')
         .equation(lhs='(Dt + mu - D*Laplacian)*h', rhs='0')
         .set_action_text('ht*(Dt(h) + mu*h - D*Lap(h) - (lam/2)*Dx(h, 0)^2) - T*ht^2')
         .operator_ir().boundary('infinite').initial('stationary')
         .build())
    return f"operator_ir={m.get('operator_ir')}"

@case("spatial_ModelB_conserved (Lap(phi^2))")
def _():
    m = (SpatialTheoryBuilder('1D conserved RD')
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=2.0, domain='positive')
         .parameter('g', default=0.3, domain='real')
         .parameter('T', default=1.0, domain='positive')
         .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='g*Laplacian*phi^2')
         .set_action_text('phit*(Dt(phi) + mu*phi - D*Lap(phi) - g*Lap(phi^2)) - T*phit^2')
         .operator_ir().boundary('infinite').initial('stationary')
         .build())
    return f"operator_ir={m.get('operator_ir')}"

@case("spatial_Burgers (Dx(phi^2,0) composite)")
def _():
    m = (SpatialTheoryBuilder('1D Burgers')
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('lam', default=0.3, domain='real')
         .parameter('T', default=1.0, domain='positive')
         .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='0')
         .set_action_text('phit*(Dt(phi) + mu*phi - D*Lap(phi) + (lam/2)*Dx(phi^2, 0)) - T*phit^2')
         .operator_ir().boundary('infinite').initial('stationary')
         .build())
    return f"operator_ir={m.get('operator_ir')}"


# -------------------------------------------- SPATIAL non-Gaussian noise source
@case("spatial_nonGaussian_S3_pt3 (build)")
def _():
    m = (SpatialTheoryBuilder('1D RD non-Gaussian noise')
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('T', default=1.0, domain='positive')
         .parameter('S3', default=0.2, domain='real')
         # linear RD + cubic noise source -S3*phit^3 (phi-tilde^3)
         .set_action_text('phit*((Dt + mu - D*Laplacian)*phi) - T*phit^2 - S3*phit^3')
         .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='0')
         .boundary('infinite').initial('stationary')
         .build())
    return m  # returned for the source-leg follow-up case

@case("spatial_nonGaussian_S3_pt3 (>=3-response-leg source via extract_source_types)")
def _():
    m = (SpatialTheoryBuilder('1D RD non-Gaussian noise (legcheck)')
         .physical_field('phi', spatial_dim=1)
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('T', default=1.0, domain='positive')
         .parameter('S3', default=0.2, domain='real')
         .set_action_text('phit*((Dt + mu - D*Laplacian)*phi) - T*phit^2 - S3*phit^3')
         .equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='0')
         .boundary('infinite').initial('stationary')
         .build())
    from msrjd.core.field_theory import FieldTheory
    from msrjd.core.vertices import extract_source_types
    ft = FieldTheory(m, taylor_order=4)
    ft.expand()
    stypes = extract_source_types(ft)
    leg_counts = sorted(len(s.response_legs) for s in stypes)
    has_3 = any(len(s.response_legs) >= 3 for s in stypes)
    return f"source_leg_counts={leg_counts} has_>=3_leg_source={has_3}"


# ------------------------------------------------------ COUPLED 2-field spatial
@case("coupled_2field_spatial (a,b cross + g*a^2 + dyson_order(3))")
def _():
    m = (SpatialTheoryBuilder('coupled 2-field spatial')
         .physical_field('a', spatial_dim=1)
         .physical_field('b', spatial_dim=1)
         .parameter('mua', default=1.0, domain='positive')
         .parameter('mub', default=1.5, domain='positive')
         .parameter('Da', default=1.0, domain='positive')
         .parameter('Db', default=2.0, domain='positive')
         .parameter('c', default=0.2, domain='real')
         .parameter('g', default=0.3, domain='real')
         .parameter('Ta', default=1.0, domain='positive')
         .parameter('Tb', default=1.0, domain='positive')
         .set_action_text(
             'at*((Dt + mua - Da*Laplacian)*a + c*b + g*a^2) - Ta*at^2'
             ' + bt*((Dt + mub - Db*Laplacian)*b + c*a) - Tb*bt^2')
         .equation(lhs='(Dt + mua - Da*Laplacian)*a', rhs='-c*b - g*a^2')
         .equation(lhs='(Dt + mub - Db*Laplacian)*b', rhs='-c*a')
         .reference_diffusion(1.0)
         .dyson_order(3)
         .boundary('infinite').initial('stationary')
         .build())
    sp = m['spatial']
    return (f"fields={len(m['physical_fields'])} eqs={len(m['equations'])} "
            f"dyson={sp.get('dyson')} D0={sp.get('reference_diffusion')}")


# --------------------------------------- dyson_order sweep on unequal-D coupled
@case("dyson_order_sweep_N=0,1,2,3,5 (no builder-side cap?)")
def _():
    def build_with(N):
        return (SpatialTheoryBuilder(f'coupled unequal-D dyson{N}')
                .physical_field('a', spatial_dim=1)
                .physical_field('b', spatial_dim=1)
                .parameter('mua', default=1.0, domain='positive')
                .parameter('mub', default=1.5, domain='positive')
                .parameter('Da', default=1.0, domain='positive')
                .parameter('Db', default=2.0, domain='positive')
                .parameter('c', default=0.2, domain='real')
                .parameter('Ta', default=1.0, domain='positive')
                .parameter('Tb', default=1.0, domain='positive')
                .set_action_text(
                    'at*((Dt + mua - Da*Laplacian)*a + c*b) - Ta*at^2'
                    ' + bt*((Dt + mub - Db*Laplacian)*b + c*a) - Tb*bt^2')
                .equation(lhs='(Dt + mua - Da*Laplacian)*a', rhs='-c*b')
                .equation(lhs='(Dt + mub - Db*Laplacian)*b', rhs='-c*a')
                .reference_diffusion(1.0)
                .dyson_order(N)
                .boundary('infinite').initial('stationary')
                .build())
    results = {}
    for N in (0, 1, 2, 3, 5):
        m = build_with(N)
        results[N] = m['spatial']['dyson'].get('order')
    return f"orders_accepted={results}"


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(PASS)} pass, {len(FAIL)} fail")
    for n, e, _tb in FAIL:
        print(f"  FAIL: {n}: {e}")
    sys.exit(1 if FAIL else 0)
