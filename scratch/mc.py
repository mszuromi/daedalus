import numpy as np, time, sys
from pipeline.theory import TheoryBuilder
from pipeline.compute import FieldTheory
from pipeline._propagator import build_propagator
from msrjd.integration.spatial.diagram_descriptor import diagram_to_cstack
from msrjd.integration.spatial.pipeline_bridge import build_pipeline_records,_legs_to_phys_idx,_formfactor_callable
from msrjd.integration.spatial.full_integrator import diagram_kinematic, _momentum_factor_batch
from msrjd.integration.spatial.causal_chambers import causal_chambers
from msrjd.diagrams.type_assignment import build_field_index_map
from sage.all import SR
def out(s): sys.stdout.write('@@ '+s+'\n'); sys.stdout.flush()

def mc_kinematic(descr, q_vec, external_times, mu, D, spatial_dim=1, N=100000, W=None, seed=0, formfactor=None):
    """Importance-sampled MC analogue of diagram_kinematic (q-eval, formfactor optional)."""
    edges=list(descr.edges); internal=list(descr.internal_vertices); n_V=len(internal)
    idx={v:i for i,v in enumerate(internal)}; n_C=sum(1 for e in edges if e.kind=='C')
    a=np.array([e.a for e in edges],dtype=float).reshape(len(edges),-1)
    b=np.array([e.b for e in edges],dtype=float).reshape(len(edges),-1)
    if W is None: W=22.0/mu
    ext_t=list(external_times.values()); me,mn=max(ext_t),min(ext_t); lo,hi=mn-W,me+3.0/mu
    internal_R=[]; s_up=[hi]*n_V; s_lo=[lo]*n_V
    for e in edges:
        if e.kind!='R': continue
        ui,vi=e.u in idx,e.v in idx
        if ui and vi: internal_R.append((idx[e.u],idx[e.v]))
        elif ui: s_up[idx[e.u]]=min(s_up[idx[e.u]],external_times[e.v])
        elif vi: s_lo[idx[e.v]]=max(s_lo[idx[e.v]],external_times[e.u])
    rng=np.random.default_rng(seed); chambers=causal_chambers(n_V,internal_R) if n_V else [()]
    total=0.0
    for order in chambers:
        placed={}; later=None; Sgap=np.zeros(N)
        for vi in reversed(order):
            upper=np.full(N,s_up[vi]) if later is None else np.minimum(s_up[vi],later)
            g=-np.log(rng.random(N))/mu; t=upper-g; placed[vi]=t; later=t; Sgap+=g
        sig=[-np.log(rng.random(N))/mu for _ in range(n_C)]
        tvals={leaf:np.full(N,tt) for leaf,tt in external_times.items()}
        for v in internal: tvals[v]=placed[idx[v]]
        w_batch=np.empty((N,len(edges))); mu_resid=np.zeros(N); ci=0
        for ei,e in enumerate(edges):
            tu,tv=tvals[e.u],tvals[e.v]
            if e.kind=='R': w_batch[:,ei]=np.maximum(tv-tu,1e-12); mu_resid+=np.maximum(tv-tu,0.0)
            else: dt=np.abs(tu-tv); w_batch[:,ei]=dt+sig[ci]; mu_resid+=dt; ci+=1
        wgt=np.exp(-mu*(mu_resid-Sgap))/(mu**(n_V+n_C))
        momfac=_momentum_factor_batch(a,b,w_batch,q_vec,D,spatial_dim)
        total+=np.sum(wgt*momfac)/N
    return total

# ---- 1-loop KPZ diagram: MC vs grid ground truth ----
b=(TheoryBuilder('kpz',n_populations=0).physical_field('h',spatial_dim=1)
  .parameter('mu',default=1.0,domain='positive').parameter('D',default=1.0,domain='positive')
  .parameter('c',default=0.3,domain='real').parameter('T',default=1.0,domain='positive')
  .equation(lhs='(Dt+mu-D*Laplacian)*h',rhs='0').set_action_text('ht*(Dt(h)+mu*h-D*Lap(h)-(c/2)*Dx(h,0)^2)-T*ht^2').operator_ir().boundary('infinite').initial('stationary').build())
ft=FieldTheory(b,taylor_order=4); ft.expand(); prop=build_propagator(ft,b,use_cache=False,verbose=False)
rvn=list(ft._ns._ring_var_names);_,pidx=build_field_index_map(rvn,ft._n_tilde);ext=_legs_to_phys_idx([('h',1),('h',1)],pidx)
base={SR.var('mu'):1.,SR.var('D'):1.,SR.var('c'):0.3,SR.var('T'):1.,SR.var('hstar1'):0.}
be=build_pipeline_records(ft,b,prop,ext,max_ell=1,verbose=False)
raw=[(td,float(SR(p).subs(base))) for td,p in be.get(1,[]) if abs(float(SR(p).subs(base)))>1e-14]
extt={0:0.0,1:0.0}; q=0.7
td0=raw[0][0]; dd=diagram_to_cstack(td0)
grid=diagram_kinematic(dd,[q],extt,1.0,1.0,spatial_dim=1,n_t=24,n_s=26,formfactor=None)
for N in (10000,100000,1000000):
    t0=time.time(); mc=mc_kinematic(dd,[q],extt,1.0,1.0,N=N,seed=1,formfactor=None); dt=time.time()-t0
    out('1-loop plain q=%.1f  grid=%.6e  MC(N=%d)=%.6e  rel=%.2e  (%.2fs)'%(q,grid,N,mc,abs(mc-grid)/abs(grid),dt))
