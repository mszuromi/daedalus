import numpy as np, time, sys
from pipeline.theory import TheoryBuilder
from pipeline.compute import FieldTheory
from pipeline._propagator import build_propagator
from msrjd.integration.spatial.diagram_descriptor import diagram_to_cstack
from msrjd.integration.spatial.pipeline_bridge import build_pipeline_records,_legs_to_phys_idx,_formfactor_callable
from msrjd.integration.spatial.full_integrator import (diagram_correlator_x, _symanzik_kernel_batch,
    _formfactor_average_x, _is_retarded_type, external_times_2pt)
from msrjd.integration.spatial.causal_chambers import causal_chambers
from msrjd.diagrams.type_assignment import build_field_index_map
from sage.all import SR
def out(s): sys.stdout.write('@@ '+s+'\n'); sys.stdout.flush()

def mc_kin_x(descr, xs, external_times, mu, D, N=1000000, W=None, seed=0, formfactor=None):
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
    xs=np.asarray(xs,float); total=np.zeros(len(xs))
    qdeg=getattr(formfactor,'q_poly_deg',8) or 8; gh=getattr(formfactor,'gh_order_needed',6) or 6
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
        pref,Bk,okk,Mb,Nb,Qb=_symanzik_kernel_batch(a,b,w_batch,D,1,return_gaussian=True)
        good=(okk&(Bk>1e-300)) if Mb is not None else np.zeros(N,dtype=bool)
        if not np.any(good): continue
        Bg=Bk[good]; wamp=(wgt*pref)[good]
        hk=(4.0*np.pi*Bg)[:,None]**(-0.5)*np.exp(-(xs[None,:]**2)/(4.0*Bg[:,None]))
        if formfactor is None:
            total+=np.einsum('p,px->x',wamp,hk)/N
        else:
            FF=_formfactor_average_x(formfactor,Mb[good],Nb[good],Qb[good],D,np.ones(int(good.sum()),bool),xs,spatial_dim=1,gh_order=gh,q_deg=qdeg)
            total+=np.real(np.einsum('p,px,px->x',wamp,hk,FF))/N
    return total

def mc_corr_x(descr,pre,xs,tau,mu,D,N,seed,ff):
    nC=sum(1 for e in descr.edges if e.kind=='C')
    v=(2.0**(-nC))*float(pre)*mc_kin_x(descr,xs,external_times_2pt(descr,tau),mu,D,N=N,seed=seed,formfactor=ff)
    if _is_retarded_type(descr) and tau==0.0: v=v*2.0
    elif _is_retarded_type(descr): v=v+(2.0**(-nC))*float(pre)*mc_kin_x(descr,xs,external_times_2pt(descr,-tau),mu,D,N=N,seed=seed+7,formfactor=ff)
    return v

def setup():
    b=(TheoryBuilder('kpz',n_populations=0).physical_field('h',spatial_dim=1)
      .parameter('mu',default=1.0,domain='positive').parameter('D',default=1.0,domain='positive')
      .parameter('c',default=0.3,domain='real').parameter('T',default=1.0,domain='positive')
      .equation(lhs='(Dt+mu-D*Laplacian)*h',rhs='0').set_action_text('ht*(Dt(h)+mu*h-D*Lap(h)-(c/2)*Dx(h,0)^2)-T*ht^2').operator_ir().boundary('infinite').initial('stationary').build())
    ft=FieldTheory(b,taylor_order=4); ft.expand(); prop=build_propagator(ft,b,use_cache=False,verbose=False)
    rvn=list(ft._ns._ring_var_names);_,pidx=build_field_index_map(rvn,ft._n_tilde);ext=_legs_to_phys_idx([('h',1),('h',1)],pidx)
    base={SR.var('mu'):1.,SR.var('D'):1.,SR.var('c'):0.3,SR.var('T'):1.,SR.var('hstar1'):0.}
    vt=[{'weight':float(SR(t['weight']).subs(base)),'n_phys':t['n_phys'],'chain':t['chain'],'mode':t['mode']} for t in ft._ns._operator_ir_vertex_terms]
    return b,ft,prop,ext,base,vt
xs=np.array([0.0,1.0,2.0])
b,ft,prop,ext,base,vt=setup()
# --- 1-loop: MC vs grid (ground truth) for the loop δC(x) ---
be=build_pipeline_records(ft,b,prop,ext,max_ell=1,verbose=False)
d1=[(diagram_to_cstack(td),float(SR(p).subs(base)),_formfactor_callable(td,vt,d=1)) for td,p in be.get(1,[]) if abs(float(SR(p).subs(base)))>1e-14]
g=np.zeros(len(xs))
for dd,pv,ff in d1: g+=diagram_correlator_x(dd,pv,xs,0.0,1.0,1.0,spatial_dim=1,n_t=22,n_s=24,formfactor=ff)
for N in (300000,3000000):
    t0=time.time(); m=np.zeros(len(xs))
    for i,(dd,pv,ff) in enumerate(d1): m+=mc_corr_x(dd,pv,xs,0.0,1.0,1.0,N,1+i,ff)
    out('1-loop δC(x) grid=%s'%np.array2string(g,precision=6))
    out('1-loop δC(x) MC(N=%d)=%s rel@0=%.2e (%.1fs)'%(N,np.array2string(m,precision=6),abs(m[0]-g[0])/abs(g[0]),time.time()-t0))
# --- 2-loop: MC vs coarse grid (nt6,ns6) for the ell=2 loop δC(x) ---
be2=build_pipeline_records(ft,b,prop,ext,max_ell=2,verbose=False)
d2=[(diagram_to_cstack(td),float(SR(p).subs(base)),_formfactor_callable(td,vt,d=1)) for td,p in be2.get(2,[]) if abs(float(SR(p).subs(base)))>1e-14]
out('--- 2-loop: %d diagrams (grid would be 1.8e8/chamber → OOM) ---'%len(d2))
t0=time.time(); g6=np.zeros(len(xs))
for dd,pv,ff in d2: g6+=diagram_correlator_x(dd,pv,xs,0.0,1.0,1.0,spatial_dim=1,n_t=6,n_s=6,formfactor=ff)
out('2-loop ell=2 δC(x) GRID nt6/ns6=%s (%.0fs)'%(np.array2string(g6,precision=6),time.time()-t0))
for N in (300000,1000000,3000000):
    t0=time.time(); m=np.zeros(len(xs))
    for i,(dd,pv,ff) in enumerate(d2): m+=mc_corr_x(dd,pv,xs,0.0,1.0,1.0,N,100+i,ff)
    out('2-loop ell=2 δC(x) MC(N=%d)=%s (%.0fs)'%(N,np.array2string(m,precision=6),time.time()-t0))
