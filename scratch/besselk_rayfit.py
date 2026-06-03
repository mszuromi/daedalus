import numpy as np, sys
from scipy.special import kv
from pipeline.theory import TheoryBuilder
from pipeline.compute import FieldTheory
from pipeline._propagator import build_propagator
from msrjd.integration.spatial.diagram_descriptor import diagram_to_cstack
from msrjd.integration.spatial.pipeline_bridge import build_pipeline_records,_legs_to_phys_idx,_formfactor_callable
from msrjd.integration.spatial.full_integrator import _symanzik_kernel_batch,_formfactor_average_x
from msrjd.diagrams.type_assignment import build_field_index_map
from sage.all import SR
def out(s): sys.stdout.write('@@ '+s+'\n'); sys.stdout.flush()
# Bessel-K identity sanity: ∫λ^p e^{-aλ-c/λ}dλ = 2(c/a)^{(p+1)/2}K_{p+1}(2√(ac))
a,c,p=1.3,0.8,1.0
lg=np.linspace(-6,6,200000); lam=np.exp(lg)
num=np.trapz(lam**p*np.exp(-a*lam-c/lam)*lam, lg)   # ∫dλ = ∫λ d(logλ)
clf=2*(c/a)**((p+1)/2)*kv(p+1,2*np.sqrt(a*c))
out('Bessel-K identity: numeric=%.6e  closed=%.6e  rel=%.2e'%(num,clf,abs(num-clf)/clf))
# real 1-loop integrand along a ray (u_v=-t_v, σ) = λ·ŝ — does it fit A·λ^p·e^{-aλ-c/λ}?
b=(TheoryBuilder('kpz',n_populations=0).physical_field('h',spatial_dim=1)
  .parameter('mu',default=1.0,domain='positive').parameter('D',default=1.0,domain='positive').parameter('c',default=0.3,domain='real').parameter('T',default=1.0,domain='positive')
  .equation(lhs='(Dt+mu-D*Laplacian)*h',rhs='0').set_action_text('ht*(Dt(h)+mu*h-D*Lap(h)-(c/2)*Dx(h,0)^2)-T*ht^2').operator_ir().boundary('infinite').initial('stationary').build())
ft=FieldTheory(b,taylor_order=4); ft.expand(); prop=build_propagator(ft,b,use_cache=False,verbose=False)
rvn=list(ft._ns._ring_var_names);_,pidx=build_field_index_map(rvn,ft._n_tilde);ext=_legs_to_phys_idx([('h',1),('h',1)],pidx)
base={SR.var('mu'):1.,SR.var('D'):1.,SR.var('c'):0.3,SR.var('T'):1.,SR.var('hstar1'):0.}
be=build_pipeline_records(ft,b,prop,ext,max_ell=1,verbose=False)
raw=[(td,float(SR(p_).subs(base))) for td,p_ in be.get(1,[]) if abs(float(SR(p_).subs(base)))>1e-14]
td=raw[0][0]; dd=diagram_to_cstack(td); ff=_formfactor_callable(td,vt:=[{'weight':float(SR(t['weight']).subs(base)),'n_phys':t['n_phys'],'chain':t['chain'],'mode':t['mode']} for t in ft._ns._operator_ir_vertex_terms],d=1)
edges=list(dd.edges); internal=list(dd.internal_vertices); n_V=len(internal); idx={v:i for i,v in enumerate(internal)}; n_C=sum(1 for e in edges if e.kind=='C')
a_e=np.array([e.a for e in edges],float).reshape(len(edges),-1); b_e=np.array([e.b for e in edges],float).reshape(len(edges),-1)
out('diagram: n_V=%d n_C=%d L=%d'%(n_V,n_C,a_e.shape[1]))
# angular direction (ordered: deeper-past vertex first), σ:
uhat=np.array([0.6,0.35][:n_V]); shat=np.array([0.5]*n_C); x0=1.0; mu=D=1.0
def integrand(lam, formfactor):
    tv={leaf:0.0 for leaf in {e.u for e in edges}|{e.v for e in edges} if leaf not in idx}
    for v in internal: tv[v]=-lam*uhat[idx[v]]
    sig=[lam*shat[k] for k in range(n_C)]
    w=np.empty((1,len(edges))); muW=0.0; ci=0
    for ei,e in enumerate(edges):
        tu,tvv=tv[e.u],tv[e.v]
        if e.kind=='R': w[0,ei]=max(tvv-tu,1e-12); muW+=max(tvv-tu,0.0)
        else: dt=abs(tu-tvv); w[0,ei]=dt+sig[ci]; muW+=dt; ci+=1
    muW+=sum(sig)
    if formfactor is None:
        pref,B,ok=_symanzik_kernel_batch(a_e,b_e,w,D,1)
        if not ok[0] or B[0]<=0: return 0.0
        hk=(4*np.pi*B[0])**(-0.5)*np.exp(-x0**2/(4*B[0])); return np.exp(-mu*muW)*pref[0]*hk
    pref,B,ok,M,N,Q=_symanzik_kernel_batch(a_e,b_e,w,D,1,return_gaussian=True)
    if not ok[0] or B[0]<=0: return 0.0
    hk=(4*np.pi*B[0])**(-0.5)*np.exp(-x0**2/(4*B[0]))
    FF=_formfactor_average_x(formfactor,M,N,Q,D,np.ones(1,bool),np.array([x0]),spatial_dim=1,gh_order=getattr(formfactor,'gh_order_needed',6),q_deg=getattr(formfactor,'q_poly_deg',8))
    return np.real(np.exp(-mu*muW)*pref[0]*hk*FF[0,0])
for label,ffd in (('PLAIN',None),('KPZ-derivative',ff)):
    lam=np.logspace(-1.2,1.3,40); I=np.array([integrand(l,ffd) for l in lam])
    g=I>0; L=np.log(lam[g]); Y=np.log(I[g])
    A=np.vstack([np.ones_like(L),L,-lam[g],-1.0/lam[g]]).T
    coef,res,*_=np.linalg.lstsq(A,Y,rcond=None)
    pred=A@coef; ss=1-np.sum((Y-pred)**2)/np.sum((Y-Y.mean())**2)
    out('%s ray fit logI=logA+p·logλ−aλ−c/λ : p=%.3f a=%.3f c=%.3f  R^2=%.6f (n=%d)'%(label,coef[1],coef[2],coef[3],ss,g.sum()))
