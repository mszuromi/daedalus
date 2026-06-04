import numpy as np, sys, time
from math import factorial, pi
from scipy.special import kv
from pipeline.theory import TheoryBuilder
from pipeline.compute import FieldTheory
from pipeline._propagator import build_propagator
from msrjd.integration.spatial.diagram_descriptor import diagram_to_cstack
from msrjd.integration.spatial.pipeline_bridge import build_pipeline_records,_legs_to_phys_idx,_formfactor_callable
from msrjd.integration.spatial.full_integrator import diagram_kinematic, _symanzik_kernel_batch
from msrjd.diagrams.type_assignment import build_field_index_map
from sage.all import SR
def out(s): sys.stdout.write('@@ '+s+'\n'); sys.stdout.flush()

def bessel_deriv(descr, xs, external_times, mu, D, ff, spatial_dim=1, N=300000, seed=0):
    edges=list(descr.edges); internal=list(descr.internal_vertices); n_V=len(internal)
    idx={v:i for i,v in enumerate(internal)}; n_C=sum(1 for e in edges if e.kind=='C')
    a=np.array([e.a for e in edges],float).reshape(len(edges),-1); b=np.array([e.b for e in edges],float).reshape(len(edges),-1)
    L=a.shape[1]; n=n_V+n_C; rng=np.random.default_rng(seed); xs=np.asarray(xs,float); nx=len(xs)
    total=np.zeros(nx,dtype=complex)
    E=rng.standard_exponential((N,n)); s=E/E.sum(1,keepdims=True)
    tvals={leaf:np.full(N,tt) for leaf,tt in external_times.items()}
    for k,v in enumerate(internal): tvals[v]=-s[:,k]
    sig=[s[:,n_V+c] for c in range(n_C)]
    w=np.empty((N,len(edges))); valid=np.ones(N,bool); ci=0
    for ei,e in enumerate(edges):
        tu,tv=tvals[e.u],tvals[e.v]
        if e.kind=='R':
            d=tv-tu; w[:,ei]=d
            if (e.u in idx) and (e.v in idx): valid&=(d>=0.0)
            else: w[:,ei]=np.maximum(d,1e-15)
        else: w[:,ei]=np.abs(tu-tv)+sig[ci]; ci+=1
    wv=w[valid]
    if wv.shape[0]==0: return np.real(total)
    pref,Bk,ok,M,Nn,Q=_symanzik_kernel_batch(a,b,wv,D,spatial_dim,return_gaussian=True)
    good=ok&(Bk>1e-300); Mg,Ng,Qg,wg=M[good],Nn[good],Q[good],wv[good]
    Uhat=np.linalg.det(Mg)
    Qeff=(Qg-np.einsum('plj,plk->pjk',Ng,np.linalg.solve(Mg,Ng)))[:,0,0]
    Fhat=Uhat*Qeff; What=wg.sum(1)
    okF=(Uhat>1e-300)&(Fhat>1e-300)&(What>0)
    Mg,Ng,Uhat,Fhat,What=Mg[okF],Ng[okF],Uhat[okF],Fhat[okF],What[okF]; Pg=Mg.shape[0]
    ahat=-np.linalg.solve(Mg,Ng)[:,:,0]; Shat=np.linalg.inv(Mg)/(2.0*D); Bhat=D*Fhat/Uhat
    powers,g=ff.moment_bessel(ahat,Shat,Bhat,xs)            # powers:(nm,), g:(nm,Pg,nx)
    Pp=n-1-(L+1)*spatial_dim/2.0
    c0=(4*pi*D)**(-L*spatial_dim/2.0)*(4*pi*D*Fhat)**(-spatial_dim/2.0); aa=mu*What
    for ix,x in enumerate(xs):
        if x==0.0: continue
        cc=x*x*Uhat/(4.0*D*Fhat); z=2.0*np.sqrt(aa*cc)
        acc=np.zeros(Pg,dtype=complex)
        for im,m in enumerate(powers):
            Pm=Pp-m; acc+=g[im,:,ix]*2.0*(cc/aa)**((Pm+1.0)/2.0)*kv(Pm+1.0,z)
        total[ix]+=np.sum(c0*acc)
    return np.real(total)/(factorial(n-1)*N)

b=(TheoryBuilder('kpz',n_populations=0).physical_field('h',spatial_dim=1)
  .parameter('mu',default=1.0,domain='positive').parameter('D',default=1.0,domain='positive').parameter('c',default=0.3,domain='real').parameter('T',default=1.0,domain='positive')
  .equation(lhs='(Dt+mu-D*Laplacian)*h',rhs='0').set_action_text('ht*(Dt(h)+mu*h-D*Lap(h)-(c/2)*Dx(h,0)^2)-T*ht^2').operator_ir().boundary('infinite').initial('stationary').build())
ft=FieldTheory(b,taylor_order=4); ft.expand(); prop=build_propagator(ft,b,use_cache=False,verbose=False)
rvn=list(ft._ns._ring_var_names);_,pidx=build_field_index_map(rvn,ft._n_tilde);ext=_legs_to_phys_idx([('h',1),('h',1)],pidx)
base={SR.var('mu'):1.,SR.var('D'):1.,SR.var('c'):0.3,SR.var('T'):1.,SR.var('hstar1'):0.}
vt=[{'weight':float(SR(t['weight']).subs(base)),'n_phys':t['n_phys'],'chain':t['chain'],'mode':t['mode']} for t in ft._ns._operator_ir_vertex_terms]
be=build_pipeline_records(ft,b,prop,ext,max_ell=1,verbose=False)
raw=[td for td,p in be.get(1,[]) if abs(float(SR(p).subs(base)))>1e-14]
dd=diagram_to_cstack(raw[0]); ff=_formfactor_callable(raw[0],vt,d=1); extt={0:0.0,1:0.0}; xs=np.array([0.5,1.0,2.0,3.0])
out('moment_bessel powers = %s'%(ff.moment_bessel(np.zeros((1,1)),np.ones((1,1,1)),np.ones(1),np.array([1.0]))[0]))
g=diagram_kinematic(dd,[0.0],extt,1.0,1.0,spatial_dim=1,n_t=26,n_s=28,xs=xs,formfactor=ff)
out('GRID  KPZ deriv = %s'%np.array2string(np.real(g),precision=6))
for N in (1000000,5000000):
    t0=time.time(); m=bessel_deriv(dd,xs,extt,1.0,1.0,ff,N=N,seed=1)
    rel=np.abs(m-np.real(g))/(np.abs(np.real(g))+1e-30)
    out('BESSEL N=%d = %s rel=%s (%.1fs)'%(N,np.array2string(m,precision=6),np.array2string(rel,precision=2),time.time()-t0))
