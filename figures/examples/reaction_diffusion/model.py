"""Two-species reaction-diffusion in d = 1, typed in the SDE front end.

    da = (Da a'' - mua a - g b - ga a^3) dt + dW_a,    <dW_a dW_a> = 2 Ta delta(x - x') dt
    db = (Db b'' - mub b + h a - gb b^3) dt + dW_b,    <dW_b dW_b> = 2 Tb delta(x - x') dt

Paper parameters: mua = mub = Da = Db = 1, g = h = 1 (linear eigenvalues -1 +- i), Ta = 1, Tb = 0.7, ga = gb = lam = 0.1.
This builds the same action as the tracked model file daedalus/models/coupled_rd_2species_1d.model.py."""
from daedalus.api.sde import SDE
from daedalus.api.noise import Gaussian
from daedalus.api.operators import Laplacian

PARAMS = dict(mua=1.0, mub=1.0, Da=1.0, Db=1.0, g=1.0, h=1.0, ga=0.1, gb=0.1, Ta=1.0, Tb=0.7)


def build():
    m = (SDE('two-species reaction-diffusion', space=1).operator('lap', Laplacian())
         .parameter('mua', default=1.0, domain='positive').parameter('mub', default=1.0, domain='positive')
         .parameter('Da', default=1.0, domain='positive').parameter('Db', default=1.0, domain='positive')
         .parameter('g', default=1.0).parameter('h', default=1.0)
         .parameter('ga', default=0.1).parameter('gb', default=0.1)
         .parameter('Ta', default=1.0, domain='positive').parameter('Tb', default=0.7, domain='positive'))
    return m.sde('''
        da = (Da*lap(a) - mua*a - g*b - ga*a^3) dt + dW_a
        db = (Db*lap(b) - mub*b + h*a - gb*b^3) dt + dW_b
    ''', noises={'a': Gaussian(var='2*Ta', space='white'), 'b': Gaussian(var='2*Tb', space='white')})


def params(lam=0.1):
    return dict(PARAMS, ga=float(lam), gb=float(lam))
