"""Spatial-extension utilities (v1): heat-kernel propagators in (t, x)."""

#: Why conserved noise (``div`` of a white vector noise) is refused; shared by
#: ``api.noise.Gaussian(space='conserved')``, the SDE front end and the noise
#: extractor of ``spatial_correlator``.
CONSERVED_NOISE_WHY = (
    'conserved noise (the divergence of a white vector noise, noise spectrum '
    'var*|q|^2) is not supported: the spatial integrator takes momentum-'
    'independent (white) noise only -- its tree heat kernels and its loop '
    'integral carry one constant noise vertex per correlation line -- and the '
    'Model B dynamics it belongs to also need the k^4 propagator (lap(lap(u))) '
    'and the conserved vertex lap(u^3), which the spatial pipeline refuses too')
