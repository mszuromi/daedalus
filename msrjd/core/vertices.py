"""
msrjd.core.vertices
====================
Decompose bigrade polynomial sectors into individual typed monomials
with field-leg metadata (VertexType, SourceType data structures).

Each monomial from the interacting action becomes a VertexType; each
monomial from the noise kernel becomes a SourceType.  These are the
atomic building blocks used by the type-assignment engine (Phase E).

Build Phase B.
"""

from sage.all import SR


# ── Data structures ──────────────────────────────────────────────────────────

class VertexType:
    """
    One monomial from an interacting-action sector (total degree >= 3).

    Attributes
    ----------
    coefficient : SR expression
        Coupling constant * combinatorial prefactor (the SR coefficient
        from the polynomial ring).
    response_legs : list of (str, int)
        Each entry is (field_base_name, population_index).  Repeated if
        the monomial has exponent > 1 in that generator.
    physical_legs : list of (str, int)
        Same format as response_legs, for physical field generators.
    bigrade : (int, int)
        (n_tilde, n_phys).
    """

    __slots__ = ('coefficient', 'response_legs', 'physical_legs', 'bigrade')

    def __init__(self, coefficient, response_legs, physical_legs, bigrade):
        self.coefficient   = coefficient
        self.response_legs = list(response_legs)
        self.physical_legs = list(physical_legs)
        self.bigrade       = tuple(bigrade)

    # Pickle support for __slots__
    def __getstate__(self):
        return {s: getattr(self, s) for s in self.__slots__}

    def __setstate__(self, state):
        for s, v in state.items():
            object.__setattr__(self, s, v)

    @property
    def in_degree(self):
        """Number of physical (incoming) legs."""
        return len(self.physical_legs)

    @property
    def out_degree(self):
        """Number of response (outgoing) legs."""
        return len(self.response_legs)

    @property
    def total_degree(self):
        return len(self.response_legs) + len(self.physical_legs)

    def __repr__(self):
        return (f'VertexType(bigrade={self.bigrade}, '
                f'resp={self.response_legs}, phys={self.physical_legs}, '
                f'coeff={self.coefficient})')


class SourceType:
    """
    One monomial from a noise-kernel sector (n_tilde >= 2, n_phys = 0).

    Attributes
    ----------
    coefficient : SR expression
    response_legs : list of (str, int)
    bigrade : (int, int)
        (n_tilde, 0).
    """

    __slots__ = ('coefficient', 'response_legs', 'bigrade')

    def __init__(self, coefficient, response_legs, bigrade):
        self.coefficient   = coefficient
        self.response_legs = list(response_legs)
        self.bigrade       = tuple(bigrade)

    # Pickle support for __slots__
    def __getstate__(self):
        return {s: getattr(self, s) for s in self.__slots__}

    def __setstate__(self, state):
        for s, v in state.items():
            object.__setattr__(self, s, v)

    @property
    def out_degree(self):
        """Number of response (outgoing) legs."""
        return len(self.response_legs)

    def __repr__(self):
        return (f'SourceType(bigrade={self.bigrade}, '
                f'resp={self.response_legs}, coeff={self.coefficient})')


# ── Ring variable name parsing ───────────────────────────────────────────────

def _parse_field_name(ring_var_name):
    """
    Parse a ring variable name like 'nt1', 'dn2', 'vt12' into
    (base_name, population_index).

    Convention: the name is a string of letters followed by digits.
    The digits are the 1-based population index.
    """
    # Find where digits start
    i = len(ring_var_name)
    while i > 0 and ring_var_name[i - 1].isdigit():
        i -= 1
    if i == len(ring_var_name) or i == 0:
        # No trailing digits or all digits — use full name, index 0
        return ring_var_name, 0
    base = ring_var_name[:i]
    idx  = int(ring_var_name[i:])
    return base, idx


# ── Decomposition ────────────────────────────────────────────────────────────

def decompose_sector(sector_poly, n_tilde, ring_var_names):
    """
    Decompose one bigrade sector polynomial into individual monomials.

    Parameters
    ----------
    sector_poly : PolynomialRing element
        One sector from FieldTheory.sectors(), e.g. the (2,1) sector.
    n_tilde : int
        Number of response-field generators (first n_tilde generators
        in the ring are response fields).
    ring_var_names : list of str
        Ring generator names in order, e.g. ['vt1','vt2','nt1','nt2','dv1','dv2','dn1','dn2'].

    Returns
    -------
    list of (VertexType or SourceType)
    """
    results = []

    for exp_vec, coeff in sector_poly.dict().items():
        resp_legs = []
        phys_legs = []

        for gen_idx, exponent in enumerate(exp_vec):
            if exponent == 0:
                continue
            name = ring_var_names[gen_idx]
            base, pop_idx = _parse_field_name(name)
            leg = (base, pop_idx)

            # Repeat for exponent multiplicity
            if gen_idx < n_tilde:
                resp_legs.extend([leg] * int(exponent))
            else:
                phys_legs.extend([leg] * int(exponent))

        n_t = len(resp_legs)
        n_p = len(phys_legs)
        bigrade = (n_t, n_p)

        if n_p == 0:
            results.append(SourceType(SR(coeff), resp_legs, bigrade))
        else:
            results.append(VertexType(SR(coeff), resp_legs, phys_legs, bigrade))

    return results


def extract_vertex_types(ft):
    """
    Extract all VertexType objects from a FieldTheory's interacting action.

    Parameters
    ----------
    ft : FieldTheory
        Must have been expanded (ft.expand() called).

    Returns
    -------
    list of VertexType
    """
    ft._require_expanded()
    vtypes = []
    for (n_t, n_p), poly in ft.vertices().items():
        # vertices() returns sectors with total degree >= 3
        # Some may be pure noise-kernel (n_p == 0) — skip those
        if n_p == 0:
            continue
        monomials = decompose_sector(poly, ft._n_tilde, list(ft._ns._ring_var_names))
        for m in monomials:
            if isinstance(m, VertexType):
                vtypes.append(m)
    return vtypes


def extract_source_types(ft):
    """
    Extract all SourceType objects from a FieldTheory's noise kernel.

    Parameters
    ----------
    ft : FieldTheory
        Must have been expanded (ft.expand() called).

    Returns
    -------
    list of SourceType
    """
    ft._require_expanded()
    stypes = []
    for (n_t, n_p), poly in ft.noise_kernel().items():
        monomials = decompose_sector(poly, ft._n_tilde, list(ft._ns._ring_var_names))
        for m in monomials:
            if isinstance(m, SourceType):
                stypes.append(m)
    return stypes


def available_degrees(vertex_types, source_types):
    """
    Compute the sets of available degree signatures.

    Parameters
    ----------
    vertex_types : list of VertexType
    source_types : list of SourceType

    Returns
    -------
    interaction_degrees : set of (int, int)
        Set of (in_degree, out_degree) pairs from vertex types.
    source_degrees : set of int
        Set of out_degree values from source types.
    """
    interaction_degrees = {(vt.in_degree, vt.out_degree) for vt in vertex_types}
    source_degrees      = {st.out_degree for st in source_types}
    return interaction_degrees, source_degrees
