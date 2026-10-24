"""Declared differential operators for the SDE / SPDE front end (:mod:`api.sde`).

An :class:`api.sde.SDE` model names the operators its equations use::

    m = SDE('Allen-Cahn', space=1)
    m.operator('dt', TimeDerivative())
    m.operator('lap', Laplacian())
    m.equation('u', lhs='dt(u)', rhs='D*lap(u) + u - u^3')

Each name may be written in two equivalent styles:

* product style, the operator as a factor in front of what it acts on:
  ``(tau*dt + 1)*x``, ``D*lap*u`` (the repository's long-standing ``Dt*x``);
* call style, the operator applied to its argument: ``dt(x)``, ``lap(u)``,
  ``lap(u^2)``.

``Dt`` is declared by default (a ``TimeDerivative``), so every model written
before declared operators existed is unchanged.  Declaring another name for the
time derivative renames it (``Dt`` is then no longer accepted).  A name is a
Python identifier (``dt``, ``lap``, ``Δ``; ``∂t`` and ``∇²`` are not
identifiers) that none of the model's fields, parameters, functions, kernels
or populations takes.

An operator object says how each engine path writes it and which paths support
it.  The front end translates the user's names into the engine's own symbols:

* ``v1_symbol``: the inert multiplicative symbol of the plain action form
  (``Dt``, ``Laplacian``), used where the operator acts linearly on one field;
* ``ir_call``: the argument-binding call of the spatial operator IR
  (``Dt(.)``, ``Lap(.)``, :mod:`api.spatial_operator_ir`), used when an
  operator acts on a nonlinear expression (a derivative vertex such as
  ``lap(u^2)``).

Later operators (gradient / divergence, a biharmonic, a fractional Laplacian,
kernel operators) are new subclasses of :class:`Operator` with their own
symbols, ``engines`` and ``nonlinear_argument`` rule; the front end translates
through these attributes (it singles out only the time derivative, which every
equation's lhs must contain, and the Laplacian, which vanishes at the
homogeneous mean-field saddle).
"""
from __future__ import annotations

__all__ = ['Operator', 'TimeDerivative', 'Laplacian']


class Operator:
    """Base class of a declared operator.

    Attributes
    ----------
    kind : str
        A short identifier of the operator (``'time_derivative'``,
        ``'laplacian'``).
    v1_symbol : str
        The multiplicative symbol of the plain action form.
    ir_call : str
        The call name of the spatial operator IR.
    spatial : bool
        True when the operator acts on space (it makes the model an SPDE).
    engines : frozenset of str
        The engine paths that support it: ``'temporal'`` (the time-domain
        pipeline) and/or ``'spatial'`` (the spatial pipeline).
    nonlinear_argument : bool
        True when the operator may act on a nonlinear expression (a derivative
        vertex, spatial operator IR only).
    """

    kind = 'operator'
    v1_symbol = ''
    ir_call = ''
    spatial = False
    engines = frozenset()
    nonlinear_argument = False
    #: the canonical names that only this kind of operator may be declared under
    canonical_names = frozenset()

    def supports(self, engine: str) -> bool:
        """True when the engine path ``engine`` (``'temporal'`` or
        ``'spatial'``) handles this operator."""
        return engine in self.engines

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))

    def __repr__(self):
        return f'{type(self).__name__}()'


class TimeDerivative(Operator):
    """The time derivative ``d/dt`` (default name ``Dt``).  It acts linearly
    on fields: ``dt(x)``, ``(tau*dt + 1)*x``."""

    kind = 'time_derivative'
    v1_symbol = 'Dt'
    ir_call = 'Dt'
    spatial = False
    engines = frozenset({'temporal', 'spatial'})
    nonlinear_argument = False
    canonical_names = frozenset({'Dt'})


class Laplacian(Operator):
    """The spatial Laplacian.  It makes the model an SPDE (spatial pipeline).
    ``lap(u)`` / ``D*lap*u`` act on one field; ``lap(u^2)`` (a conserved,
    Model-B-type derivative vertex) goes through the spatial operator IR."""

    kind = 'laplacian'
    v1_symbol = 'Laplacian'
    ir_call = 'Lap'
    spatial = True
    engines = frozenset({'spatial'})
    nonlinear_argument = True
    canonical_names = frozenset({'Laplacian', 'Lap'})

