# encoding: utf-8

"""Linear algebra with matrix product arrays

Currently, we support computing extremal eigenvalues and eigenvectors
of MPOs.

"""

from __future__ import absolute_import, division, print_function

import functools as ft
import itertools as it
import numpy as np
from scipy import sparse as sp

import gc

from six.moves import range

from . import mparray as mp
from . import utils
from ._named_ndarray import named_ndarray
from .factory import random_mpa

__all__ = ['eig', 'eig_sum']

def _eig_leftvec_add(leftvec, mpo_lten, mps_lten, mps_lten2=None):
    """Add one column to the left vector.

    :param leftvec: existing left vector
        It has three indices: mps bond, mpo bond, complex conjugate mps bond
    :param op_lten: Local tensor of the MPO
    :param mps_lten: Local tensor of the current MPS eigenstate

    leftvecs[i] is L_{i-1}, see [:ref:`Sch11 <Sch11>`, arXiv version, Fig. 39
    and p. 63 and Fig. 38 and Eq. (191) on p. 62]. Regarding Fig. 39,
    things are as follows:

    Figure:

    Upper row: MPS matrices
    Lower row: Complex Conjugate MPS matrices
    Middle row: MPO matrices with row (column) indices to bottom (top)

    Figure, left part:

    a_{i-1} (left): 'mps_bond' of leftvec
    a_{i-1} (right): 'left_mps_bond' of mps_lten
    b_{i-1} (left): 'mpo_bond' of leftvec
    b_{i-1} (right): 'left_mpo_bond' of mpo_lten
    a'_{i-1} (left): 'cc_mps_bond' of leftvec
    a'_{i+1} (left): 'left_mps_bond' of mps_lten.conj()
    a_i: 'right_mps_bond' of mps_lten
    b_i: 'right_mpo_bond' of mpo_lten
    a'_i: 'right_mps_bond' of mps_lten.conj()

    """
    leftvec_names = ('mps_bond', 'mpo_bond', 'cc_mps_bond')
    mpo_names = ('left_mpo_bond', 'phys_row', 'phys_col', 'right_mpo_bond')
    mps_names = ('left_mps_bond', 'phys', 'right_mps_bond')
    leftvec = named_ndarray(leftvec, leftvec_names)
    mpo_lten = named_ndarray(mpo_lten, mpo_names)
    mps_lten = named_ndarray(mps_lten, mps_names)
    mps_lten2 = mps_lten if mps_lten2 is None else named_ndarray(mps_lten2,
                                                                 mps_names)

    contract_mps = (('mps_bond', 'left_mps_bond'),)
    leftvec = leftvec.tensordot(mps_lten, contract_mps)
    rename_mps = (('right_mps_bond', 'mps_bond'),)
    leftvec = leftvec.rename(rename_mps)

    contract_mpo = (
        ('mpo_bond', 'left_mpo_bond'),
        ('phys', 'phys_col'))
    leftvec = leftvec.tensordot(mpo_lten, contract_mpo)
    contract_cc_mps = (
        ('cc_mps_bond', 'left_mps_bond'),
        ('phys_row', 'phys'))
    leftvec = leftvec.tensordot(mps_lten2.conj(), contract_cc_mps)
    rename_mps_mpo = (
        ('right_mpo_bond', 'mpo_bond'),
        ('right_mps_bond', 'cc_mps_bond'))
    leftvec = leftvec.rename(rename_mps_mpo)

    leftvec = leftvec.to_array(leftvec_names)
    return leftvec


def _eig_rightvec_add(rightvec, mpo_lten, mps_lten):
    """Add one column to the right vector.

    :param rightvec: existing right vector
        It has three indices: mps bond, mpo bond, complex conjugate mps bond
    :param op_lten: Local tensor of the MPO
    :param mps_lten: Local tensor of the current MPS eigenstate

    This does the same thing as _eig_leftvec_add(), except that
    'left' and 'right' are exchanged in the contractions (but not in
    the axis names of the input tensors).

    """
    rightvec_names = ('mps_bond', 'mpo_bond', 'cc_mps_bond')
    mpo_names = ('left_mpo_bond', 'phys_row', 'phys_col', 'right_mpo_bond')
    mps_names = ('left_mps_bond', 'phys', 'right_mps_bond')
    rightvec = named_ndarray(rightvec, rightvec_names)
    mpo_lten = named_ndarray(mpo_lten, mpo_names)
    mps_lten = named_ndarray(mps_lten, mps_names)

    contract_mps = (('mps_bond', 'right_mps_bond'),)
    rightvec = rightvec.tensordot(mps_lten, contract_mps)
    rename_mps = (('left_mps_bond', 'mps_bond'),)
    rightvec = rightvec.rename(rename_mps)

    contract_mpo = (
        ('mpo_bond', 'right_mpo_bond'),
        ('phys', 'phys_col'))
    rightvec = rightvec.tensordot(mpo_lten, contract_mpo)
    contract_cc_mps = (
        ('cc_mps_bond', 'right_mps_bond'),
        ('phys_row', 'phys'))
    rightvec = rightvec.tensordot(mps_lten.conj(), contract_cc_mps)
    rename_mps_mpo = (
        ('left_mpo_bond', 'mpo_bond'),
        ('left_mps_bond', 'cc_mps_bond'))
    rightvec = rightvec.rename(rename_mps_mpo)

    rightvec = rightvec.to_array(rightvec_names)
    return rightvec


def _eig_leftvec_add_mps(lv, lt1, lt2):
    """Add one column to the left vector (MPS version)"""
    # MPS 1: Interpreted as |psiXpsi| part of the operator
    # MPS 2: The current eigvectector candidate
    # NB: It would be more efficient to store lt1.conj() instead of lt1.
    # lv axes: 0: mps1 bond, 1: mps2 bond
    lv = np.tensordot(lv, lt1.conj(), axes=(0, 0))
    # lv axes: 0: mps2 bond, 1: physical leg, 2: mps1 bond
    lv = np.tensordot(lv, lt2, axes=((0, 1), (0, 1)))
    # lv axes: 0: mps1 bond, 1: mps2 bond
    return lv


def _eig_rightvec_add_mps(rv, lt1, lt2):
    """Add one column to the right vector (MPS version)"""
    # rv axes: 0: mps1 bond, 1: mps2 bond
    rv = np.tensordot(rv, lt1.conj(), axes=(0, 2))
    # rv axes: 0: mps2 bond, 1: mps1 bond, 2: physical leg
    rv = np.tensordot(rv, lt2, axes=((0, 2), (2, 1)))
    # rv axes: 0: mps1 bond, 1: mps2 bond
    return rv


def _eig_sum_leftvec_add(
        mpas, mpas_ndims, leftvec_out, leftvec, pos, mps_lten):
    """Add one column to the left vector (MPA list dispatching)"""
    for i, mpa, ndims, lv in zip(it.count(), mpas, mpas_ndims, leftvec):
        if ndims == 2:
            leftvec_out[i] = _eig_leftvec_add(lv, mpa.lt[pos], mps_lten)
        elif ndims == 1:
            leftvec_out[i] = _eig_leftvec_add_mps(lv, mpa.lt[pos], mps_lten)
        else:
            raise ValueError('ndims = {!r} not supported'.format(ndims))


def _eig_sum_rightvec_add(
        mpas, mpas_ndims, rightvec_out, rightvec, pos, mps_lten):
    """Add one column to the right vector (MPA list dispatching)"""
    for i, mpa, ndims, rv in zip(it.count(), mpas, mpas_ndims, rightvec):
        if ndims == 2:
            rightvec_out[i] = _eig_rightvec_add(rv, mpa.lt[pos], mps_lten)
        elif ndims == 1:
            rightvec_out[i] = _eig_rightvec_add_mps(rv, mpa.lt[pos], mps_lten)
        else:
            raise ValueError('ndims = {!r} not supported'.format(ndims))
    return rightvec

# @profile
def _eig_local_op(leftvec, mpo_ltens, rightvec):
    """Create the operator for local eigenvalue minimization on few sites

    :param leftvec: Left vector
        Three indices: mps bond, mpo bond, complex conjugate mps bond
    :param mpo_ltens: List of local tensors of the MPO
    :param rightvec: Right vector
        Three indices: mps bond, mpo bond, complex conjugate mps bond

    See [:ref:`Sch11 <Sch11>`, arXiv version, Fig. 38 on p. 62].
    If ``len(mpo_ltens) == 1``\ , this method implements the contractions
    across the dashed lines in the figure. For ``let(mpo_ltens) > 1``, we
    return the operator for what is probably called "multi-site DMRG".

    Indices and axis names map as follows:

    Upper row: MPS matrices
    Lower row: Complex Conjugate MPS matrices
    Middle row: MPO matrices with row (column) indices to bottom (top)

    a_{i-1}: 'mps_bond' of leftvec
    a'_{i-1}: 'cc_mps_bond' of leftvec
    a_i: 'mps_bond' of rightvec
    a'_i: 'mps_bond' of rightvec
    sigma_i: 'phys_col' of mpo_lten
    sigma'_i: 'phys_row' of mpo_lten

    """
    # Produce one MPO local tensor supported on len(mpo_ltens) sites.
    nr_sites = len(mpo_ltens)
    mpo_lten = mpo_ltens[0]
    for lten in mpo_ltens[1:]:
        mpo_lten = utils.matdot(mpo_lten, lten)
    mpo_lten = utils.local_to_global(mpo_lten, nr_sites,
                                     left_skip=1, right_skip=1)
    s = mpo_lten.shape
    mpo_lten = mpo_lten.reshape(
        (s[0], np.prod(s[1:1 + nr_sites]), np.prod(s[1 + nr_sites:-1]), s[-1]))

    # Do the contraction mentioned above.
    leftvec_names = ('left_mps_bond', 'left_mpo_bond', 'left_cc_mps_bond')
    mpo_names = ('left_mpo_bond', 'phys_row', 'phys_col', 'right_mpo_bond')
    rightvec_names = ('right_mps_bond', 'right_mpo_bond', 'right_cc_mps_bond')
    leftvec = named_ndarray(leftvec, leftvec_names)
    mpo_lten = named_ndarray(mpo_lten, mpo_names)
    rightvec = named_ndarray(rightvec, rightvec_names)

    contract = (('left_mpo_bond', 'left_mpo_bond'),)
    op = leftvec.tensordot(mpo_lten, contract)
    contract = (('right_mpo_bond', 'right_mpo_bond'),)
    op = op.tensordot(rightvec, contract)

    op_names = (
        'left_cc_mps_bond', 'phys_row', 'right_cc_mps_bond',
        'left_mps_bond', 'phys_col', 'right_mps_bond',
    )
    op = op.to_array(op_names)
    op = op.reshape((np.prod(op.shape[0:3]), -1))
    return op


def _eig_local_op_mps(lv, ltens, rv):
    """Local operator contribution from an MPS"""
    # MPS 1 / ltens: Interpreted as |psiXpsi| part of the operator
    # MPS 2: The current eigvectector candidate
    op = lv.T
    # op axes: 0 mps2 bond, 1: mps1 bond
    s = op.shape
    op = op.reshape((s[0], 1, s[1]))
    # op axes: 0 mps2 bond, 1: physical legs, 2: mps1 bond
    for lt in ltens:
        # op axes: 0: mps2 bond, 1: physical legs, 2: mps1 bond
        op = np.tensordot(op, lt.conj(), axes=(2, 0))
        # op axes: 0: mps2 bond, 1, 2: physical legs, 3: mps1 bond
        s = op.shape
        op = op.reshape((s[0], -1, s[3]))
        # op axes: 0: mps2 bond, 1: physical legs, 2: mps1 bond
    op = np.tensordot(op, rv, axes=(2, 0))
    # op axes: 0: mps2 bond, 1: physical legs, 2: mps2 bond
    op = np.outer(op.conj(), op)
    # op axes:
    # 0: (0a: left cc mps2 bond, 0b: physical row leg, 0c: right cc mps2 bond),
    # 1: (1a: left mps2 bond, 1b: physical column leg, 1c: right mps2 bond)
    return op

# @profile
def _eig_minimize_locally(leftvec, mpo_ltens, rightvec, eigvec_ltens,
                          eigs):
    """Perform the local eigenvalue minimization on few sites

    Return a new (expectedly smaller) eigenvalue and a new local
    tensor for the MPS eigenvector.

    :param leftvec: Left vector
        Three indices: mps bond, mpo bond, complex conjugate mps bond
    :param mpo_ltens: List of local tensors of the MPO
    :param rightvec: Right vector
        Three indices: mps bond, mpo bond, complex conjugate mps bond
    :param eigvec_ltens: List of local tensors of the MPS eigenvector
    :returns: mineigval, mineigval_eigvec_lten

    See [:ref:`Sch11 <Sch11>`, arXiv version, Fig. 42 on p. 67].  This method
    computes the operator ('op'), defined by everything except the
    circle of the first term in the figure. It then obtains the
    minimal eigenvalue (lambda in the figure) and eigenvector (circled
    part / single matrix in the figure).

    We use the figure as follows:

    Upper row: MPS matrices
    Lower row: Complex Conjugate MPS matrices
    Middle row: MPO matrices with row (column) indices to bottom (top)

    """
    op = _eig_local_op(leftvec, list(mpo_ltens), rightvec)
    return _eig_minimize_locally2(op, list(eigvec_ltens), eigs)

# @profile
def _eig_minimize_locally2(local_op, eigvec_ltens, eigs):
    """Implement the main part of :func:`_eig_minimize_locally`

    See :func:`_eig_minimize_locally` for a description.

    """
    eigvec_rank = max(lten.shape[0] for lten in eigvec_ltens)
    eigvec_lten = eigvec_ltens[0]
    for lten in eigvec_ltens[1:]:
        eigvec_lten = utils.matdot(eigvec_lten, lten)
    eigval, eigvec = eigs(local_op, v0=eigvec_lten.flatten())
    if eigvec.ndim == 1:
        if len(eigval.flat) != 1:
            raise ValueError('eigvals from eigs() must be length one')
    elif eigvec.ndim == 2:
        if eigval.shape != (1,) or eigvec.shape[1] != 1:
            raise ValueError('eigs() must return exactly one eigenvalue')
        eigvec = eigvec[:, 0]
    else:
        raise ValueError('eigs() returned array of wrong dimension')
    eigval = eigval.flat[0]
    eigvec_lten = eigvec.reshape(eigvec_lten.shape)
    if len(eigvec_ltens) == 1:
        eigvec_lten = (eigvec_lten,)
    else:
        # If we minimize on multiple sites, we must compress to the
        # desired rank.
        #
        # TODO: Return the truncation error.
        #
        # "the truncation error of conventional DMRG [...] has emerged
        # as a highly reliable tool for gauging the quality of
        # results" [Sch11, Sec. 6.4, p. 74]
        eigvec_lten = mp.MPArray.from_array(eigvec_lten, 1, has_virtual=True)
        eigvec_lten.compress(method='svd', rank=eigvec_rank)
        eigvec_lten = eigvec_lten.lt
    return eigval, eigvec_lten


def _eig_sum_minimize_locally(
        mpas, mpas_ndims, leftvec, pos, rightvec, eigvec_ltens, eigs):
    """Local minimization (MPA list dispatching)"""
    # Our task is quite simple: Compute the local operator for each
    # contribution in the sum and sum the results, then minimize.
    op = 0
    for mpa, ndims, lv, rv in zip(mpas, mpas_ndims, leftvec, rightvec):
        if ndims == 2:
            op += _eig_local_op(lv, list(mpa.lt[pos]), rv)
        elif ndims == 1:
            op += _eig_local_op_mps(lv, list(mpa.lt[pos]), rv)
        else:
            raise ValueError('ndims = {!r} not supported'.format(ndims))

    return _eig_minimize_locally2(op, list(eigvec_ltens), eigs)

# @profile
def eig(mpo, num_sweeps, var_sites=2,
        startvec=None, startvec_rank=None, randstate=None, eigs=None):
    r"""Iterative search for MPO eigenvalues

    .. note::

       This function can return completely inaccurate values. You are
       responsible for supplying a large enough :code:`startvec_rank`
       (or ``startvec`` with large enough rank) and
       :code:`num_sweeps`.

    This function attempts to find eigenvalues by iteratively
    optimizing :math:`\lambda = \langle \psi \vert H \vert \psi
    \rangle` where :math:`H` is the operator supplied in the argument
    :code:`mpo`.  Specifically, we attempt to de- or increase
    :math:`\lambda` by optimizing over several neighbouring local
    tensors of the MPS :math:`\vert \psi \rangle` simultaneously (the
    number given by :code:`var_sites`).

    The algorithm used here is described e.g. in
    [:ref:`Sch11 <Sch11>`, Sec. 6.3].
    For :code:`var_sites = 1`, it is called "variational MPS ground state
    search" or "single-site DMRG" [:ref:`Sch11 <Sch11>`, Sec. 6.3, p. 69]. For
    :code:`var_sites > 1`, it is called "multi-site DMRG".

    :param MPArray mpo: A matrix product operator (MPA with two physical legs)
    :param int num_sweeps: Number of sweeps to do (required)
    :param int var_sites: Number of neighbouring sites to be varied
        simultaneously
    :param startvec: Initial guess for eigenvector (default: random MPS with
        rank `startvec_rank`)
    :param startvec_rank: Rank of random start vector (required and
        used only if no start vector is given)
    :param randstate: ``numpy.random.RandomState`` instance or ``None``
    :param eigs: Function which computes one eigenvector of the local
        eigenvalue problem on :code:`var_sites` sites

    :returns: eigval, eigvec_mpa

    The :code:`eigs` parameter defaults to

    .. code-block:: python

       eigs = functools.partial(scipy.sparse.linalg.eigsh, k=1, tol=1e-6)

    By default, :func:`eig` computes the eigenvalue with largest
    magnitude. To compute e.g. the smallest eigenvalue (sign
    included), supply :code:`which='SA'` to ``eigsh``. For other
    possible values, refer to the SciPy documentation.

    It is recommendable to supply a value for the :code:`tol`
    parameter of :code:`eigsh()`. Otherwise, :code:`eigsh()` will work
    at machine precision which is rarely necessary.

    .. note::

       One should keep in mind that a variational method (such as the
       one implemented in this function) can only provide e.g. an
       upper bound on the lowest eigenvalue of an MPO. Deciding
       whether a given MPO has an eigenvalue which is smaller than a
       given threshold has been shown to be NP-hard (in the number of
       parameters of the MPO representation) [KGE14]_.

    Comments on the implementation, for :code:`var_sites = 1`:

    References are to the arXiv version of [Sch11]_ assuming we replace
    zero-based with one-based indices there.

    .. code::plain

       leftvecs[i] is L_{i-1}  \
       rightvecs[i] is R_{i}   |  See Fig. 38 and Eq. (191) on p. 62.
       mpo[i] is W_{i}         /
       eigvec[i] is M_{i}         This is just the MPS matrix.

    :code:`Psi^A_{i-1}` and :code:`Psi^B_{i}` are identity matrices because of
    normalization. (See Fig. 42 on p. 67 and the text; see also
    Figs. 14 and 15 and pages 28 and 29.)

    """
    # Possible TODOs:
    #  - Can we refactor this function into several shorter functions?
    #  - compute the overlap between 'eigvec' from successive iterations
    #    to check whether we have converged
    #  - compute var(H) = <psi| H^2 |psi> - (<psi| H |psi>)^2 every n-th
    #    iteration to check whether we have converged (this criterion is
    #    better but more expensive to compute)
    #  - increase the rank of 'eigvec' if var(H) remains above
    #    a given threshold
    #  - for multi-site updates, track the error in the SVD truncation
    #    (see comment there why)
    #  - return these details for tracking errors in larger computations

    if eigs is None:
        eigs = ft.partial(sp.linalg.eigsh, k=1, tol=1e-6, which='LM')

    nr_sites = len(mpo)
    assert nr_sites - var_sites > 0, (
        'Require ({} =) nr_sites > var_sites (= {})'
        .format(nr_sites, var_sites))

    if startvec is None:
        if startvec_rank is None:
            raise ValueError('`startvec_rank` required if `startvec` is None')
        if startvec_rank == 1:
            raise ValueError('startvec_rank must be at least 2')
        # Choose `startvec` with complex entries because real matrices
        # can have non-real eigenvalues (conjugate pairs), implying
        # non-real eigenvectors. This matches numpy.linalg.eig's behaviour.
        shape = [(dim[0],) for dim in mpo.shape]
        startvec = random_mpa(nr_sites, shape, startvec_rank,
                              randstate=randstate, dtype=np.complex_)
        startvec.canonicalize(right=1)
        startvec /= mp.norm(startvec)
    else:
        # Do not modify the `startvec` argument.
        startvec = startvec.copy()
        # pass
    # Can we avoid this overly complex check by improving
    # _eig_minimize_locally()? eigs() will fail under the excluded
    # conditions because of too small matrices.
    assert not any(rank12 == (1, 1) for rank12 in
                   zip((1,) + startvec.ranks, startvec.ranks + (1,))), \
        ('startvec must not contain two consecutive ranks 1, '
         'ranks including dummy values = (1,) + {!r} + (1,)'
         .format(startvec.ranks))
    # For
    #
    #   pos in range(nr_sites - var_sites),
    #
    # we find the ground state of an operator supported on
    #
    #   range(pos, pos_end),  pos_end = pos + var_sites
    #
    # leftvecs[pos] and rightvecs[pos] contain the vectors needed to
    # construct that operator for that. Therefore, leftvecs[pos] is
    # constructed from matrices on
    #
    #   range(0, pos - 1)
    #
    # and rightvecs[pos] is constructed from matrices on
    #
    #   range(pos_end, nr_sites),  pos_end = pos + var_sites
    eigvec = startvec
    eigvec.canonicalize(right=1)
    leftvecs = [np.array(1, ndmin=3)] + [None] * (nr_sites - var_sites)
    rightvecs = [None] * (nr_sites - var_sites) + [np.array(1, ndmin=3)]
    for pos in reversed(range(nr_sites - var_sites)):
        rightvecs[pos] = _eig_rightvec_add(rightvecs[pos + 1],
                                           mpo.lt[pos + var_sites],
                                           eigvec.lt[pos + var_sites])

    # The iteration pattern is very similar to
    # :func:`mpnum.mparray.MPArray._adapt_to()`. See there for more
    # comments.
    for num_sweep in range(num_sweeps):
        # Sweep from left to right
        for pos in range(nr_sites - var_sites + 1):
            if pos == 0 and num_sweep > 0:
                # Don't do first site again if we are not in the first sweep.
                continue

            if pos > 0:
                eigvec.canonicalize(left=pos)
                rightvecs[pos - 1] = None
                leftvecs[pos] = _eig_leftvec_add(
                    leftvecs[pos - 1], mpo.lt[pos - 1], eigvec.lt[pos - 1])
            pos_end = pos + var_sites
            eigval, eigvec_lten = _eig_minimize_locally(
                leftvecs[pos], mpo.lt[pos:pos_end], rightvecs[pos],
                eigvec.lt[pos:pos_end], eigs)
            eigvec.lt[pos:pos_end] = eigvec_lten

        # Sweep from right to left (don't do last site again)
        for pos in reversed(range(nr_sites - var_sites)):
            pos_end = pos + var_sites
            if pos < nr_sites - var_sites:
                # We always do this, because we don't do the last site again.
                eigvec.canonicalize(right=pos_end)
                leftvecs[pos + 1] = None
                rightvecs[pos] = _eig_rightvec_add(
                    rightvecs[pos + 1], mpo.lt[pos_end], eigvec.lt[pos_end])
            eigval, eigvec_lten = _eig_minimize_locally(
                leftvecs[pos], mpo.lt[pos:pos_end], rightvecs[pos],
                eigvec.lt[pos:pos_end], eigs)
            eigvec.lt[pos:pos_end] = eigvec_lten

        # gc.collect()

    return eigval, eigvec


def eig_sum(mpas, num_sweeps, var_sites=2,
            startvec=None, startvec_rank=None, randstate=None, eigs=None):
    r"""Iterative search for eigenvalues of a sum of MPOs/MPSs

    Try to compute the ground state of the sum of the objects in
    ``mpas``. MPOs are taken as-is. An MPS :math:`\vert\psi\rangle`
    adds :math:`\vert\psi\rangle \langle\psi\vert` to the sum.

    This function executes the same algorithm as :func:`eig` applied
    to an uncompressed MPO sum of the elements in ``mpas``, but it
    obtains the ingredients for the local optimization steps using
    less memory and execution time. In particular, this function does
    not have to convert an MPS in ``mpas`` to an MPO.

    .. todo:: Add information on how the runtime of :func:`eig` and
              :func:`eig_sum` scale with the the different ranks. For
              the time being, refer to the benchmark test.

    :param mpas: A sequence of MPOs or MPSs

    Remaining parameters and description: See :func:`eig`.

    Algorithm: [:ref:`Sch11 <Sch11>`, Sec. 6.3]

    """
    # Possible TODOs: See :func:`eig`
    if eigs is None:
        eigs = ft.partial(sp.linalg.eigsh, k=1, tol=1e-6)

    mpas = list(mpas)
    nr_mpas = len(mpas)
    nr_sites = len(mpas[0])
    assert all(len(m) == nr_sites for m in mpas)
    ndims = [m.ndims[0] for m in mpas]
    assert nr_sites - var_sites > 0, (
        'Require ({} =) nr_sites > var_sites (= {})'
        .format(nr_sites, var_sites))

    if startvec is None:
        if startvec_rank is None:
            raise ValueError('`startvec_rank` required if `startvec` is None')
        if startvec_rank == 1:
            raise ValueError('startvec_rank must be at least 2')
        shape = [(dim[0],) for dim in mpas[0].shape]
        startvec = random_mpa(nr_sites, shape, startvec_rank,
                              randstate=randstate, dtype=np.complex_)
        startvec.canonicalize(right=1)
        startvec /= mp.norm(startvec)
    else:
        # Do not modify the `startvec` argument.
        startvec = startvec.copy()
    # Can we avoid this overly complex check by improving
    # _eig_minimize_locally()? eigs() will fail under the excluded
    # conditions because of too small matrices.
    assert not any(rank12 == (1, 1) for rank12 in
                   zip((1,) + startvec.ranks, startvec.ranks + (1,))), \
        ('startvec must not contain two consecutive ranks 1, '
         'ranks including dummy values = (1,) + {!r} + (1,)'
         .format(startvec.ranks))
    # For
    #
    #   pos in range(nr_sites - var_sites),
    #
    # we find the ground state of an operator supported on
    #
    #   range(pos, pos_end),  pos_end = pos + var_sites
    #
    # leftvecs[pos] and rightvecs[pos] contain the vectors needed to
    # construct that operator for that. Therefore, leftvecs[pos] is
    # constructed from matrices on
    #
    #   range(0, pos - 1)
    #
    # and rightvecs[pos] is constructed from matrices on
    #
    #   range(pos_end, nr_sites),  pos_end = pos + var_sites
    eigvec = startvec
    eigvec.canonicalize(right=1)
    leftvecs = [[np.array(1, ndmin=1 + pl) for pl in ndims]]
    leftvecs.extend([None] * nr_mpas for _ in range(nr_sites - var_sites))
    rightvecs = [[None] * nr_mpas for _ in range(nr_sites - var_sites)]
    rightvecs.append(leftvecs[0][:])
    for pos in reversed(range(nr_sites - var_sites)):
        _eig_sum_rightvec_add(
            mpas, ndims, rightvecs[pos], rightvecs[pos + 1],
            pos + var_sites, eigvec.lt[pos + var_sites])

    # The iteration pattern is very similar to
    # :func:`mpnum.mparray.MPArray._adapt_to()`. See there for more
    # comments.
    for num_sweep in range(num_sweeps):
        # Sweep from left to right
        for pos in range(nr_sites - var_sites + 1):
            if pos == 0 and num_sweep > 0:
                # Don't do first site again if we are not in the first sweep.
                continue

            if pos > 0:
                eigvec.canonicalize(left=pos)
                rightvecs[pos - 1] = [None] * nr_mpas
                _eig_sum_leftvec_add(
                    mpas, ndims, leftvecs[pos], leftvecs[pos - 1],
                    pos - 1, eigvec.lt[pos - 1])
            pos_end = pos + var_sites
            eigval, eigvec_lten = _eig_sum_minimize_locally(
                mpas, ndims, leftvecs[pos], slice(pos, pos_end), rightvecs[pos],
                eigvec.lt[pos:pos_end], eigs)
            eigvec.lt[pos:pos_end] = eigvec_lten

        # Sweep from right to left (don't do last site again)
        for pos in reversed(range(nr_sites - var_sites)):
            pos_end = pos + var_sites
            if pos < nr_sites - var_sites:
                # We always do this, because we don't do the last site again.
                eigvec.canonicalize(right=pos_end)
                leftvecs[pos + 1] = [None] * nr_mpas
                _eig_sum_rightvec_add(
                    mpas, ndims, rightvecs[pos], rightvecs[pos + 1],
                    pos_end, eigvec.lt[pos_end])
            eigval, eigvec_lten = _eig_sum_minimize_locally(
                mpas, ndims, leftvecs[pos], slice(pos, pos_end), rightvecs[pos],
                eigvec.lt[pos:pos_end], eigs)
            eigvec.lt[pos:pos_end] = eigvec_lten

    return eigval, eigvec
