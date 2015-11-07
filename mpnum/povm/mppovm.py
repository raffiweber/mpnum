#!/usr/bin/env python
# encoding: utf-8
from __future__ import absolute_import, division, print_function

import itertools as it

import mpnum.mparray as mp
import mpnum.mpsmpo as mpsmpo


class MPPovm(mp.MPArray):
    """MPArray representation of multipartite POVM

    There are two different ways to write down a POVM in matrix product form

    1) As a list of matrix product operators, where each entry corresponds to
        a single POVM element

    2) As a matrix proudct array with 3 physical legs:

                [POVM index, column index, row index]

        that is, the first physical leg of the MPArray corresponds to the index
        of the POVM element. This representation is especially helpful for
        computing expectation values with MPSs/MPDOs.

    Here, we choose the second.
    """

    def __iter__(self):
        """Returns an iterator over all POVM elements. The result is the i-th
        POVM element in MPO form.
        """
        return self.paxis_iter(axes=0)

    @classmethod
    def from_local_povm(cls, lelems, width):
        """Generates a product POVM on `width` sites.

        :param lelems: POVM elements as an iterator over all local elements
            (i.e. an iterator over numpy arrays representing the latter)
        :param int width: Number of sites the POVM lives on
        :returns: :class:`MPPovm` which is a product POVM of the `lelems`

        """
        return cls.from_kron(it.repeat(lelems, width))

    @property
    def probability_map(self):
        """Map that takes a raveled density matrix to the POVM probabilities
        """
        return mp.MPArray(mp._local_reshape(ten, (ten.shape[1], -1)).conj()
                          for ten in self._ltens)

    def expectations(self, mpa):
        """Computes the exp. values of the POVM elements with given state

        :param mpa: MPDO
        :returns: Iterator over the expectation values, the n-th element is
            the expectation value correponding to the reduced state on sites
            [n,...,n + len(self) - 1]

        """
        assert len(self) <= len(mpa)

        if all(pleg == 1 for pleg in mpa.plegs):
            raise NotImplementedError("MPS expectations coming soon")
        elif all(pleg == 2 for pleg in mpa.plegs):
            pmap = self.probability_map
            for rho_red in mpsmpo.reductions_mpo(mpa, len(self)):
                yield mp.dot(pmap, rho_red.ravel())
            return

        raise ValueError("Could not understand data dype.")
