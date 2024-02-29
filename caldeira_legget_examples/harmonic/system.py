from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from scipy.constants import hbar
from surface_potential_analysis.basis.basis import (
    FundamentalPositionBasis,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)

from caldeira_legget_examples.util import get_noise_operator as get_noise_operator_cl

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import (
        BasisWithLengthLike,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.potential.potential import Potential

    _B0Inv = TypeVar(
        "_B0Inv",
        bound=StackedBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]],
    )
    _L0Inv = TypeVar("_L0Inv", bound=int)

ATOM_OMEGA = 1 / hbar
ATOM_MASS = hbar**2
ATOM_GAMMA = 1e-2 / hbar


def get_potential(
    size: _L0Inv,
) -> Potential[StackedBasisLike[FundamentalPositionBasis[_L0Inv, Literal[1]]]]:
    basis = StackedBasis(FundamentalPositionBasis(np.array([50]), size))

    data = np.zeros(basis.shape, dtype=np.complex128)
    data = (
        0.5
        * ATOM_MASS
        * ATOM_OMEGA**2
        * (BasisUtil(basis).x_points_stacked[0] - basis[0].delta_x / 2) ** 2
    )

    return {"basis": basis, "data": data}


def get_hamiltonian(
    size: _L0Inv,
) -> SingleBasisOperator[
    StackedBasisLike[FundamentalPositionBasis[_L0Inv, Literal[1]]]
]:
    potential = get_potential(size)
    return total_surface_hamiltonian(potential, ATOM_MASS, np.array([0]))


def get_noise_operator(
    basis: _B0Inv,
    temperature: float,
) -> SingleBasisOperator[_B0Inv]:
    return get_noise_operator_cl(basis, ATOM_MASS, temperature, ATOM_GAMMA)
