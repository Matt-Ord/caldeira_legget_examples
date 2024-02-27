from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from scipy.constants import Boltzmann, hbar
from surface_potential_analysis.basis.basis_like import (
    BasisWithLengthLike,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.operator.conversion import (
    add_operator,
    convert_operator_to_basis,
)
from surface_potential_analysis.operator.operator import as_operator
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator import SingleBasisOperator

_B0Inv = TypeVar(
    "_B0Inv",
    bound=StackedBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]],
)


def get_noise_operator(
    basis: _B0Inv,
    mass: float,
    temperature: float,
    gamma: float,
) -> SingleBasisOperator[_B0Inv]:
    basis_x = stacked_basis_as_fundamental_position_basis(basis)
    mu = np.sqrt(4 * mass * Boltzmann * temperature / hbar**2)
    x_noise = convert_operator_to_basis(
        as_operator(
            {
                "basis": StackedBasis(basis_x, basis_x),
                "data": (hbar * np.sqrt(gamma) * mu)
                * BasisUtil(basis_x[0]).x_points[0].astype(np.complex128),
            },
        ),
        StackedBasis(basis, basis),
    )

    basis_k = stacked_basis_as_fundamental_momentum_basis(basis)

    nu = 1 / mu
    k_noise = convert_operator_to_basis(
        as_operator(
            {
                "basis": StackedBasis(basis_k, basis_k),
                "data": (hbar * np.sqrt(gamma) * nu)
                * BasisUtil(basis_k).k_points[0].astype(np.complex128),
            },
        ),
        StackedBasis(basis, basis),
    )

    return add_operator(k_noise, x_noise)
