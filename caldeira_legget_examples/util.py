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
    convert_operator_to_basis,
)
from surface_potential_analysis.operator.operator import (
    add_operator,
    as_operator,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator import SingleBasisOperator

_SB0 = TypeVar(
    "_SB0",
    bound=StackedBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]],
)


def get_caldeira_leggett_noise_operator(
    basis: _SB0,
    mass: float,
    temperature: float,
    gamma: float,
) -> SingleBasisOperator[_SB0]:
    r"""Get the Caldeira Leggett Noise operator.

    \hat{A} = \sqrt{\gamma}(\sqrt{4 m kT} \hat{x} + i\sqrt{1/{4mkT}} \hat{p})

    Parameters
    ----------
    basis : _SB0
    mass : float
    temperature : float
    gamma : float

    Returns
    -------
    SingleBasisOperator[_SB0]
        _description_

    """
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


def get_eta(gamma: float, mass: float) -> float:
    return 2 * mass * gamma
