from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore unknown
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisWithVolumeLike,
    TupleBasis,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.operator.conversion import (
    convert_diagonal_operator_to_basis,
)
from surface_potential_analysis.operator.operator import (
    add_operator,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator import SingleBasisOperator

_B0 = TypeVar(
    "_B0",
    bound=StackedBasisWithVolumeLike[Any, Any, Any],
)


def get_eta(gamma: float, mass: float) -> float:
    return 2 * mass * gamma


def get_caldeira_leggett_noise_operator(
    basis: _B0,
    mass: float,
    temperature: float,
    gamma: float,
) -> SingleBasisOperator[_B0]:
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
    mu = np.sqrt(4 * mass * Boltzmann * temperature)
    x_noise = convert_diagonal_operator_to_basis(
        {
            "basis": TupleBasis(basis_x, basis_x),
            "data": (mu * np.sqrt(gamma) / hbar)
            * BasisUtil(basis_x[0]).x_points[0].astype(np.complex128),
        },
        TupleBasis(basis, basis),
    )

    basis_k = stacked_basis_as_fundamental_momentum_basis(basis)

    k_noise = convert_diagonal_operator_to_basis(
        {
            "basis": TupleBasis(basis_k, basis_k),
            "data": (hbar * np.sqrt(gamma) / mu)
            * BasisUtil(basis_k).k_points[0].astype(np.complex128),
        },
        TupleBasis(basis, basis),
    )

    return add_operator(k_noise, x_noise)
