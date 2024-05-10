from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from scipy.constants import Boltzmann, hbar
from surface_potential_analysis.basis.basis_like import (
    BasisLike,
    BasisWithLengthLike,
)
from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_position_basis,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.kernel.kernel import (
    get_single_factorized_noise_operators_diagonal,
)
from surface_potential_analysis.operator.conversion import (
    convert_operator_to_basis,
    sample_operator_list,
)
from surface_potential_analysis.operator.operator import (
    add_operator,
    as_operator,
)
from surface_potential_analysis.operator.operator_list import (
    SingleBasisOperatorList,
    add_list_list,
    as_operator_list,
    get_commutator_operator_list,
    scale_operator_list,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalPositionBasis,
    )
    from surface_potential_analysis.kernel.kernel import (
        SingleBasisDiagonalNoiseKernel,
        SingleBasisDiagonalNoiseOperatorList,
        SingleBasisNoiseOperatorList,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator

_SB0 = TypeVar(
    "_SB0",
    bound=StackedBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]],
)
_B1 = TypeVar(
    "_B1",
    bound=BasisLike[Any, Any],
)

_B0 = TypeVar(
    "_B0",
    bound=BasisLike[Any, Any],
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


def get_displacements(
    n_x_points: np.ndarray[tuple[int], np.dtype[np.int_]],
) -> np.ndarray[tuple[int, int], np.dtype[np.int_]]:
    n = n_x_points.size
    delta = n_x_points[:, np.newaxis] - n_x_points[np.newaxis, :]

    return (delta + n // 2) % n - (n // 2)


def get_eta(gamma: float, mass: float) -> float:
    return 2 * mass * gamma


def get_potential_noise_kernel(
    basis: StackedBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]],
    eta: float,
    temperature: float,
    *,
    lambda_factor: float = 2 * np.sqrt(2),
) -> SingleBasisDiagonalNoiseKernel[
    StackedBasisLike[FundamentalPositionBasis[Any, Literal[1]]]
]:
    basis_x = StackedBasis(basis_as_fundamental_position_basis(basis[0]))

    util = BasisUtil(basis_x[0])
    n_x_points = util.nx_points
    displacements = get_displacements(n_x_points) * util.dx

    lambda_ = np.max(np.abs(displacements)) / lambda_factor
    # mu = A / lambda
    mu = np.sqrt(2 * eta * Boltzmann * temperature / hbar**2)
    a = mu * lambda_
    correlation = a**2 * np.exp(-(displacements**2) / (2 * lambda_**2)).astype(
        np.complex128,
    )

    return {
        "basis": StackedBasis(
            StackedBasis(basis_x, basis_x),
            StackedBasis(basis_x, basis_x),
        ),
        "data": correlation.ravel(),
    }


def get_temperature_corrected_operators(
    hamiltonian: SingleBasisOperator[_B1],
    operators: SingleBasisOperatorList[
        _B0,
        _B1,
    ],
    temperature: float,
) -> SingleBasisOperatorList[
    _B0,
    _B1,
]:
    commutator = get_commutator_operator_list(hamiltonian, operators)
    correction = scale_operator_list(-1 / (4 * Boltzmann * temperature), commutator)
    return add_list_list(operators, correction)


def get_temperature_corrected_noise_operators(
    hamiltonian: SingleBasisOperator[_B1],
    operators: SingleBasisDiagonalNoiseOperatorList[
        _B0,
        _B1,
    ],
    temperature: float,
) -> SingleBasisNoiseOperatorList[
    _B0,
    _B1,
]:
    operators_full = as_operator_list(operators)
    corrected_operators = get_temperature_corrected_operators(
        hamiltonian,
        operators_full,
        temperature,
    )

    return {
        "basis": corrected_operators["basis"],
        "data": corrected_operators["data"],
        "eigenvalue": operators["eigenvalue"],
    }


def get_noise_operators(
    hamiltonian: SingleBasisOperator[
        StackedBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]]
    ],
    eta: float,
    temperature: float,
) -> SingleBasisNoiseOperatorList[
    FundamentalBasis[int],
    StackedBasisLike[FundamentalPositionBasis[Any, Literal[1]]],
]:
    """Get the noise operators for a gausssian kernel in the given basis.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_BL0]
    mass : float
    temperature : float
    gamma : float

    Returns
    -------
    SingleBasisNoiseOperatorList[
        FundamentalBasis[int],
        FundamentalPositionBasis[Any, Literal[1]],
    ]

    """
    kernel = get_potential_noise_kernel(
        hamiltonian["basis"][0],
        eta,
        temperature,
    )
    operators = get_single_factorized_noise_operators_diagonal(kernel)
    hamiltonian_converted = convert_operator_to_basis(
        hamiltonian,
        operators["basis"][1],
    )

    return get_temperature_corrected_noise_operators(
        hamiltonian_converted,
        operators,
        temperature,
    )


def get_noise_operators_sampled(
    operators: SingleBasisNoiseOperatorList[_B0, _SB0],
    *,
    n: int | None = None,
) -> SingleBasisNoiseOperatorList[_B0, Any]:
    """Given a set of noise operators, get the equivalent operators in a sampled basis.

    This is useful to remove the periodicity in momentum space.
    This removes any scattering between neighboring k from +k_n/2 to -k_n/2.
    This is because these states are far apart in the large (over sampled) basis.


    Returns
    -------
    SingleBasisNoiseOperatorList[_B0, _SB0]

    """
    sampled = sample_operator_list(
        operators,
        sample=(operators["basis"][1][0].fundamental_n // 2 if n is None else n,),
    )

    return {
        "basis": sampled["basis"],
        "data": sampled["data"],
        "eigenvalue": operators["eigenvalue"],
    }
