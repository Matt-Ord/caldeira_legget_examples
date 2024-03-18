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
from surface_potential_analysis.kernel.kernel import (
    get_noise_kernel,
    get_single_factorized_noise_operators_diagonal,
)
from surface_potential_analysis.operator.conversion import (
    convert_operator_list_to_basis,
    convert_operator_to_basis,
)
from surface_potential_analysis.operator.operator import (
    add_operator,
    apply_function_to_operator,
    as_operator,
)
from surface_potential_analysis.operator.operator_list import (
    as_operator_list,
    matmul_list_operator,
    matmul_operator_list,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import FundamentalPositionBasis
    from surface_potential_analysis.kernel.kernel import (
        SingleBasisDiagonalNoiseKernel,
        SingleBasisNoiseKernel,
    )
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


def get_displacements(
    n_x_points: np.ndarray[tuple[int], np.dtype[np.int_]],
) -> np.ndarray[tuple[int, int], np.dtype[np.int_]]:
    n = n_x_points.size
    delta = n_x_points[:, np.newaxis] - n_x_points[np.newaxis, :]

    return (delta + n // 2) % n - (n // 2)


def get_potential_noise_kernel(
    basis: StackedBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]],
    mass: float,
    temperature: float,
    gamma: float,
) -> SingleBasisDiagonalNoiseKernel[
    StackedBasisLike[FundamentalPositionBasis[Any, Literal[1]]]
]:
    mu = np.sqrt(4 * mass * Boltzmann * temperature / hbar**2)
    beta = mu**2 * gamma / 2

    basis_x = stacked_basis_as_fundamental_position_basis(basis)

    util = BasisUtil(basis_x[0])
    n_x_points = util.nx_points
    displacements = get_displacements(n_x_points) * util.dx
    correlation = np.exp(-((displacements * beta) ** 2) / 2).astype(np.complex128)

    return {
        "basis": StackedBasis(
            StackedBasis(basis_x, basis_x),
            StackedBasis(basis_x, basis_x),
        ),
        "data": correlation.ravel(),
    }


def get_full_noise_kernel(
    hamiltonian: SingleBasisOperator[
        StackedBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]]
    ],
    mass: float,
    temperature: float,
    gamma: float,
) -> SingleBasisNoiseKernel[
    StackedBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]]
]:
    potential_kernel = get_potential_noise_kernel(
        hamiltonian["basis"][0],
        mass,
        temperature,
        gamma,
    )
    diagonal_operators = get_single_factorized_noise_operators_diagonal(
        potential_kernel,
    )
    operators = as_operator_list(
        diagonal_operators,
    )
    converted = convert_operator_list_to_basis(operators, hamiltonian["basis"])

    lhs = apply_function_to_operator(
        hamiltonian,
        lambda x: np.exp(-x / (temperature * Boltzmann)),
    )
    rhs = apply_function_to_operator(
        hamiltonian,
        lambda x: np.exp(x / (temperature * Boltzmann)),
    )
    rwa_operators = matmul_list_operator(
        matmul_operator_list(lhs, converted),
        rhs,
    )

    return get_noise_kernel(
        {
            "basis": rwa_operators["basis"],
            "data": rwa_operators["data"],
            "eigenvalue": diagonal_operators["eigenvalue"],
        },
    )
