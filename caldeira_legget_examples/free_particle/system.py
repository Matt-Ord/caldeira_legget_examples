from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from scipy.constants import hbar
from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalPositionBasis,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)
from surface_potential_analysis.kernel.gaussian import (
    get_effective_gaussian_noise_kernel,
)
from surface_potential_analysis.kernel.gaussian import (
    get_effective_gaussian_noise_operators as get_noise_operators_generic,
)
from surface_potential_analysis.kernel.kernel import (
    get_noise_operators_sampled,
)
from surface_potential_analysis.operator.conversion import (
    convert_operator_list_to_basis,
)
from surface_potential_analysis.operator.operator_list import select_operator

if TYPE_CHECKING:
    from surface_potential_analysis.kernel.kernel import (
        SingleBasisDiagonalNoiseKernel,
        SingleBasisNoiseOperatorList,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.potential.potential import Potential

    _L0Inv = TypeVar("_L0Inv", bound=int)


ATOM_MASS = (10 * hbar) ** 2
ATOM_GAMMA = 1 / 120
ATOM_ETA = 2 * ATOM_MASS * ATOM_GAMMA


def get_potential(
    size: _L0Inv,
) -> Potential[StackedBasisLike[FundamentalPositionBasis[_L0Inv, Literal[1]]]]:
    basis = StackedBasis(FundamentalPositionBasis(np.array([1]), size))

    data = np.zeros(basis.shape, dtype=np.complex128)
    return {"basis": basis, "data": data}


def get_hamiltonian(
    size: _L0Inv,
) -> SingleBasisOperator[
    StackedBasisLike[FundamentalPositionBasis[_L0Inv, Literal[1]]]
]:
    potential = get_potential(size)
    return total_surface_hamiltonian(potential, ATOM_MASS, np.array([0]))


def get_potential_noise_kernel(
    size: _L0Inv,
    temperature: float,
) -> SingleBasisDiagonalNoiseKernel[
    StackedBasisLike[FundamentalPositionBasis[_L0Inv, Literal[1]]]
]:
    hamiltonian = get_hamiltonian(size)
    return get_effective_gaussian_noise_kernel(
        hamiltonian["basis"][0],
        ATOM_ETA,
        temperature,
    )


def get_noise_operators(
    size: _L0Inv,
    temperature: float,
) -> SingleBasisNoiseOperatorList[
    FundamentalBasis[int],
    StackedBasisLike[FundamentalPositionBasis[_L0Inv, Literal[1]]],
]:
    hamiltonian = get_hamiltonian(size)
    return get_noise_operators_generic(
        hamiltonian,
        ATOM_ETA,
        temperature,
    )


def get_non_periodic_noise_operators(
    size: _L0Inv,
    temperature: float,
) -> SingleBasisNoiseOperatorList[
    FundamentalBasis[int],
    StackedBasisLike[FundamentalPositionBasis[_L0Inv, Literal[1]]],
]:
    hamiltonian_large = get_hamiltonian(2 * size)
    full_operators = get_noise_operators_generic(
        hamiltonian_large,
        ATOM_ETA,
        temperature,
    )
    sampled_operators = get_noise_operators_sampled(full_operators, n=size)
    hamiltonian = get_hamiltonian(size)
    converted = convert_operator_list_to_basis(sampled_operators, hamiltonian["basis"])
    return {
        "basis": converted["basis"],
        "data": converted["data"],
        "eigenvalue": sampled_operators["eigenvalue"],
    }


def get_most_significant_noise_operator(
    size: _L0Inv,
    temperature: float,
) -> SingleBasisOperator[
    StackedBasisLike[FundamentalPositionBasis[_L0Inv, Literal[1]]]
]:
    operators = get_noise_operators(size, temperature)

    idx = np.argmax(np.abs(operators["eigenvalue"]))
    operator = select_operator(
        operators,
        idx=idx,
    )

    operator["data"] *= np.lib.scimath.sqrt(operators["eigenvalue"][idx])
    return operator
