from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from scipy.constants import hbar
from surface_potential_analysis.basis.basis import (
    FundamentalPositionBasis,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)
from surface_potential_analysis.kernel.kernel import (
    get_single_factorized_noise_operators,
)
from surface_potential_analysis.operator.operator_list import select_operator

from caldeira_legget_examples.util import (
    get_full_noise_kernel as get_full_noise_kernel_generic,
)
from caldeira_legget_examples.util import (
    get_potential_noise_kernel as get_potential_noise_kernel_generic,
)

if TYPE_CHECKING:
    from surface_potential_analysis.kernel.kernel import (
        SingleBasisDiagonalNoiseKernel,
        SingleBasisNoiseKernel,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.potential.potential import Potential

    _L0Inv = TypeVar("_L0Inv", bound=int)


ATOM_MASS = (10 * hbar) ** 2
ATOM_GAMMA = 1 / 300


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
    return get_potential_noise_kernel_generic(
        hamiltonian["basis"][0],
        ATOM_MASS,
        temperature,
        ATOM_GAMMA,
    )


def get_full_noise_kernel(
    size: _L0Inv,
    temperature: float,
) -> SingleBasisNoiseKernel[
    StackedBasisLike[FundamentalPositionBasis[_L0Inv, Literal[1]]]
]:
    hamiltonian = get_hamiltonian(size)
    return get_full_noise_kernel_generic(
        hamiltonian,
        ATOM_MASS,
        temperature,
        ATOM_GAMMA,
    )


def get_most_significant_noise_operator(
    size: _L0Inv,
    temperature: float,
) -> SingleBasisOperator[
    StackedBasisLike[FundamentalPositionBasis[_L0Inv, Literal[1]]]
]:
    kernel = get_full_noise_kernel(size, temperature)
    operators = get_single_factorized_noise_operators(kernel)
    idx = np.argmax(np.abs(operators["eigenvalue"]))
    operator = select_operator(
        operators,
        idx=idx,
    )

    operator["data"] *= np.lib.scimath.sqrt(operators["eigenvalue"][idx])
    return operator
