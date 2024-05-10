from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from scipy.constants import electron_volt
from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalPositionBasis,
    FundamentalTransformedPositionBasis,
    FundamentalTransformedPositionBasis1d,
    TransformedPositionBasis1d,
)
from surface_potential_analysis.basis.evenly_spaced_basis import (
    EvenlySpacedTransformedPositionBasis,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)
from surface_potential_analysis.operator.conversion import (
    convert_operator_list_to_basis,
    sample_operator,
)
from surface_potential_analysis.potential.conversion import convert_potential_to_basis
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.util.decorators import npy_cached_dict

from caldeira_legget_examples.util import (
    get_caldeira_leggett_noise_operator,
    get_eta,
    get_noise_operators_sampled,
    get_potential_noise_kernel,
)
from caldeira_legget_examples.util import (
    get_noise_operators as get_noise_operators_generic,
)

if TYPE_CHECKING:
    from surface_potential_analysis.kernel.kernel import (
        SingleBasisDiagonalNoiseKernel,
        SingleBasisNoiseOperatorList,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.potential.potential import Potential

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)

LATTICE_CONSTANT = 3.615 * 10**-10
BARRIER_ENERGY = 55 * 10**-3 * electron_volt
SODIUM_MASS = 3.8175458e-26
SODIUM_GAMMA = 0.2e12
SODIUM_ETA = get_eta(SODIUM_GAMMA, SODIUM_MASS)


def get_potential() -> (
    Potential[StackedBasis[FundamentalTransformedPositionBasis1d[Literal[3]]]]
):
    delta_x = np.sqrt(3) * LATTICE_CONSTANT / 2
    axis = FundamentalTransformedPositionBasis1d[Literal[3]](np.array([delta_x]), 3)
    vector = 0.25 * BARRIER_ENERGY * np.array([2, -1, -1]) * np.sqrt(3)
    return {"basis": StackedBasis(axis), "data": vector}


def get_interpolated_potential(
    resolution: tuple[_L0Inv],
) -> Potential[
    StackedBasisLike[FundamentalTransformedPositionBasis[_L0Inv, Literal[1]]]
]:
    potential = get_potential()
    old = potential["basis"][0]
    basis = StackedBasis(
        TransformedPositionBasis1d[_L0Inv, Literal[3]](
            old.delta_x,
            old.n,
            resolution[0],
        ),
    )
    scaled_potential = potential["data"] * np.sqrt(resolution[0] / old.n)
    return convert_potential_to_basis(
        {"basis": basis, "data": scaled_potential},
        stacked_basis_as_fundamental_momentum_basis(basis),
    )


def get_extended_interpolated_potential(
    shape: tuple[_L0Inv],
    resolution: tuple[_L1Inv],
) -> Potential[
    StackedBasisLike[
        EvenlySpacedTransformedPositionBasis[_L1Inv, _L0Inv, Literal[0], Literal[1]]
    ]
]:
    interpolated = get_interpolated_potential(resolution)
    old = interpolated["basis"][0]
    basis = StackedBasis(
        EvenlySpacedTransformedPositionBasis[_L1Inv, _L0Inv, Literal[0], Literal[1]](
            old.delta_x * shape[0],
            n=old.n,
            step=shape[0],
            offset=0,
        ),
    )
    scaled_potential = interpolated["data"] * np.sqrt(basis.fundamental_n / old.n)
    if shape[0] % 2 == 1:
        # Place a minima at the center
        scaled_potential *= np.exp(1j * np.pi)

    return {"basis": basis, "data": scaled_potential}


def get_hamiltonian(
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
) -> SingleBasisOperator[StackedBasisLike[FundamentalPositionBasis[int, Literal[1]]],]:
    potential = get_extended_interpolated_potential(shape, (resolution[0] + 2,))
    converted = convert_potential_to_basis(
        potential,
        stacked_basis_as_fundamental_position_basis(potential["basis"]),
    )
    return sample_operator(
        total_surface_hamiltonian(converted, SODIUM_MASS, np.array([0])),
        sample=tuple(s * r for s, r in zip(shape, resolution, strict=True)),
    )


def get_noise_operator(
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
    temperature: float,
) -> SingleBasisOperator[StackedBasisLike[FundamentalPositionBasis[int, Literal[1]]]]:
    hamiltonain = get_hamiltonian(shape, resolution)
    return get_caldeira_leggett_noise_operator(
        hamiltonain["basis"][0],
        SODIUM_MASS,
        temperature,
        SODIUM_GAMMA,
    )


def get_noise_kernel(
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
    temperature: float,
) -> SingleBasisDiagonalNoiseKernel[
    StackedBasisLike[FundamentalPositionBasis[Any, Literal[1]]],
]:
    hamiltonian = get_hamiltonian(shape, resolution)
    return get_potential_noise_kernel(
        hamiltonian["basis"][0],
        SODIUM_ETA,
        temperature,
    )


def get_noise_operators_cache(
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
    temperature: float,
) -> Path:
    return Path(f"noise_operators_{shape[0]}_{resolution[0]}_{temperature}")


@npy_cached_dict(get_noise_operators_cache, load_pickle=True)
def get_noise_operators(
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
    temperature: float,
) -> SingleBasisNoiseOperatorList[
    FundamentalBasis[int],
    StackedBasisLike[FundamentalPositionBasis[Any, Literal[1]]],
]:
    hamiltonian_large = get_hamiltonian(shape, (resolution[0] * 2,))
    full_operators = get_noise_operators_generic(
        hamiltonian_large,
        SODIUM_ETA,
        temperature,
    )
    sampled_operators = get_noise_operators_sampled(full_operators)
    hamiltonian = get_hamiltonian(shape, resolution)
    converted = convert_operator_list_to_basis(sampled_operators, hamiltonian["basis"])
    return {
        "basis": converted["basis"],
        "data": converted["data"],
        "eigenvalue": sampled_operators["eigenvalue"],
    }
