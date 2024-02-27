from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from scipy.constants import electron_volt
from surface_potential_analysis.basis.basis import (
    FundamentalTransformedPositionBasis,
    FundamentalTransformedPositionBasis1d,
    TransformedPositionBasis1d,
)
from surface_potential_analysis.basis.basis_like import (
    BasisWithLengthLike,
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
from surface_potential_analysis.potential.conversion import convert_potential_to_basis
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
)

from localized_sse.calderia_leggett import get_noise_operator as get_noise_operator_cl

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.potential.potential import Potential

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_B0Inv = TypeVar(
    "_B0Inv",
    bound=StackedBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]],
)

LATTICE_CONSTANT = 3.615 * 10**-10
BARRIER_ENERGY = 55 * 10**-3 * electron_volt
SODIUM_MASS = 3.8175458e-26
SODIUM_GAMMA = 0.2e12


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

    return {"basis": basis, "data": scaled_potential}


def get_hamiltonian(
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
) -> SingleBasisOperator[
    StackedBasisLike[FundamentalTransformedPositionBasis[int, Literal[1]]],
]:
    potential = get_extended_interpolated_potential(shape, resolution)
    converted = convert_potential_to_basis(
        potential,
        stacked_basis_as_fundamental_momentum_basis(potential["basis"]),
    )
    return total_surface_hamiltonian(converted, SODIUM_MASS, np.array([0]))


def get_noise_operator(
    basis: _B0Inv,
    temperature: float,
) -> SingleBasisOperator[_B0Inv]:
    return get_noise_operator_cl(basis, SODIUM_MASS, temperature, SODIUM_GAMMA)
