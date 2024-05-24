from __future__ import annotations

from dataclasses import dataclass
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
    sample_operator,
)
from surface_potential_analysis.potential.conversion import convert_potential_to_basis
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)

from caldeira_legget_examples.util import (
    get_caldeira_leggett_noise_operator,
    get_eta,
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


@dataclass
class PeriodicSystem:
    """Represents the properties of a 1D Periodic System."""

    id: str
    """A unique ID, for use in caching"""
    barrier_energy: float
    lattice_constant: float
    mass: float
    gamma: float

    @property
    def eta(self) -> float:  # noqa: D102, ANN101
        return get_eta(self.gamma, self.mass)


SODIUM_COPPER_SYSTEM = PeriodicSystem(
    id="NaCu",
    barrier_energy=55e-3 * electron_volt,
    lattice_constant=3.615e-10,
    mass=3.8175458e-26,
    gamma=0.2e12,
)

LITHIUM_COPPER_SYSTEM = PeriodicSystem(
    id="LiCu",
    barrier_energy=45e-3 * electron_volt,
    lattice_constant=3.615e-10,
    mass=1.152414898e-26,
    gamma=1.2e12,
)

HYDROGEN_NICKEL_SYSTEM = PeriodicSystem(
    id="HNi",
    barrier_energy=55e-3 * electron_volt,  # TODO: find energy
    lattice_constant=3.615e-10,  # TODO: find constant
    mass=1.67e-27,
    gamma=0.2e12,
)


def get_potential(
    system: PeriodicSystem,
) -> Potential[StackedBasis[FundamentalTransformedPositionBasis1d[Literal[3]]]]:
    delta_x = np.sqrt(3) * system.lattice_constant / 2
    axis = FundamentalTransformedPositionBasis1d[Literal[3]](np.array([delta_x]), 3)
    vector = 0.25 * system.barrier_energy * np.array([2, -1, -1]) * np.sqrt(3)
    return {"basis": StackedBasis(axis), "data": vector}


def get_interpolated_potential(
    system: PeriodicSystem,
    resolution: tuple[_L0Inv],
) -> Potential[
    StackedBasisLike[FundamentalTransformedPositionBasis[_L0Inv, Literal[1]]]
]:
    potential = get_potential(system)
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
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L1Inv],
) -> Potential[
    StackedBasisLike[
        EvenlySpacedTransformedPositionBasis[_L1Inv, _L0Inv, Literal[0], Literal[1]]
    ]
]:
    interpolated = get_interpolated_potential(system, resolution)
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
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
) -> SingleBasisOperator[StackedBasisLike[FundamentalPositionBasis[int, Literal[1]]],]:
    potential = get_extended_interpolated_potential(system, shape, (resolution[0] + 2,))
    converted = convert_potential_to_basis(
        potential,
        stacked_basis_as_fundamental_position_basis(potential["basis"]),
    )
    return sample_operator(
        total_surface_hamiltonian(converted, system.mass, np.array([0])),
        sample=tuple(s * r for s, r in zip(shape, resolution, strict=True)),
    )


def get_noise_operator(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
    temperature: float,
) -> SingleBasisOperator[StackedBasisLike[FundamentalPositionBasis[int, Literal[1]]]]:
    hamiltonain = get_hamiltonian(system, shape, resolution)
    return get_caldeira_leggett_noise_operator(
        hamiltonain["basis"][0],
        system.mass,
        temperature,
        system.gamma,
    )


def get_noise_kernel(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
    temperature: float,
) -> SingleBasisDiagonalNoiseKernel[
    StackedBasisLike[FundamentalPositionBasis[Any, Literal[1]]],
]:
    hamiltonian = get_hamiltonian(system, shape, resolution)
    return get_effective_gaussian_noise_kernel(
        hamiltonian["basis"][0],
        system.eta,
        temperature,
    )


def _get_noise_operators_standard(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
    temperature: float,
) -> SingleBasisNoiseOperatorList[
    FundamentalBasis[int],
    StackedBasisLike[FundamentalPositionBasis[Any, Literal[1]]],
]:
    hamiltonian = get_hamiltonian(system, shape, resolution)
    return get_noise_operators_generic(
        hamiltonian,
        system.eta,
        temperature,
    )


def _get_noise_operators_corrected(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
    temperature: float,
) -> SingleBasisNoiseOperatorList[
    FundamentalBasis[int],
    StackedBasisLike[FundamentalPositionBasis[Any, Literal[1]]],
]:
    full_operators = _get_noise_operators_standard(
        system,
        shape,
        (resolution[0] * 2,),
        temperature,
    )
    sampled_operators = get_noise_operators_sampled(full_operators)
    hamiltonian = get_hamiltonian(system, shape, resolution)
    converted = convert_operator_list_to_basis(sampled_operators, hamiltonian["basis"])
    return {
        "basis": converted["basis"],
        "data": converted["data"],
        "eigenvalue": sampled_operators["eigenvalue"],
    }


def get_noise_operators(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
    temperature: float,
    *,
    corrected: bool = True,
) -> SingleBasisNoiseOperatorList[
    FundamentalBasis[int],
    StackedBasisLike[FundamentalPositionBasis[Any, Literal[1]]],
]:
    if corrected:
        return _get_noise_operators_corrected(system, shape, resolution, temperature)
    return _get_noise_operators_standard(system, shape, resolution, temperature)
