from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Literal, Self, TypeVar

import numpy as np
from scipy.constants import Boltzmann, electron_volt, hbar  # type: ignore unknown
from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalTransformedPositionBasis,
    FundamentalTransformedPositionBasis1d,
    TransformedPositionBasis,
    TransformedPositionBasis1d,
)
from surface_potential_analysis.basis.evenly_spaced_basis import (
    EvenlySpacedTransformedPositionBasis,
)
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)
from surface_potential_analysis.kernel.gaussian import (
    get_temperature_corrected_effective_gaussian_noise_operators,
)
from surface_potential_analysis.potential.conversion import convert_potential_to_basis
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
)

from caldeira_legget_examples.util import (
    get_caldeira_leggett_noise_operator,
    get_eta,
)

if TYPE_CHECKING:
    from surface_potential_analysis.kernel.kernel import (
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
    def eta(self: Self) -> float:  # noqa: D102
        return get_eta(self.gamma, self.mass)

    def __hash__(self: Self) -> int:
        h = hashlib.sha256(usedforsecurity=False)
        h.update(self.id.encode())
        h.update(str(self.barrier_energy).encode())
        h.update(str(self.lattice_constant).encode())
        h.update(str(self.mass).encode())

        return int.from_bytes(h.digest(), "big")


@dataclass
class PeriodicSystemConfig:
    """Configure the simlation-specific detail of the system."""

    shape: tuple[int]
    resolution: tuple[int]
    temperature: float
    operator_truncation: Iterable[int] | None = None


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

FREE_LITHIUM_SYSTEM = PeriodicSystem(
    id="LiFree",
    barrier_energy=0,
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

# A free particle, designed to be ran at T=get_dimensionless_temperature
FREE_SYSTEM = PeriodicSystem(
    id="Free",
    barrier_energy=0,
    lattice_constant=1,
    mass=(hbar) ** 2,
    gamma=1 / hbar,
)


def get_dimensionless_temperature(system: PeriodicSystem) -> float:
    return (
        hbar**2 * (2 * np.pi / system.lattice_constant) ** 2 / (system.mass * Boltzmann)
    )


def get_potential(
    system: PeriodicSystem,
) -> Potential[TupleBasis[FundamentalTransformedPositionBasis1d[Literal[3]]]]:
    delta_x = np.sqrt(3) * system.lattice_constant / 2  # TODO: this is a bug...
    axis = FundamentalTransformedPositionBasis1d[Literal[3]](np.array([delta_x]), 3)
    vector = 0.25 * system.barrier_energy * np.array([2, -1, -1]) * np.sqrt(3)
    return {"basis": TupleBasis(axis), "data": vector}


def get_interpolated_potential(
    system: PeriodicSystem,
    resolution: tuple[_L0Inv],
) -> Potential[
    TupleBasisWithLengthLike[FundamentalTransformedPositionBasis[_L0Inv, Literal[1]]]
]:
    potential = get_potential(system)
    old = potential["basis"][0]
    basis = TupleBasis(
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


def get_basis(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> TupleBasisWithLengthLike[TransformedPositionBasis[int, int, Literal[1]]]:
    return TupleBasis[TransformedPositionBasis[int, int, Literal[1]]](
        *tuple(
            FundamentalTransformedPositionBasis[int, Literal[1]](
                np.array([np.sqrt(3) * s * system.lattice_constant / 2]),
                s * r,
            )
            for (s, r) in zip(config.shape, config.resolution)
        ),
    )


def get_extended_interpolated_potential(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L1Inv],
) -> Potential[
    TupleBasisWithLengthLike[
        EvenlySpacedTransformedPositionBasis[_L1Inv, _L0Inv, Literal[0], Literal[1]]
    ]
]:
    interpolated = get_interpolated_potential(system, resolution)
    old = interpolated["basis"][0]
    basis = TupleBasis(
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
    config: PeriodicSystemConfig,
) -> SingleBasisOperator[
    TupleBasisWithLengthLike[TransformedPositionBasis[int, int, Literal[1]]],
]:
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    basis = get_basis(system, config)
    converted = convert_potential_to_basis(potential, basis)
    return total_surface_hamiltonian(converted, system.mass, np.array([0]))


def _get_noise_operators_standard(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> SingleBasisNoiseOperatorList[
    FundamentalBasis[int],
    TupleBasisWithLengthLike[TransformedPositionBasis[int, int, Literal[1]]],
]:
    hamiltonian = get_hamiltonian(system, config)
    return get_temperature_corrected_effective_gaussian_noise_operators(
        hamiltonian,
        system.eta,
        config.temperature,
        truncation=config.operator_truncation,
    )


def _get_noise_operators_linear(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> SingleBasisNoiseOperatorList[
    FundamentalBasis[int],
    TupleBasisWithLengthLike[TransformedPositionBasis[int, int, Literal[1]]],
]:
    hamiltonain = get_hamiltonian(system, config)
    operator = get_caldeira_leggett_noise_operator(
        hamiltonain["basis"][0],
        system.mass,
        config.temperature,
        system.gamma,
    )
    return {
        "basis": TupleBasis(FundamentalBasis(1), operator["basis"]),
        "data": operator["data"],
        "eigenvalue": np.array([1], dtype=np.complex128),
    }


def get_noise_operators(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    ty: Literal["standard", "linear"] = "standard",
) -> SingleBasisNoiseOperatorList[
    FundamentalBasis[int],
    TupleBasisWithLengthLike[TransformedPositionBasis[Any, Any, Literal[1]]],
]:
    match ty:
        case "standard":
            return _get_noise_operators_standard(system, config)
        case "linear":
            return _get_noise_operators_linear(system, config)
    msg = "Invalid ty provided"
    raise TypeError(msg)
