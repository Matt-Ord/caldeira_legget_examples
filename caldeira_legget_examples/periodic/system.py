from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Literal, Self, TypeVar

import numpy as np
from scipy.constants import Boltzmann, electron_volt, hbar  # type: ignore unknown
from slate.basis.cropped import CroppedBasis
from slate.basis.recast import RecastBasis
from slate.basis.stacked._diagonal_basis import diagonal_basis
from slate.basis.stacked._tuple_basis import tuple_basis
from slate.basis.transformed import TransformedBasis
from slate.basis.truncated import TruncatedBasis, Truncation
from slate.metadata._metadata import LabelSpacing
from slate.metadata.length import AxisDirections, SpacedLengthMetadata
from surface_potential_analysis.basis.legacy import (
    FundamentalBasis,
    FundamentalPositionBasis,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)
from surface_potential_analysis.kernel.build import (
    get_temperature_corrected_diagonal_noise_operators,
    truncate_diagonal_noise_operator_list,
)
from surface_potential_analysis.kernel.gaussian import (
    get_effective_gaussian_parameters,
    get_linear_gaussian_operators_explicit_taylor_stacked,
    get_periodic_gaussian_operators_explicit_taylor_stacked,
)
from surface_potential_analysis.kernel.kernel import (
    get_diagonal_kernel_from_diagonal_operators,
)
from surface_potential_analysis.potential.potential import (
    Potential,
)

from caldeira_legget_examples.util import (
    get_eta,
)

if TYPE_CHECKING:
    from slate.basis._basis import Basis
    from slate.metadata import VolumeMetadata
    from slate.metadata.stacked import StackedMetadata
    from surface_potential_analysis.basis.legacy import BasisLike
    from surface_potential_analysis.kernel.kernel import (
        SingleBasisDiagonalNoiseKernel,
        SingleBasisDiagonalNoiseOperatorList,
        SingleBasisNoiseOperatorList,
    )
    from surface_potential_analysis.operator.operator import Operator

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)


@dataclass(frozen=True)
class PeriodicSystem:
    """Represents the properties of a 1D Periodic System."""

    id: str
    """A unique ID, for use in caching"""
    barrier_energy: float
    lattice_constant: float
    mass: float
    gamma: float

    @property
    def classical_gamma(self: Self) -> float:  # noqa: D102
        # Note factor of 4 * hbar**2 here...
        # This is due to conventions we use when defining gamma
        return 4 * hbar**2 * self.gamma

    @property
    def eta(self: Self) -> float:  # noqa: D102
        return get_eta(self.gamma, self.mass)

    def __hash__(self: Self) -> int:
        h = hashlib.sha256(usedforsecurity=False)
        h.update(self.id.encode())
        h.update(str(self.barrier_energy).encode())
        h.update(str(self.lattice_constant).encode())
        h.update(str(self.mass).encode())
        h.update(str(self.gamma).encode())

        return int.from_bytes(h.digest(), "big")

    def with_barrier_energy(self: Self, energy: float) -> PeriodicSystem:
        """PeriodicSystem with barrier energy.

        Parameters
        ----------
        self : Self
        energy : float

        Returns
        -------
        PeriodicSystem

        """
        return PeriodicSystem(
            id=self.id,
            barrier_energy=energy,
            lattice_constant=self.lattice_constant,
            mass=self.mass,
            gamma=self.gamma,
        )

    def with_gamma(self: Self, gamma: float) -> PeriodicSystem:
        """PeriodicSystem with barrier energy.

        Parameters
        ----------
        self : Self
        energy : float

        Returns
        -------
        PeriodicSystem

        """
        return PeriodicSystem(
            id=self.id,
            barrier_energy=self.barrier_energy,
            lattice_constant=self.lattice_constant,
            mass=self.mass,
            gamma=gamma,
        )


OperatorType = Literal["periodic", "linear"]


@dataclass(frozen=True)
class PeriodicSystemConfig:
    """Configure the simlation-specific detail of the system."""

    shape: tuple[int]
    resolution: tuple[int]
    temperature: float
    operator_truncation: Iterable[int] | None = None
    operator_type: OperatorType = "periodic"

    def with_operator_type(
        self: Self,
        operator_type: OperatorType,
    ) -> PeriodicSystemConfig:
        """Config with operator_type.

        Parameters
        ----------
        self : Self
        temperature : float

        Returns
        -------
        PeriodicSystemConfig

        """
        return PeriodicSystemConfig(
            self.shape,
            self.resolution,
            self.temperature,
            self.operator_truncation,
            operator_type,
        )

    def with_temperature(self: Self, temperature: float) -> PeriodicSystemConfig:
        """Config with temperature.

        Parameters
        ----------
        self : Self
        temperature : float

        Returns
        -------
        PeriodicSystemConfig

        """
        return PeriodicSystemConfig(
            self.shape,
            self.resolution,
            temperature,
            self.operator_truncation,
            self.operator_type,
        )


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
) -> Potential[np.complex128]:
    delta_x = np.sqrt(3) * system.lattice_constant / 2  # TODO: this is a bug...
    vectors = 0.25 * system.barrier_energy * np.array([2, -1, -1]) * np.sqrt(3)

    basis_x = tuple_basis(
        (
            FundamentalBasis(
                SpacedLengthMetadata((3,), spacing=LabelSpacing(delta=delta_x)),
            ),
        ),
        AxisDirections(vectors=(np.array([1.0]),)),
    )
    return Potential(
        TransformedBasis(
            diagonal_basis(
                (basis_x, basis_x.conjugate_basis()),
                None,
            ),
        ),
        vectors.astype(np.complex128),
    )


def get_interpolated_potential(
    system: PeriodicSystem,
    resolution: tuple[_L0Inv],
) -> Potential[np.complex128]:
    potential = get_potential(system)
    basis_x = tuple_basis(
        (
            FundamentalBasis(
                SpacedLengthMetadata(
                    resolution,
                    spacing=LabelSpacing(
                        delta=potential.basis.metadata.children[0].children[0].delta,
                    ),
                ),
            ),
        ),
        AxisDirections(vectors=(np.array([1.0]),)),
    )

    basis = RecastBasis(
        diagonal_basis(
            (basis_x, basis_x.conjugate_basis()),
            None,
        ),
        basis_x,
        CroppedBasis(potential.basis.size, TransformedBasis(basis_x)),
    )

    scaled_potential = potential.raw_data * np.sqrt(
        resolution[0] / potential.basis.size,
    )
    return Potential(basis, scaled_potential)


def get_basis(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> Basis[VolumeMetadata, np.complex128]:
    return tuple_basis(
        (
            FundamentalBasis(
                SpacedLengthMetadata(
                    tuple(a * b for a, b in zip(config.resolution, config.shape)),
                    spacing=LabelSpacing(
                        delta=np.sqrt(3)
                        * config.shape[0]
                        * system.lattice_constant
                        / 2,
                    ),
                ),
            ),
        ),
        AxisDirections(vectors=(np.array([1]),)),
    )


def get_extended_interpolated_potential(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L1Inv],
) -> Potential[np.complex128]:
    potential = get_interpolated_potential(system, resolution)

    basis_x = tuple_basis(
        (
            FundamentalBasis(
                SpacedLengthMetadata(
                    tuple(a * b for a, b in zip(resolution, shape)),
                    spacing=LabelSpacing(
                        delta=potential.basis.metadata.children[0].children[0].delta
                        * shape[0],
                    ),
                ),
            ),
        ),
        AxisDirections(vectors=(np.array([1.0]),)),
    )

    basis = RecastBasis(
        diagonal_basis(
            (basis_x, basis_x.conjugate_basis()),
            None,
        ),
        basis_x,
        TruncatedBasis(
            Truncation(potential.basis.size, shape[0], 0),
            TransformedBasis(basis_x),
        ),
    )

    scaled_potential = potential.raw_data * np.sqrt(
        basis.fundamental_size / potential.basis.fundamental_size,
    )
    if shape[0] % 2 == 1:
        # Place a minima at the center
        scaled_potential *= np.exp(1j * np.pi)

    return Potential(basis, scaled_potential)


def get_hamiltonian(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> Operator[np.complex128, Basis[StackedMetadata[VolumeMetadata, None], Any]]:
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    return total_surface_hamiltonian(potential, system.mass, np.array([0]))


def _get_noise_operators_periodic(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> SingleBasisDiagonalNoiseOperatorList[
    FundamentalBasis[int],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    hamiltonian = get_hamiltonian(system, config)
    basis = hamiltonian["basis"][0]
    a, _lambda = get_effective_gaussian_parameters(
        basis,
        system.eta,
        config.temperature,
    )
    operators = get_periodic_gaussian_operators_explicit_taylor_stacked(
        basis,
        a,
        _lambda,
    )
    truncation = (
        range(operators["basis"][0].n)
        if config.operator_truncation is None
        else config.operator_truncation
    )
    return truncate_diagonal_noise_operator_list(
        operators,
        truncation,
    )


def _get_noise_operators_linear(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> SingleBasisDiagonalNoiseOperatorList[
    FundamentalBasis[int],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    hamiltonian = get_hamiltonian(system, config)
    basis = hamiltonian["basis"][0]
    a, _lambda = get_effective_gaussian_parameters(
        basis,
        system.eta,
        config.temperature,
    )

    operators = get_linear_gaussian_operators_explicit_taylor_stacked(
        basis,
        a,
        _lambda,
    )
    truncation = (
        range(operators["basis"][0].n)
        if config.operator_truncation is None
        else config.operator_truncation
    )
    return truncate_diagonal_noise_operator_list(
        operators,
        truncation,
    )


def get_noise_operators(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> SingleBasisDiagonalNoiseOperatorList[
    FundamentalBasis[int],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    match config.operator_type:
        case "periodic":
            return _get_noise_operators_periodic(system, config)
        case "linear":
            return _get_noise_operators_linear(system, config)
    msg = "Invalid operator type provided"
    raise TypeError(msg)


def get_noise_kernel(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> SingleBasisDiagonalNoiseKernel[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]]
]:
    operators = get_noise_operators(system, config)
    return get_diagonal_kernel_from_diagonal_operators(operators)


def get_temperature_corrected_noise_operators(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> SingleBasisNoiseOperatorList[
    BasisLike[Any, Any],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    operators = get_noise_operators(system, config)
    hamiltonian = get_hamiltonian(system, config)
    return get_temperature_corrected_diagonal_noise_operators(
        hamiltonian,
        operators,
        config.temperature,
    )
