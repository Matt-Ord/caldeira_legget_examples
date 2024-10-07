from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Self, TypeVar

import numpy as np
import sdeint  # type: ignore unknown
from scipy.constants import Boltzmann, hbar  # type: ignore unknown
from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
)
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.dynamics.schrodinger.solve import (
    solve_schrodinger_equation,
    solve_schrodinger_equation_decomposition,
)
from surface_potential_analysis.dynamics.stochastic_schrodinger.solve import (
    solve_stochastic_schrodinger_equation_rust_banded,
)
from surface_potential_analysis.operator.operator import SingleBasisOperator
from surface_potential_analysis.operator.operator_list import (
    select_operator,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_basis,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    StateVectorList,
)
from surface_potential_analysis.util.decorators import cached

from .system import (
    PeriodicSystem,
    PeriodicSystemConfig,
    get_hamiltonian,
    get_potential,
    get_potential_derivative,
    get_temperature_corrected_noise_operators,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        TransformedPositionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
        TupleBasisLike,
        TupleBasisWithLengthLike,
    )
    from surface_potential_analysis.state_vector.eigenstate_list import ValueList
    from surface_potential_analysis.state_vector.state_vector import (
        StateVector,
    )
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    _B0Inv = TypeVar(
        "_B0Inv",
        bound=StackedBasisWithVolumeLike[Any, Any, Any],
    )


def get_initial_state(basis: _B0Inv) -> StateVector[_B0Inv]:
    data = np.zeros(basis.fundamental_n, dtype=np.complex128)
    util = BasisUtil(basis)
    middle = util.fundamental_n // 2
    data = np.exp(-((util.fundamental_nx_points - middle) ** 2) / 50)
    data /= np.sqrt(np.sum(np.abs(data) ** 2))
    return convert_state_vector_to_basis(
        {
            "basis": stacked_basis_as_fundamental_position_basis(basis),
            "data": data,
        },
        basis,
    )


def get_coherent_evolution(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> StateVectorList[
    EvenlySpacedTimeBasis[int, int, int],
    TupleBasisWithLengthLike[TransformedPositionBasis[int, int, Literal[1]]],
]:
    hamiltonian = get_hamiltonian(system, config)
    initial_state = get_initial_state(hamiltonian["basis"][0])
    times = EvenlySpacedTimeBasis(200, 1000, 0, 4e-12)

    return solve_schrodinger_equation(
        initial_state,
        times,
        hamiltonian,
    )


def get_coherent_evolution_decomposition(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> StateVectorList[
    EvenlySpacedTimeBasis[int, int, int],
    TupleBasisWithLengthLike[TransformedPositionBasis[int, int, Literal[1]]],
]:
    hamiltonian = get_hamiltonian(system, config)
    initial_state = get_initial_state(hamiltonian["basis"][0])
    times = EvenlySpacedTimeBasis(200, 10000, 0, 4e-12)

    return solve_schrodinger_equation_decomposition(
        initial_state,
        times,
        hamiltonian,
    )


@dataclass
class PeriodicSimulationConfig:
    """Simulation specific config."""

    n: int = field(kw_only=True)
    step: int = field(kw_only=True)
    dt_ratio: float = field(default=500, kw_only=True)
    n_trajectories: int = field(default=1, kw_only=True)
    n_realizations: int = field(default=1, kw_only=True)

    def __hash__(self: Self) -> int:
        """Generate a hash."""
        return hash((
            self.n,
            self.step,
            self.dt_ratio,
            self.n_realizations,
            self.n_trajectories,
        ))


def _get_simulation_times(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    simulation_config: PeriodicSimulationConfig,
) -> EvenlySpacedTimeBasis[int, int, int]:
    hamiltonian = get_hamiltonian(system, config)
    dt = hbar / (np.max(np.abs(hamiltonian["data"])) * simulation_config.dt_ratio)
    return EvenlySpacedTimeBasis(
        simulation_config.n,
        simulation_config.step,
        0,
        simulation_config.n * simulation_config.step * dt,
    )


def _get_stochastic_evolution_cache(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    simulation_config: PeriodicSimulationConfig,
) -> Path:
    return Path(
        f"data/stochastic.{system.id}.{hash(system)}.{config.shape[0]}.{config.resolution[0]}.{config.operator_type}.{hash(simulation_config)}.{config.temperature}K",
    )


@cached(_get_stochastic_evolution_cache)
def get_stochastic_evolution(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    simulation_config: PeriodicSimulationConfig,
) -> StateVectorList[
    TupleBasisLike[
        FundamentalBasis[int],
        EvenlySpacedTimeBasis[int, int, int],
    ],
    TupleBasisWithLengthLike[TransformedPositionBasis[int, int, Literal[1]]],
]:
    hamiltonian = get_hamiltonian(system, config)

    initial_state = get_initial_state(hamiltonian["basis"][0])

    times = _get_simulation_times(
        system,
        config,
        simulation_config,
    )

    operators = get_temperature_corrected_noise_operators(system, config)
    operator_list = list[SingleBasisOperator[Any]]()
    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    print("Collapse Operators")  # noqa: T201
    print("------------------")  # noqa: T201
    for idx in args:
        operator = select_operator(
            operators,
            idx=idx,
        )
        operator["data"] *= np.lib.scimath.sqrt(operators["eigenvalue"][idx] * hbar)

        print(f"{np.max(np.abs(operator['data'])) ** 2:e}")  # noqa: T201
        operator_list.append(operator)

    print()  # noqa: T201
    print("Coherent Operator")  # noqa: T201
    print("------------------")  # noqa: T201
    print(f"{np.max(np.abs(hamiltonian['data'])):e}")  # noqa: T201

    print(f"dt = {times.fundamental_dt}")  # noqa: T201
    # This is roughly the number of timesteps per full rotation of phase
    # should be much less than 1...
    print(  # noqa: T201
        f"ratio = {times.fundamental_dt * np.max(np.abs(hamiltonian['data'])) / hbar}",
    )

    return solve_stochastic_schrodinger_equation_rust_banded(
        initial_state,
        times,
        hamiltonian,
        operator_list,
        n_trajectories=simulation_config.n_trajectories,
        n_realizations=simulation_config.n_realizations,
        method="Order2ExplicitWeak",
    )


def _get_potential_derivative_function(
    system: PeriodicSystem,
) -> Callable[[float], float]:
    derivative = get_potential_derivative(get_potential(system))

    k_points = BasisUtil(derivative["basis"]).k_points[0]

    def _fn(x: float) -> float:
        phases = 1j * k_points * x
        return np.einsum(  # type:ignore unknown
            "i,i->",
            derivative["data"],
            np.exp(phases) / np.sqrt(derivative["basis"].n),
        )

    return _fn


def get_langevin_evolution(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    simulation_config: PeriodicSimulationConfig,
) -> ValueList[
    TupleBasisLike[
        FundamentalBasis[int],
        EvenlySpacedTimeBasis[int, int, int],
    ]
]:
    times = _get_simulation_times(
        system,
        config,
        simulation_config,
    )

    _force = _get_potential_derivative_function(system)

    def _drift(
        state: np.ndarray[Any, np.dtype[np.float64]],
        _t: float,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        x, v = state
        dx = v
        dv = (-system.gamma * v - _force(x)) / system.mass
        return np.array([dx, dv])

    diffusion = np.sqrt(
        2 * system.gamma * hbar**2 * Boltzmann * config.temperature / system.mass,
    )

    def _diffusion(
        _state: np.ndarray[Any, np.dtype[np.float64]],
        _t: float,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        # diffusion acts on velocity
        return np.array([0, diffusion])

    with ThreadPoolExecutor() as executor:
        data = np.array(
            list(
                executor.map(
                    lambda _: sdeint.itoint(_drift, _diffusion, 0, times.times),  # type: ignore unknown
                    range(simulation_config.n_trajectories),
                ),
            ),
        )

    return {
        "basis": TupleBasis(FundamentalBasis(simulation_config.n_trajectories), times),
        "data": data.astype(np.complex128),
    }
