from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from scipy.constants import hbar
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
from surface_potential_analysis.util.decorators import npy_cached_dict, timed

from .system import (
    PeriodicSystem,
    PeriodicSystemConfig,
    get_hamiltonian,
    get_noise_operators,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        TransformedPositionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
        TupleBasisLike,
        TupleBasisWithLengthLike,
    )
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


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)


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


def _get_stochastic_evolution_cache(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    n: int,
    step: int,
    dt_ratio: float = 500,
    n_trajectories: int = 1,
) -> Path:
    return Path(
        f"examples/data/{system.id}/stochastic.{config.shape[0]}.{config.resolution[0]}.{n}.{step}.{dt_ratio}.{n_trajectories}.{config.temperature}K.npz",
    )


@npy_cached_dict(
    _get_stochastic_evolution_cache,
    load_pickle=True,
)
@timed
def get_stochastic_evolution(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    n: int,
    step: int,
    dt_ratio: float = 500,
    n_trajectories: int = 1,
) -> StateVectorList[
    TupleBasisLike[
        FundamentalBasis[int],
        EvenlySpacedTimeBasis[int, int, int],
    ],
    TupleBasisWithLengthLike[TransformedPositionBasis[int, int, Literal[1]]],
]:
    hamiltonian = get_hamiltonian(system, config)

    initial_state = get_initial_state(hamiltonian["basis"][0])
    dt = hbar / (np.max(np.abs(hamiltonian["data"])) * dt_ratio)
    times = EvenlySpacedTimeBasis(n, step, 0, n * step * dt)

    operators = get_noise_operators(system, config)
    operator_list = list[SingleBasisOperator[Any]]()
    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    print("Collapse Operators")  # noqa: T201
    print("------------------")  # noqa: T201
    for idx in args[1:5]:
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
    # This is roughly the number of timesteps per full rotation of phase
    # should be much less than 1...
    print(times.fundamental_dt * np.max(np.abs(hamiltonian["data"])) / hbar)  # noqa: T201

    return solve_stochastic_schrodinger_equation_rust_banded(
        initial_state,
        times,
        hamiltonian,
        operator_list,
        n_trajectories=n_trajectories,
        method="Order2ExplicitWeak",
    )
