from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from scipy.constants import Boltzmann, hbar
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.dynamics.schrodinger.solve import (
    solve_schrodinger_equation,
    solve_schrodinger_equation_decomposition,
)
from surface_potential_analysis.dynamics.stochastic_schrodinger.solve import (
    solve_stochastic_schrodinger_equation,
)
from surface_potential_analysis.kernel.kernel import (
    get_single_factorized_noise_operators_diagonal,
)
from surface_potential_analysis.operator.operator import SingleBasisOperator
from surface_potential_analysis.operator.operator_list import (
    as_operator_list,
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
from surface_potential_analysis.util.decorators import npy_cached_dict

from .system import (
    get_hamiltonian,
    get_most_significant_noise_operator,
    get_potential_noise_kernel,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalPositionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisLike,
    )
    from surface_potential_analysis.state_vector.state_vector import (
        StateVector,
    )
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    _B0Inv = TypeVar(
        "_B0Inv",
        bound=StackedBasisLike[*tuple[Any, ...]],
    )


def get_initial_state(basis: _B0Inv) -> StateVector[_B0Inv]:
    data = np.zeros(basis.fundamental_n, dtype=np.complex128)
    data[basis.fundamental_n // 2] = 1
    return convert_state_vector_to_basis(
        {
            "basis": stacked_basis_as_fundamental_position_basis(basis),
            "data": data,
        },
        basis,
    )


def get_coherent_evolution() -> (
    StateVectorList[
        EvenlySpacedTimeBasis[int, int, int],
        StackedBasisLike[FundamentalPositionBasis[int, Literal[1]]],
    ]
):
    hamiltonian = get_hamiltonian(21)
    initial_state = get_initial_state(hamiltonian["basis"][0])
    times = EvenlySpacedTimeBasis(10, 200, 0, hbar * 0.5)

    return solve_schrodinger_equation(initial_state, times, hamiltonian)


def get_coherent_evolution_decomposition() -> (
    StateVectorList[
        EvenlySpacedTimeBasis[int, int, int],
        StackedBasisLike[FundamentalPositionBasis[int, Literal[1]]],
    ]
):
    hamiltonian = get_hamiltonian(21)
    initial_state = get_initial_state(hamiltonian["basis"][0])
    times = EvenlySpacedTimeBasis(100, 1, 0, hbar * 0.5)

    return solve_schrodinger_equation_decomposition(initial_state, times, hamiltonian)


# @npy_cached_dict(
#     Path("examples/data/harmonic/stochastic.256000.3.npz"),
#     load_pickle=True,
# )
def get_stochastic_evolution() -> (
    StateVectorList[
        StackedBasisLike[
            FundamentalBasis[Literal[1]],
            EvenlySpacedTimeBasis[int, int, int],
        ],
        StackedBasisLike[FundamentalPositionBasis[int, Literal[1]]],
    ]
):
    hamiltonian = get_hamiltonian(21)
    initial_state = get_initial_state(hamiltonian["basis"][0])
    times = EvenlySpacedTimeBasis(100, 10000, 0, hbar * 0.5)
    noise = get_most_significant_noise_operator(21, 10 / Boltzmann)
    noise["data"] *= hbar
    print(np.max(np.abs(noise["data"])) / hbar)
    print(np.max(np.abs(hamiltonian["data"])))

    return solve_stochastic_schrodinger_equation(
        initial_state,
        times,
        hamiltonian,
        [noise],
    )


@npy_cached_dict(
    Path("examples/data/free_particle/stochastic.5.npz"),
    load_pickle=True,
)
def get_stochastic_evolution_high_t() -> (
    StateVectorList[
        StackedBasisLike[
            FundamentalBasis[Literal[1]],
            EvenlySpacedTimeBasis[int, int, int],
        ],
        StackedBasisLike[FundamentalPositionBasis[int, Literal[1]]],
    ]
):
    hamiltonian = get_hamiltonian(21)
    initial_state = get_initial_state(hamiltonian["basis"][0])
    times = EvenlySpacedTimeBasis(1000, 20, 0, hbar * 0.05)
    kernel = get_potential_noise_kernel(21, 10 / Boltzmann)
    operators = get_single_factorized_noise_operators_diagonal(kernel)
    operator_list = list[SingleBasisOperator[Any]]()
    args = np.argsort(np.abs(operators["eigenvalue"]))
    print(operators["eigenvalue"][args])
    for idx in args:
        operator = select_operator(
            as_operator_list(operators),
            idx=idx,
        )

        operator["data"] *= (
            hbar * 10 * 10**16 * np.lib.scimath.sqrt(operators["eigenvalue"][idx])
        )
        print(np.min(np.abs(operator["data"])) ** 2 / hbar)
        print(np.max(np.abs(operator["data"])) ** 2 / hbar)
        operator_list.append(operator)
    print(np.max(np.abs(hamiltonian["data"])))

    return solve_stochastic_schrodinger_equation(
        initial_state,
        times,
        hamiltonian,
        operator_list,
    )
