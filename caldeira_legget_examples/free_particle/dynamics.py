from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from scipy.constants import Boltzmann, hbar
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.dynamics.schrodinger.solve import (
    solve_schrodinger_equation,
    solve_schrodinger_equation_decomposition,
)
from surface_potential_analysis.dynamics.stochastic_schrodinger.solve import (
    solve_stochastic_schrodinger_equation_rust,
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
from surface_potential_analysis.util.decorators import npy_cached_dict

from .system import (
    get_hamiltonian,
    get_non_periodic_noise_operators,
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
    util = BasisUtil(basis)
    middle = util.fundamental_n // 2
    width = util.fundamental_n // 10
    data = np.exp(-((util.fundamental_nx_points - middle) ** 2) / width)
    data /= np.sqrt(np.sum(np.abs(data) ** 2))

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


# n_states.n_time.n_step.operator_prefactor
@npy_cached_dict(
    Path("examples/data/free_particle/stochastic.41.8000.1000.5.npz"),
    load_pickle=True,
)
def get_stochastic_evolution() -> (
    StateVectorList[
        StackedBasisLike[
            FundamentalBasis[Literal[1]],
            EvenlySpacedTimeBasis[int, int, int],
        ],
        StackedBasisLike[FundamentalPositionBasis[int, Literal[1]]],
    ]
):
    times = EvenlySpacedTimeBasis(8000, 1000, 0, 80 * hbar)
    temperature = 3 / Boltzmann
    n_states = 41

    hamiltonian = get_hamiltonian(n_states)
    initial_state = get_initial_state(hamiltonian["basis"][0])

    operators = get_non_periodic_noise_operators(n_states, temperature)
    operator_list = list[SingleBasisOperator[Any]]()
    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    print("Collapse Operators")
    print("------------------")
    for idx in args[1:13]:
        operator = select_operator(
            operators,
            idx=idx,
        )

        operator["data"] *= 5 * np.lib.scimath.sqrt(operators["eigenvalue"][idx])

        print(np.max(np.abs(operator["data"])) ** 2)
        operator_list.append(operator)

    print("")
    print("Coherent Operator")
    print("------------------")
    print(np.max(np.abs(hamiltonian["data"])))

    return solve_stochastic_schrodinger_equation_rust(
        initial_state,
        times,
        hamiltonian,
        operator_list,
    )
