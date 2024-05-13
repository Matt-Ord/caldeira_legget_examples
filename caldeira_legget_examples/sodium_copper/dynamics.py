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
    solve_stochastic_schrodinger_equation,
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

from .system import get_hamiltonian, get_noise_operator, get_noise_operators

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalPositionBasis,
    )
    from surface_potential_analysis.basis.basis_like import BasisWithLengthLike
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
        bound=StackedBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]],
    )


def get_initial_state(basis: _B0Inv) -> StateVector[_B0Inv]:
    data = np.zeros(basis.fundamental_n, dtype=np.complex128)
    util = BasisUtil(basis)
    middle = util.fundamental_n // 2
    data = np.exp(-((util.fundamental_nx_points - middle) ** 2) / 100)
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
    hamiltonian = get_hamiltonian((5,), (81,))
    initial_state = get_initial_state(hamiltonian["basis"][0])
    times = EvenlySpacedTimeBasis(200, 1000, 0, 4e-12)

    return solve_schrodinger_equation(
        initial_state,
        times,
        hamiltonian,
    )


def get_coherent_evolution_decomposition() -> (
    StateVectorList[
        EvenlySpacedTimeBasis[int, int, int],
        StackedBasisLike[FundamentalPositionBasis[int, Literal[1]]],
    ]
):
    hamiltonian = get_hamiltonian((5,), (81,))
    initial_state = get_initial_state(hamiltonian["basis"][0])
    times = EvenlySpacedTimeBasis(200, 10000, 0, 4e-12)

    return solve_schrodinger_equation_decomposition(
        initial_state,
        times,
        hamiltonian,
    )


@npy_cached_dict(
    Path("examples/data/sodium_copper/stochastic.cl.npz"),
    load_pickle=True,
)
def get_stochastic_evolution_caldeira_leggett() -> (
    StateVectorList[
        StackedBasisLike[
            FundamentalBasis[Literal[1]],
            EvenlySpacedTimeBasis[int, int, int],
        ],
        StackedBasisLike[FundamentalPositionBasis[int, Literal[1]]],
    ]
):
    hamiltonian = get_hamiltonian((5,), (81,))
    initial_state = get_initial_state(hamiltonian["basis"][0])
    times = EvenlySpacedTimeBasis(400, 1000, 0, 2e-11)
    noise = get_noise_operator(hamiltonian["basis"][0], 155)
    print(np.max(np.abs(noise["data"] / hbar)) ** 2)
    print(np.max(np.abs(hamiltonian["data"] / hbar)))

    return solve_stochastic_schrodinger_equation(
        initial_state,
        times,
        hamiltonian,
        [noise],
    )


@npy_cached_dict(
    Path("examples/data/sodium_copper/stochastic.3.31.800.16000.run.1.npz"),
    load_pickle=True,
)
@timed
def get_stochastic_evolution() -> (
    StateVectorList[
        StackedBasisLike[
            FundamentalBasis[Literal[1]],
            EvenlySpacedTimeBasis[int, int, int],
        ],
        StackedBasisLike[FundamentalPositionBasis[int, Literal[1]]],
    ]
):
    resolution = (31,)
    hamiltonian = get_hamiltonian((3,), resolution)
    initial_state = get_initial_state(hamiltonian["basis"][0])
    times = EvenlySpacedTimeBasis(800, 16000, 0, 8e-11)
    temperature = 155

    operators = get_noise_operators((3,), resolution, temperature)
    operator_list = list[SingleBasisOperator[Any]]()
    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    print(operators["basis"])
    print("Collapse Operators")
    print("------------------")
    for idx in args[1:7]:
        operator = select_operator(
            operators,
            idx=idx,
        )

        operator["data"] *= np.lib.scimath.sqrt(operators["eigenvalue"][idx] * hbar)

        print(np.max(np.abs(operator["data"])) ** 2)
        operator_list.append(operator)

    print("")
    print("Coherent Operator")
    print("------------------")
    print(np.max(np.abs(hamiltonian["data"])))
    ## This is roughly the number of timesteps per full rotation of phase
    ## should be much less than 1...
    print(times.fundamental_dt * np.max(np.abs(hamiltonian["data"])) / hbar)

    return solve_stochastic_schrodinger_equation_rust_banded(
        initial_state,
        times,
        hamiltonian,
        operator_list,
    )
