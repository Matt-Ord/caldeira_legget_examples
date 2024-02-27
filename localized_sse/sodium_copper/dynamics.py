from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.dynamics.schrodinger.solve import (
    solve_schrodinger_equation,
    solve_schrodinger_equation_decomposition,
)
from surface_potential_analysis.dynamics.stochastic_schrodinger.solve import (
    solve_stochastic_schrodinger_equation,
    solve_stochastic_schrodinger_equation_localized,
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

from localized_sse.sodium_copper_2d.system import (
    get_hamiltonian,
    get_noise_operator,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalTransformedPositionBasis,
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
        StackedBasisLike[
            FundamentalBasis[Literal[1]],
            EvenlySpacedTimeBasis[int, int, int],
        ],
        StackedBasisLike[FundamentalTransformedPositionBasis[int, Literal[1]]],
    ]
):
    hamiltonian = get_hamiltonian((6,), (100,))
    initial_state = get_initial_state(hamiltonian["basis"][0])
    times = EvenlySpacedTimeBasis(20, 10000, 0, 0.4e-12)

    return solve_schrodinger_equation(
        initial_state,
        times,
        hamiltonian,
    )


def get_coherent_evolution_decomposition() -> (
    StateVectorList[
        StackedBasisLike[
            FundamentalBasis[Literal[1]],
            EvenlySpacedTimeBasis[int, int, int],
        ],
        StackedBasisLike[FundamentalTransformedPositionBasis[int, Literal[1]]],
    ]
):
    hamiltonian = get_hamiltonian((6,), (100,))
    initial_state = get_initial_state(hamiltonian["basis"][0])
    times = EvenlySpacedTimeBasis(20, 10000, 0, 0.4e-12)

    return solve_schrodinger_equation_decomposition(
        initial_state,
        times,
        hamiltonian,
    )


@npy_cached_dict(Path("examples/data/stochastic.npz"), load_pickle=True)
def get_stochastic_evolution() -> (
    StateVectorList[
        StackedBasisLike[
            FundamentalBasis[Literal[1]],
            EvenlySpacedTimeBasis[int, int, int],
        ],
        StackedBasisLike[FundamentalTransformedPositionBasis[int, Literal[1]]],
    ]
):
    hamiltonian = get_hamiltonian((6,), (100,))
    initial_state = get_initial_state(hamiltonian["basis"][0])
    times = EvenlySpacedTimeBasis(20, 10000, 0, 0.4e-12)
    noise = get_noise_operator(hamiltonian["basis"][0], 155)

    return solve_stochastic_schrodinger_equation(
        initial_state,
        times,
        hamiltonian,
        [noise],
    )


@npy_cached_dict(Path("examples/data/stochastic_localized.npz"), load_pickle=True)
def get_stochastic_evolution_localized() -> (
    StateVectorList[
        StackedBasisLike[
            FundamentalBasis[Literal[1]],
            EvenlySpacedTimeBasis[int, int, int],
        ],
        StackedBasisLike[FundamentalTransformedPositionBasis[int, Literal[1]]],
    ]
):
    hamiltonian = get_hamiltonian((6,), (100,))
    initial_state = get_initial_state(hamiltonian["basis"][0])
    times = EvenlySpacedTimeBasis(20, 10000, 0, 2e-12)
    noise = get_noise_operator(hamiltonian["basis"][0], 155)

    return solve_stochastic_schrodinger_equation_localized(
        initial_state,
        times,
        hamiltonian,
        [noise],
    )