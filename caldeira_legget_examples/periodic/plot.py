from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from surface_potential_analysis.basis.basis_like import BasisLike, BasisWithLengthLike
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.kernel.plot import (
    plot_diagonal_kernel,
    plot_diagonal_kernel_truncation_error,
)
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.operator.operator import get_commutator
from surface_potential_analysis.operator.operator_list import select_operator
from surface_potential_analysis.operator.plot import (
    plot_eigenvalues,
    plot_operator_2d,
    plot_operator_diagonal_sparsity,
    plot_operator_sparsity,
)
from surface_potential_analysis.potential.plot import plot_potential_1d_x
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)
from surface_potential_analysis.state_vector.eigenvalue_list_plot import (
    plot_eigenstate_occupations,
)
from surface_potential_analysis.state_vector.plot import (
    animate_state_over_list_1d_k,
    animate_state_over_list_1d_x,
    plot_all_band_occupations,
    plot_average_band_occupation,
    plot_state_1d_k,
    plot_state_1d_x,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    StateVectorList,
    state_vector_list_into_iter,
)

from .dynamics import (
    get_coherent_evolution,
    get_coherent_evolution_decomposition,
    get_stochastic_evolution,
)
from .system import (
    PeriodicSystem,
    get_extended_interpolated_potential,
    get_hamiltonian,
    get_noise_kernel,
    get_noise_operators,
)

if TYPE_CHECKING:
    from surface_potential_analysis.state_vector.eigenstate_collection import ValueList

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)


def plot_system_potential(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
) -> None:
    """Plot the potential against position."""
    potential = get_extended_interpolated_potential(system, shape, resolution)
    fig, _, _ = plot_potential_1d_x(potential)
    fig.show()
    input()


def plot_system_eigenstates(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
) -> None:
    """Plot the potential against position."""
    potential = get_extended_interpolated_potential(system, shape, resolution)
    fig, ax, _ = plot_potential_1d_x(potential)

    hamiltonian = get_hamiltonian(system, shape, resolution)
    eigenstates = calculate_eigenvectors_hermitian(hamiltonian)

    ax1 = ax.twinx()
    fig2, ax2 = plt.subplots()
    for _i, state in enumerate(state_vector_list_into_iter(eigenstates)):
        plot_state_1d_x(state, ax=ax1)

        plot_state_1d_k(state, ax=ax2)

    fig.show()
    fig2.show()
    input()


def plot_coherent_evolution(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
) -> None:
    potential = get_extended_interpolated_potential(system, shape, resolution)
    fig, ax, _ = plot_potential_1d_x(potential)

    states = get_coherent_evolution(system)

    ax1 = ax.twinx()
    for state in state_vector_list_into_iter(states):
        plot_state_1d_x(state, ax=ax1)
    fig.show()

    fig, ax, _ = plot_potential_1d_x(potential)
    _, _, _anim0 = animate_state_over_list_1d_x(states, ax=ax.twinx())
    fig.show()

    _, _, _anim1 = animate_state_over_list_1d_k(states)
    fig.show()
    input()


def plot_coherent_evolution_decomposition(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
) -> None:
    potential = get_extended_interpolated_potential(system, shape, resolution)
    fig, ax, _ = plot_potential_1d_x(potential)

    states = get_coherent_evolution_decomposition(system)

    ax1 = ax.twinx()
    for state in state_vector_list_into_iter(states):
        plot_state_1d_x(state, ax=ax1)
    fig.show()

    fig, ax, _ = plot_potential_1d_x(potential)
    _, _, _anim0 = animate_state_over_list_1d_x(states, ax=ax.twinx())
    fig.show()

    fig, _, _anim1 = animate_state_over_list_1d_k(states)
    fig.show()

    input()


def plot_stochastic_evolution(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L1Inv],
    *,
    dt_ratio: float = 500,
    n: int = 800,
    step: int = 4000,
) -> None:
    potential = get_extended_interpolated_potential(system, shape, resolution)
    fig, ax, _ = plot_potential_1d_x(potential)

    states = get_stochastic_evolution(
        system,
        shape,
        resolution,
        n=n,
        step=step,
        dt_ratio=dt_ratio,
    )

    ax1 = ax.twinx()
    for _i, state in enumerate(state_vector_list_into_iter(states)):
        plot_state_1d_x(state, ax=ax1)
    fig.show()

    fig, ax, _ = plot_potential_1d_x(potential)
    _, _, _anim0 = animate_state_over_list_1d_x(states, ax=ax.twinx())
    fig.show()

    ax.set_title("Plot of wavefunction in Position against Time")
    line0 = Line2D([0], [0], label="Abs Wavefunction", color="tab:blue")
    ax.legend(handles=[line0])

    # writer = matplotlib.animation.FFMpegWriter(fps=20)
    # _anim0.save("position.mp4", writer=writer)

    fig, ax, _anim1 = animate_state_over_list_1d_k(states)
    fig, _, _anim2 = animate_state_over_list_1d_k(states, ax=ax, measure="real")
    fig, _, _anim3 = animate_state_over_list_1d_k(states, ax=ax, measure="imag")
    fig.show()

    seq_2 = list[list[Line2D]]()
    for frame in _anim2.frame_seq:
        line: Line2D = frame[0]
        line.set_color("tab:orange")
        line.set_linestyle("--")

        seq_2.append([line])

    _anim2.frame_seq = iter(seq_2)

    seq_3 = list[list[Line2D]]()
    for frame in _anim3.frame_seq:
        line: Line2D = frame[0]
        line.set_color("tab:green")
        line.set_linestyle("--")

        seq_3.append([line])

    _anim3.frame_seq = iter(seq_3)

    ax.set_title("Plot of Wavefunction in Momentum against Time")

    line1 = Line2D([0], [0], label="Abs Wavefunction", color="tab:blue")
    line2 = Line2D(
        [0],
        [0],
        label="Re Wavefunction",
        color="tab:orange",
        linestyle="--",
    )
    line3 = Line2D(
        [0],
        [0],
        label="Imag Wavefunction",
        color="tab:green",
        linestyle="--",
    )
    ax.legend(handles=[line1, line2, line3])

    # writer = matplotlib.animation.FFMpegWriter(fps=20)
    # _anim1.save("momentum.mp4", writer=writer, extra_anim=[_anim2, _anim3])

    input()


_B0 = TypeVar(
    "_B0",
    bound=BasisLike[Any, Any],
)


_BL0 = TypeVar("_BL0", bound=BasisWithLengthLike[Any, Any, Any])


def _get_max_occupation_x(
    states: StateVectorList[
        _B0,
        StackedBasisLike[_BL0],
    ],
) -> ValueList[_B0]:
    states_x = convert_state_vector_list_to_basis(
        states,
        stacked_basis_as_fundamental_position_basis(states["basis"][1]),
    )

    return {
        "basis": states_x["basis"][0],
        "data": np.argmax(
            np.abs(states_x["data"].reshape(states_x["basis"].shape)),
            axis=1,
        ),
    }


def _get_max_occupation_k(
    states: StateVectorList[
        _B0,
        StackedBasisLike[_BL0],
    ],
) -> ValueList[_B0]:
    states_x = convert_state_vector_list_to_basis(
        states,
        stacked_basis_as_fundamental_momentum_basis(states["basis"][1]),
    )

    return {
        "basis": states_x["basis"][0],
        "data": np.argmax(
            np.fft.fftshift(
                np.abs(states_x["data"].reshape(states_x["basis"].shape)),
                axes=1,
            ),
            axis=1,
        ),
    }


def plot_point_evolution(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L1Inv],
    *,
    dt_ratio: float = 500,
) -> None:
    states = get_stochastic_evolution(
        system,
        shape,
        resolution,
        n=800,
        step=4000,
        dt_ratio=dt_ratio,
    )

    fig, ax = plt.subplots()
    x_points = _get_max_occupation_x(states)["data"]
    ax.plot(x_points)

    ax.plot(np.unwrap(x_points, period=states["basis"][1].fundamental_n))

    ax.plot(_get_max_occupation_k(states)["data"])
    fig.show()

    input()


def plot_stochastic_occupation(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
    *,
    dt_ratio: float = 500,
    n: int = 800,
    step: int = 4000,
) -> None:
    states = get_stochastic_evolution(
        system,
        shape,
        resolution,
        n=n,
        step=step,
        dt_ratio=dt_ratio,
    )
    hamiltonian = get_hamiltonian(system, shape, resolution)

    fig0, ax0 = plot_all_band_occupations(hamiltonian, states)

    fig1, ax1 = fig0, ax0
    # fig1, ax1, _ani = animate_all_band_occupations(hamiltonian, states)

    fig2, ax2, line = plot_average_band_occupation(hamiltonian, states)

    eigenstates = calculate_eigenvectors_hermitian(hamiltonian)

    for ax in [ax0, ax1, ax2]:
        _, _, line = plot_eigenstate_occupations(eigenstates, 150, ax=ax)
        line.set_linestyle("--")
        line.set_label("Expected")

        ax.legend([line], ["Expected occupation"])

    fig0.show()
    fig1.show()
    fig2.show()
    input()


def plot_thermal_occupation(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
) -> None:
    hamiltonian = get_hamiltonian(system, shape, resolution)

    eigenstates = calculate_eigenvectors_hermitian(hamiltonian)

    fig, _, _ = plot_eigenstate_occupations(eigenstates, 150)

    fig.show()
    input()


def plot_kernel(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
) -> None:
    kernel = get_noise_kernel(system, shape, resolution, 150)
    fig, _, _ = plot_diagonal_kernel(kernel)
    fig.show()

    fig, _ = plot_diagonal_kernel_truncation_error(kernel)
    fig.show()

    input()


def plot_hamiltonian_sparsity(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
) -> None:
    hamiltonian = get_hamiltonian(system, shape, resolution)
    fig, _ = plot_operator_sparsity(hamiltonian)
    fig.show()
    input()


def plot_largest_collapse_operator(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
    *,
    temperature: float = 155,
) -> None:
    operators = get_noise_operators(system, shape, resolution, temperature)

    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    x_basis = stacked_basis_as_fundamental_position_basis(
        operators["basis"][1][0],
    )
    k_basis = stacked_basis_as_fundamental_momentum_basis(
        operators["basis"][1][0],
    )

    for idx in args[2:8]:
        operator = select_operator(operators, idx=idx)

        operator_x = convert_operator_to_basis(
            operator,
            StackedBasis(x_basis, x_basis),
        )

        fig, ax, _ = plot_operator_2d(operator_x)
        ax.set_title("Operator in X")
        fig.show()

        operator_k = convert_operator_to_basis(
            operator,
            StackedBasis(k_basis, k_basis),
        )

        fig, ax, _ = plot_operator_2d(operator_k)
        ax.set_title("Operator in K")
        fig.show()

    operator = get_hamiltonian(system, shape, resolution)
    operator_k = convert_operator_to_basis(
        operator,
        StackedBasis(k_basis, k_basis),
    )

    fig, ax, _ = plot_operator_2d(operator_k)
    ax.set_title("Hamiltonian in K")
    fig.show()

    input()


def plot_largest_collapse_operator_eigenvalues(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
    *,
    temperature: float = 155,
) -> None:
    operators = get_noise_operators(
        system,
        shape,
        resolution,
        temperature,
        corrected=False,
    )

    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    for idx in args[2:8]:
        operator = select_operator(operators, idx=idx)

        fig, _ax, _ = plot_eigenvalues(operator)
        fig.show()

    input()


def plot_largest_collapse_operator_commutator(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
    *,
    temperature: float = 150,
) -> None:
    operators = get_noise_operators(
        system,
        shape,
        resolution,
        temperature,
        corrected=False,
    )

    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1].tolist()

    for i, i_idx in list(enumerate(args))[2:8]:
        for j, j_idx in list(enumerate(args))[2:8]:
            operator_i = select_operator(operators, idx=i_idx)
            operator_j = select_operator(operators, idx=j_idx)

            commutator = get_commutator(operator_i, operator_j)

            fig, ax, _ = plot_eigenvalues(commutator)
            ax.set_title(f"commutator {i}, {j}")
            fig.show()

    input()


def plot_collapse_operator_2d(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
    *,
    temperature: float = 155,
) -> None:
    operators = get_noise_operators(
        system,
        shape,
        resolution,
        temperature,
    )

    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    k_basis = stacked_basis_as_fundamental_momentum_basis(
        operators["basis"][1][0],
    )

    for idx in args[5:8]:
        operator = select_operator(operators, idx=idx)
        print(operators["eigenvalue"][idx])
        operator_k = convert_operator_to_basis(
            operator,
            StackedBasis(k_basis, k_basis),
        )

        fig, ax = plot_operator_sparsity(operator_k)
        ax.set_title("Operator sparsity in K")
        fig.show()

        fig, ax = plot_operator_diagonal_sparsity(operator_k)
        ax.set_title("Operator sparsity in K")
        fig.show()

    operator = get_hamiltonian(system, shape, resolution)
    operator_k = convert_operator_to_basis(
        operator,
        StackedBasis(k_basis, k_basis),
    )

    fig, ax, _ = plot_operator_2d(operator_k)
    ax.set_title("Hamiltonian in K")
    fig.show()

    input()
