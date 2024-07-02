from __future__ import annotations

from typing import TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
)
from surface_potential_analysis.kernel.conversion import (
    convert_noise_operator_list_to_basis,
)
from surface_potential_analysis.kernel.kernel import (
    as_diagonal_kernel,
    get_noise_kernel,
)
from surface_potential_analysis.kernel.plot import (
    plot_diagonal_kernel,
    plot_diagonal_kernel_truncation_error,
    plot_diagonal_noise_operators_eigenvalues,
    plot_noise_operators_single_sample_x,
)
from surface_potential_analysis.operator.conversion import (
    convert_operator_to_basis,
)
from surface_potential_analysis.operator.operator_list import (
    select_operator,
)
from surface_potential_analysis.operator.plot import (
    plot_eigenstate_occupations,
    plot_eigenvalues,
    plot_operator_2d,
    plot_operator_along_diagonal,
    plot_operator_diagonal_sparsity,
    plot_operator_sparsity,
)
from surface_potential_analysis.potential.plot import plot_potential_1d_x
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)
from surface_potential_analysis.state_vector.plot import (
    animate_state_over_list_1d_k,
    animate_state_over_list_1d_x,
    plot_average_band_occupation,
    plot_average_displacement_1d_x,
    plot_periodic_averaged_occupation_1d_x,
    plot_state_1d_k,
    plot_state_1d_x,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    get_state_along_axis,
    state_vector_list_into_iter,
)

from .dynamics import (
    get_coherent_evolution,
    get_coherent_evolution_decomposition,
    get_stochastic_evolution,
)
from .system import (
    PeriodicSystem,
    PeriodicSystemConfig,
    get_extended_interpolated_potential,
    get_hamiltonian,
    get_noise_operators,
)

_L0Inv = TypeVar("_L0Inv", bound=int)


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
    config: PeriodicSystemConfig,
) -> None:
    """Plot the potential against position."""
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    fig, ax, _ = plot_potential_1d_x(potential)

    hamiltonian = get_hamiltonian(system, config)
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
    config: PeriodicSystemConfig,
) -> None:
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    fig, ax, _ = plot_potential_1d_x(potential)

    states = get_coherent_evolution(system, config)

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
    config: PeriodicSystemConfig,
) -> None:
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    fig, ax, _ = plot_potential_1d_x(potential)

    states = get_coherent_evolution_decomposition(system, config)

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
    config: PeriodicSystemConfig,
    *,
    dt_ratio: float = 500,
    n: int = 800,
    step: int = 4000,
    n_trajectories: int = 1,
) -> None:
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    fig, ax, _ = plot_potential_1d_x(potential)

    all_states = get_stochastic_evolution(
        system,
        config,
        n=n,
        step=step,
        dt_ratio=dt_ratio,
        n_trajectories=n_trajectories,
    )

    states = get_state_along_axis(all_states, axes=(1,), idx=(0,))

    ax1 = ax.twinx()
    fig2, ax2 = plt.subplots()
    for _i, state in enumerate(state_vector_list_into_iter(states)):
        if _i < 500:
            plot_state_1d_x(state, ax=ax1)
            plot_state_1d_k(state, ax=ax2)
    fig.show()
    fig2.show()

    input()

    fig, ax, _ = plot_potential_1d_x(potential)
    _, _, _anim0 = animate_state_over_list_1d_x(states, ax=ax.twinx())
    fig.show()

    ax.set_title("Plot of wavefunction in Position against Time")
    line0 = Line2D([0], [0], label="Abs Wavefunction", color="tab:blue")
    ax.legend(handles=[line0])

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

    input()


def plot_point_evolution(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    dt_ratio: float = 500,
    n: int = 800,
    step: int = 4000,
    n_trajectories: int = 1,
) -> None:
    states = get_stochastic_evolution(
        system,
        config,
        n=n,
        step=step,
        dt_ratio=dt_ratio,
        n_trajectories=n_trajectories,
    )

    fig, ax = plot_periodic_averaged_occupation_1d_x(states)

    y_start = 3 * states["basis"][1][0].delta_x[0]
    ax.vlines(
        states["basis"][0][1].times[100],
        ymin=y_start,
        ymax=y_start + (0.33 * states["basis"][1][0].delta_x),
        colors="black",
    )
    ax.text(
        states["basis"][0][1].times[800],
        y_start,
        "Unit Cell Width",
    )
    fig.show()

    fig, ax, _line = plot_average_displacement_1d_x(states)
    fig.show()
    input()


def plot_stochastic_occupation(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    dt_ratio: float = 500,
    n: int = 800,
    step: int = 4000,
    n_trajectories: int = 1,
) -> None:
    states = get_stochastic_evolution(
        system,
        config,
        n=n,
        step=step,
        dt_ratio=dt_ratio,
        n_trajectories=n_trajectories,
    )
    hamiltonian = get_hamiltonian(system, config)

    # fig0, ax0 = plot_all_band_occupations(hamiltonian, states)

    # fig1, ax1 = fig0, ax0
    # fig1, ax1, _ani = animate_all_band_occupations(hamiltonian, states)

    fig2, ax2, line = plot_average_band_occupation(hamiltonian, states)

    for ax in [ax2]:
        _, _, line = plot_eigenstate_occupations(hamiltonian, config.temperature, ax=ax)
        line.set_linestyle("--")
        line.set_label("Expected")

        ax.legend([line], ["Boltzmann occupation"])

    # fig0.show()
    # fig1.show()
    fig2.show()
    input()


def plot_thermal_occupation(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    hamiltonian = get_hamiltonian(system, config)

    fig, _, _ = plot_eigenstate_occupations(hamiltonian, config.temperature)

    fig.show()
    input()


def plot_kernel(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    operators = get_noise_operators(system, config)
    basis_x = stacked_basis_as_fundamental_position_basis(operators["basis"][1][0])
    converted = convert_noise_operator_list_to_basis(
        operators,
        TupleBasis(basis_x, basis_x),
    )
    kernel = as_diagonal_kernel(get_noise_kernel(converted))
    fig, _, _ = plot_diagonal_kernel(kernel)
    fig.show()

    fig, _ = plot_diagonal_kernel_truncation_error(kernel)
    fig.show()

    input()


def plot_hamiltonian_sparsity(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    hamiltonian = get_hamiltonian(system, config)
    fig, _ = plot_operator_sparsity(hamiltonian)
    fig.show()
    input()


def plot_largest_collapse_operator(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    operators = get_noise_operators(system, config)

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
            TupleBasis(x_basis, x_basis),
        )

        fig, ax, _ = plot_operator_2d(operator_x)
        ax.set_title("Operator in X")
        fig.show()

        operator_k = convert_operator_to_basis(
            operator,
            TupleBasis(k_basis, k_basis),
        )

        fig, ax, _ = plot_operator_2d(operator_k)
        ax.set_title("Operator in K")
        fig.show()

    operator = get_hamiltonian(system, config)
    operator_k = convert_operator_to_basis(
        operator,
        TupleBasis(k_basis, k_basis),
    )

    fig, ax, _ = plot_operator_2d(operator_k)
    ax.set_title("Hamiltonian in K")
    fig.show()

    input()


def plot_largest_collapse_operator_eigenvalues(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    operators = get_noise_operators(
        system,
        config,
    )

    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    for idx in args[2:8]:
        operator = select_operator(operators, idx=idx)

        fig, _ax, _ = plot_eigenvalues(operator)
        fig.show()

    input()


def plot_kernel_eigenvalues(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    operators = get_noise_operators(
        system,
        config,
        ty="standard",
    )
    fig, ax, _ = plot_diagonal_noise_operators_eigenvalues(operators)

    operators = get_noise_operators(
        system,
        config,
        ty="corrected",
    )
    fig, _, _ = plot_diagonal_noise_operators_eigenvalues(operators, ax=ax)

    fig.show()

    input()


def plot_collapse_operator_1d(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    operators = get_noise_operators(
        system,
        config,
        ty="standard",
    )
    operators_corrected = get_noise_operators(
        system,
        config,
        ty="corrected",
    )
    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    x_basis = stacked_basis_as_fundamental_position_basis(
        operators["basis"][1][0],
    )

    for idx in args[7:8]:
        operator = select_operator(operators, idx=idx)
        operator_corrected = select_operator(operators_corrected, idx=idx)
        operator_x = convert_operator_to_basis(
            operator,
            TupleBasis(x_basis, x_basis),
        )
        operator_corrected_x = convert_operator_to_basis(
            operator_corrected,
            TupleBasis(x_basis, x_basis),
        )

        fig, ax, line = plot_operator_along_diagonal(operator_x, measure="abs")
        line.set_label("standard operator")
        fig.show()

        fig, _, line = plot_operator_along_diagonal(operator_corrected_x, measure="abs")
        line.set_label("corrected operator")

        ax.legend()
        fig.show()

    input()


def plot_collapse_operator_2d(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    operators = get_noise_operators(
        system,
        config,
        ty="standard",
    )
    operators_corrected = get_noise_operators(
        system,
        config,
        ty="corrected",
    )
    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    k_basis = stacked_basis_as_fundamental_momentum_basis(
        operators["basis"][1][0],
    )

    for idx in args[:4]:
        operator = select_operator(operators, idx=idx)
        operator_corrected = select_operator(operators_corrected, idx=idx)
        operator_k = convert_operator_to_basis(
            operator,
            TupleBasis(k_basis, k_basis),
        )
        operator_corrected_k = convert_operator_to_basis(
            operator_corrected,
            TupleBasis(k_basis, k_basis),
        )

        fig, ax = plot_operator_sparsity(operator_k)
        _, _ = plot_operator_sparsity(operator_corrected_k, ax=ax)
        ax.set_title("Operator sparsity in K")
        fig.show()

        fig, ax = plot_operator_diagonal_sparsity(operator_k)
        _, _ = plot_operator_diagonal_sparsity(operator_corrected_k, ax=ax)
        ax.set_title("Operator diagonal sparsity in K")
        fig.show()

        fig, ax, _ = plot_operator_2d(operator_k, measure="abs")
        ax.set_title(f"standard operator, idx={idx}")
        fig.show()

        fig, ax, _ = plot_operator_2d(operator, measure="abs")
        ax.set_title(f"standard operator original basis, idx={idx}")
        fig.show()

        fig, ax, _ = plot_operator_2d(operator_corrected_k, measure="abs")
        ax.set_title(f"corrected operator, idx={idx}")
        fig.show()

    input()


def plot_effective_potential(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9.6, 4.8))
    fig, ax0, _ = plot_potential_1d_x(potential, ax=ax[0])
    ax0.set_xlabel("")

    standard_operators = get_noise_operators(system, config, ty="standard")
    fig, _, line = plot_noise_operators_single_sample_x(
        standard_operators,
        ax=ax[1],
        truncation=15,
    )
    line.set_label("standard")

    linear_operators = get_noise_operators(system, config, ty="linear")
    fig, _, line = plot_noise_operators_single_sample_x(
        linear_operators,
        ax=ax[1],
    )
    line.set_label("linear")

    ax[1].legend()
    fig.show()

    input()
