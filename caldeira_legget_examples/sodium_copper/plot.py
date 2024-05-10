import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis.stacked_basis import StackedBasis
from surface_potential_analysis.kernel.plot import (
    plot_diagonal_kernel,
    plot_diagonal_kernel_truncation_error,
)
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.operator.operator_list import select_operator
from surface_potential_analysis.operator.plot import (
    plot_operator_2d,
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
from surface_potential_analysis.state_vector.eigenvalue_list_plot import (
    plot_eigenstate_occupations,
)
from surface_potential_analysis.state_vector.plot import (
    animate_all_band_occupations,
    animate_state_over_list_1d_k,
    animate_state_over_list_1d_x,
    plot_all_band_occupations,
    plot_average_band_occupation,
    plot_state_1d_k,
    plot_state_1d_x,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    state_vector_list_into_iter,
)

from .dynamics import (
    get_coherent_evolution,
    get_coherent_evolution_decomposition,
    get_stochastic_evolution,
)
from .system import (
    get_extended_interpolated_potential,
    get_hamiltonian,
    get_noise_kernel,
    get_noise_operators,
)


def plot_system_potential() -> None:
    """Plot the potential against position."""
    potential = get_extended_interpolated_potential((5,), (81,))
    fig, _, _ = plot_potential_1d_x(potential)
    fig.show()
    input()


def plot_system_eigenstates() -> None:
    """Plot the potential against position."""
    potential = get_extended_interpolated_potential((3,), (31,))
    fig, ax, _ = plot_potential_1d_x(potential)

    hamiltonian = get_hamiltonian((3,), (31,))
    eigenstates = calculate_eigenvectors_hermitian(hamiltonian)

    ax1 = ax.twinx()
    fig2, ax2 = plt.subplots()
    for _i, state in enumerate(state_vector_list_into_iter(eigenstates)):
        plot_state_1d_x(state, ax=ax1)

        plot_state_1d_k(state, ax=ax2)

    fig.show()
    fig2.show()
    input()


def plot_coherent_evolution() -> None:
    potential = get_extended_interpolated_potential((5,), (81,))
    fig, ax, _ = plot_potential_1d_x(potential)

    states = get_coherent_evolution()

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


def plot_coherent_evolution_decomposition() -> None:
    potential = get_extended_interpolated_potential((5,), (81,))
    fig, ax, _ = plot_potential_1d_x(potential)

    states = get_coherent_evolution_decomposition()

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


def plot_stochastic_evolution() -> None:
    potential = get_extended_interpolated_potential((3,), (31,))
    fig, ax, _ = plot_potential_1d_x(potential)

    states = get_stochastic_evolution()

    ax1 = ax.twinx()
    for _i, state in enumerate(state_vector_list_into_iter(states)):
        plot_state_1d_x(state, ax=ax1)
    fig.show()

    fig, ax, _ = plot_potential_1d_x(potential)
    _, _, _anim0 = animate_state_over_list_1d_x(states, ax=ax.twinx())
    fig.show()

    fig, _, _anim1 = animate_state_over_list_1d_k(states)
    fig.show()

    input()


def plot_stochastic_occupation() -> None:
    states = get_stochastic_evolution()
    hamiltonian = get_hamiltonian((3,), (41,))

    fig0, ax0 = plot_all_band_occupations(hamiltonian, states)

    fig1, ax1, _ani = animate_all_band_occupations(hamiltonian, states)

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


def plot_thermal_occupation() -> None:
    hamiltonian = get_hamiltonian((2,), (25,))

    eigenstates = calculate_eigenvectors_hermitian(hamiltonian)

    fig, _, _ = plot_eigenstate_occupations(eigenstates, 150)

    fig.show()
    input()


def plot_kernel() -> None:
    kernel = get_noise_kernel((2,), (25,), 150)
    fig, _, _ = plot_diagonal_kernel(kernel)
    fig.show()

    fig, _ = plot_diagonal_kernel_truncation_error(kernel)
    fig.show()

    input()


def plot_hamiltonian_sparsity() -> None:
    hamiltonian = get_hamiltonian((8,), (81,))
    fig, _ = plot_operator_sparsity(hamiltonian)
    fig.show()
    input()


def plot_largest_collapse_operator() -> None:
    temperature = 155
    n_states = (41,)

    operators = get_noise_operators((3,), n_states, temperature)

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

    operator = get_hamiltonian((3,), n_states)
    operator_k = convert_operator_to_basis(
        operator,
        StackedBasis(k_basis, k_basis),
    )

    fig, ax, _ = plot_operator_2d(operator_k)
    ax.set_title("Hamiltonian in K")
    fig.show()

    input()


def plot_collapse_operator_sparsity_k() -> None:
    temperature = 155
    n_states = (41,)

    operators = get_noise_operators((3,), n_states, temperature)

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

    operator = get_hamiltonian((3,), n_states)
    operator_k = convert_operator_to_basis(
        operator,
        StackedBasis(k_basis, k_basis),
    )

    fig, ax, _ = plot_operator_2d(operator_k)
    ax.set_title("Hamiltonian in K")
    fig.show()

    input()
