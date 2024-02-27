from surface_potential_analysis.potential.plot import plot_potential_1d_x
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)
from surface_potential_analysis.state_vector.eigenvalue_list_plot import (
    plot_eigenstate_occupations,
)
from surface_potential_analysis.state_vector.plot import (
    plot_all_band_occupations,
    plot_state_1d_x,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    state_vector_list_into_iter,
)

from .dynamics import (
    get_coherent_evolution,
    get_coherent_evolution_decomposition,
    get_stochastic_evolution,
    get_stochastic_evolution_localized,
)
from .system import (
    get_extended_interpolated_potential,
    get_hamiltonian,
    get_system_eigenstates,
)


def plot_system_potential() -> None:
    """Plot the potential against position."""
    potential = get_extended_interpolated_potential((6,), (100,))
    fig, _, _ = plot_potential_1d_x(potential)
    fig.show()
    input()


def plot_system_eigenstates() -> None:
    """Plot the potential against position."""
    potential = get_extended_interpolated_potential((6,), (100,))
    fig, ax, _ = plot_potential_1d_x(potential)

    ax1 = ax.twinx()

    eigenstates = get_system_eigenstates((6,), (100,), subset_by_index=(0, 6))
    for _i, state in enumerate(state_vector_list_into_iter(eigenstates)):
        plot_state_1d_x(state, ax=ax1)

    fig.show()

    input()


def plot_coherent_evolution() -> None:
    potential = get_extended_interpolated_potential((6,), (100,))
    fig, ax, _ = plot_potential_1d_x(potential)

    result = get_coherent_evolution()

    ax1 = ax.twinx()
    for state in state_vector_list_into_iter(result):
        plot_state_1d_x(state, ax=ax1)
    fig.show()

    input()


def plot_coherent_evolution_decomposition() -> None:
    potential = get_extended_interpolated_potential((6,), (100,))
    fig, ax, _ = plot_potential_1d_x(potential)

    result = get_coherent_evolution_decomposition()

    ax1 = ax.twinx()
    for state in state_vector_list_into_iter(result):
        plot_state_1d_x(state, ax=ax1)
    fig.show()

    input()


def plot_stochastic_evolution() -> None:
    potential = get_extended_interpolated_potential((6,), (100,))
    fig, ax, _ = plot_potential_1d_x(potential)

    result = get_stochastic_evolution()

    ax1 = ax.twinx()
    for i, state in enumerate(state_vector_list_into_iter(result)):
        if i > 10 and i < 25:
            plot_state_1d_x(state, ax=ax1)
    fig.show()

    input()


def plot_stochastic_occupation() -> None:
    states = get_stochastic_evolution()
    hamiltonian = get_hamiltonian((6,), (100,))
    fig, ax = plot_all_band_occupations(hamiltonian, states)
    eigenstates = calculate_eigenvectors_hermitian(hamiltonian)
    _, _, line = plot_eigenstate_occupations(eigenstates, 155, ax=ax)
    line.set_linestyle("--")
    line.set_label("Expected")

    ax.legend([line], ["Expected occupation at 155K"])
    fig.show()
    input()


def plot_stochastic_evolution_localized() -> None:
    potential = get_extended_interpolated_potential((6,), (100,))
    fig, ax, _ = plot_potential_1d_x(potential)

    result = get_stochastic_evolution_localized()

    ax1 = ax.twinx()
    for _i, state in enumerate(state_vector_list_into_iter(result)):
        plot_state_1d_x(state, ax=ax1)
    fig.show()

    input()
