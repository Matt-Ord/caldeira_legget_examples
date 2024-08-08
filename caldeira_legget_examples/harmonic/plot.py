import matplotlib.animation
from scipy.constants import Boltzmann
from surface_potential_analysis.potential.plot import plot_potential_1d_x
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)
from surface_potential_analysis.state_vector.eigenvalue_list_plot import (
    plot_eigenstate_occupations,
)
from surface_potential_analysis.state_vector.plot import (
    animate_all_eigenstate_occupations,
    animate_state_over_list_1d_k,
    animate_state_over_list_1d_x,
    plot_all_eigenstate_occupations,
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
from .system import get_hamiltonian, get_potential


def plot_system_potential() -> None:
    """Plot the potential against position."""
    potential = get_potential(100)
    fig, _, _ = plot_potential_1d_x(potential)
    fig.show()
    input()


def plot_system_eigenstates() -> None:
    """Plot the potential against position."""
    potential = get_potential(100)
    fig, ax, _ = plot_potential_1d_x(potential)

    hamiltonian = get_hamiltonian(100)
    eigenstates = calculate_eigenvectors_hermitian(hamiltonian)

    ax1 = ax.twinx()
    for _i, state in enumerate(state_vector_list_into_iter(eigenstates)):
        plot_state_1d_x(state, ax=ax1)

    fig.show()

    input()


def plot_coherent_evolution() -> None:
    potential = get_potential(100)
    fig, ax, _ = plot_potential_1d_x(potential)

    states = get_coherent_evolution()

    ax1 = ax.twinx()
    for state in state_vector_list_into_iter(states):
        plot_state_1d_x(state, ax=ax1)
    fig.show()

    fig, _, anim = animate_state_over_list_1d_x(states)
    fig.show()

    writervideo = matplotlib.animation.FFMpegWriter(fps=20)
    anim.save("coherent.harmonic.gif", writer=writervideo)
    input()


def plot_coherent_evolution_decomposition() -> None:
    potential = get_potential(100)
    fig, ax, _ = plot_potential_1d_x(potential)

    states = get_coherent_evolution_decomposition()

    ax1 = ax.twinx()
    for state in state_vector_list_into_iter(states):
        plot_state_1d_x(state, ax=ax1)
    fig.show()

    fig, _, _anim0 = animate_state_over_list_1d_x(states)
    fig.show()

    input()


def plot_stochastic_evolution() -> None:
    potential = get_potential(100)
    fig, ax, _ = plot_potential_1d_x(potential)

    states = get_stochastic_evolution()

    ax1 = ax.twinx()
    for state in state_vector_list_into_iter(states):
        plot_state_1d_x(state, ax=ax1)
    fig.show()

    fig, _, _anim0 = animate_state_over_list_1d_x(states)
    fig.show()

    fig, _, _anim1 = animate_state_over_list_1d_k(states)
    fig.show()

    input()


def plot_stochastic_occupation() -> None:
    states = get_stochastic_evolution()
    hamiltonian = get_hamiltonian(200)
    fig0, ax0 = plot_all_eigenstate_occupations(hamiltonian, states)

    fig1, ax1, _anim0 = animate_all_eigenstate_occupations(hamiltonian, states)

    eigenstates = calculate_eigenvectors_hermitian(hamiltonian)

    for ax in [ax0, ax1]:
        _, _, line = plot_eigenstate_occupations(eigenstates, 100 / Boltzmann, ax=ax)
        line.set_linestyle("--")
        line.set_label("Expected")

        ax.legend([line], ["Expected occupation"])

    fig0.show()
    fig1.show()
    input()
