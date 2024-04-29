from matplotlib import pyplot as plt
from scipy.constants import Boltzmann
from surface_potential_analysis.operator.plot import plot_eigenstate_occupations
from surface_potential_analysis.state_vector.plot import (
    animate_all_band_occupations,
    animate_state_over_list_1d_k,
    animate_state_over_list_1d_x,
    plot_all_band_occupations,
    plot_state_1d_x,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    state_vector_list_into_iter,
)

from caldeira_legget_examples.free_particle.system import get_hamiltonian

from .dynamics import (
    get_coherent_evolution,
    get_coherent_evolution_decomposition,
    get_stochastic_evolution,
)


def plot_coherent_evolution() -> None:
    states = get_coherent_evolution()

    fig, ax = plt.subplots()
    for state in state_vector_list_into_iter(states):
        plot_state_1d_x(state, ax=ax)
    fig.show()

    fig, _, _anim_0 = animate_state_over_list_1d_x(states)
    fig.show()

    fig, _, _anim1 = animate_state_over_list_1d_x(states, measure="real")
    fig.show()

    input()


def plot_coherent_evolution_decomposition() -> None:
    states = get_coherent_evolution_decomposition()

    fig, ax = plt.subplots()
    for state in state_vector_list_into_iter(states):
        plot_state_1d_x(state, ax=ax)
    fig.show()

    fig, _, _anim_0 = animate_state_over_list_1d_x(states)
    fig.show()

    input()


def plot_stochastic_evolution() -> None:
    states = get_stochastic_evolution()

    fig, ax = plt.subplots()
    for state in state_vector_list_into_iter(states):
        plot_state_1d_x(state, ax=ax)
    fig.show()

    fig, _, _anim0 = animate_state_over_list_1d_x(states)
    fig.show()

    fig, ax, _anim3 = animate_state_over_list_1d_k(states)
    fig.show()

    input()


def plot_stochastic_occupation() -> None:
    states = get_stochastic_evolution()

    hamiltonian = get_hamiltonian(31)
    fig0, ax0 = plot_all_band_occupations(hamiltonian, states)

    fig1, ax1, _anim0 = animate_all_band_occupations(hamiltonian, states)

    for ax in [ax0, ax1]:
        _, _, line = plot_eigenstate_occupations(hamiltonian, 3 / Boltzmann, ax=ax)
        line.set_linestyle("--")
        line.set_label("Expected")

        ax.legend([line], ["Expected occupation"])

    fig0.show()
    fig1.show()
    input()
