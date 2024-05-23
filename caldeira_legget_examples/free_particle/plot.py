from matplotlib import pyplot as plt
from scipy.constants import Boltzmann
from surface_potential_analysis.operator.plot import plot_eigenstate_occupations
from surface_potential_analysis.state_vector.plot import (
    animate_state_over_list_1d_k,
    animate_state_over_list_1d_x,
    plot_average_band_occupation,
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
    _, _, _anim4 = animate_state_over_list_1d_k(states, ax=ax, measure="real")
    fig.show()

    input()


def plot_stochastic_occupation() -> None:
    states = get_stochastic_evolution()

    hamiltonian = get_hamiltonian(41)
    fig2, ax2, _anim2 = plot_average_band_occupation(hamiltonian, states)

    for ax in [ax2]:
        _, _, line = plot_eigenstate_occupations(hamiltonian, 3 / Boltzmann, ax=ax)
        line.set_linestyle("--")

        ax.legend([line], ["Expected occupation"])

    fig2.show()
    input()
