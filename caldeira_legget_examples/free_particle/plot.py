from matplotlib import pyplot as plt
from surface_potential_analysis.state_vector.plot import (
    animate_state_over_list_1d_x,
    plot_state_1d_x,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    state_vector_list_into_iter,
)

from .dynamics import (
    get_coherent_evolution,
    get_coherent_evolution_decomposition,
    get_stochastic_evolution,
    get_stochastic_evolution_high_t,
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

    input()


def plot_stochastic_evolution_high_t() -> None:
    states = get_stochastic_evolution_high_t()

    fig, ax = plt.subplots()
    for state in state_vector_list_into_iter(states):
        plot_state_1d_x(state, ax=ax)
    fig.show()

    fig, _, _anim0 = animate_state_over_list_1d_x(states)
    fig.show()

    fig, ax, _anim1 = animate_state_over_list_1d_x(states, measure="real")
    _, _, _anim2 = animate_state_over_list_1d_x(states, measure="imag", ax=ax)
    fig.show()

    input()
