import numpy as np
from scipy.constants import Boltzmann
from surface_potential_analysis.kernel.kernel import (
    get_single_factorized_noise_operators,
    get_single_factorized_noise_operators_diagonal,
    truncate_diagonal_noise_kernel,
    truncate_noise_kernel,
)
from surface_potential_analysis.kernel.plot import (
    plot_diagonal_kernel,
    plot_diagonal_kernel_truncation_error,
    plot_kernel,
    plot_kernel_truncation_error,
)
from surface_potential_analysis.operator.operator import (
    apply_operator_to_state,
    as_operator,
)
from surface_potential_analysis.operator.operator_list import (
    select_operator,
    select_operator_diagonal,
)
from surface_potential_analysis.operator.plot import (
    plot_eigenstate_occupations_operator,
    plot_eigenvalues_operator,
)
from surface_potential_analysis.state_vector.plot import (
    plot_state_1d_x,
)

from .system import get_full_noise_kernel, get_hamiltonian, get_potential_noise_kernel


def plot_noise_kernel() -> None:
    kernel = get_potential_noise_kernel(21, 10 / Boltzmann)
    fig, _, _ = plot_diagonal_kernel(kernel)
    fig.show()
    input()


def plot_truncated_noise_kernel() -> None:
    kernel = get_potential_noise_kernel(21, 10 / Boltzmann)
    truncated = truncate_diagonal_noise_kernel(kernel, n=6)
    fig, _, _ = plot_diagonal_kernel(truncated)
    fig.show()

    truncated["data"] = truncated["data"] - kernel["data"]
    fig, _, _ = plot_diagonal_kernel(truncated)
    fig.show()
    input()


def plot_noise_kernel_truncation_error() -> None:
    kernel = get_potential_noise_kernel(21, 10 / Boltzmann)

    fig, _ = plot_diagonal_kernel_truncation_error(kernel, scale="linear")
    fig.show()
    input()


def plot_potential_noise_kernel_largest_operator() -> None:
    kernel = get_potential_noise_kernel(21, 10 / Boltzmann)
    operators = get_single_factorized_noise_operators_diagonal(kernel)

    initial_state = {
        "basis": operators["basis"][1][1],
        "data": np.zeros(operators["basis"][1][1].n),
    }

    initial_state["data"][:] = 1 / np.sqrt(initial_state["basis"].fundamental_n)
    for idx in range(6):
        operator = select_operator_diagonal(
            operators,
            idx=np.argsort(np.abs(operators["eigenvalue"]))[idx],
        )
        state = apply_operator_to_state(
            as_operator(operator),
            initial_state,
        )

        fig, ax, _ = plot_state_1d_x(state, measure="abs")
        _, _, _ = plot_state_1d_x(state, measure="real", ax=ax)
        _, _, _ = plot_state_1d_x(state, measure="imag", ax=ax)
        ax.legend()

        ax.set_title(f"{idx}, E={np.sort(np.abs(operators['eigenvalue']))[idx]}")
        fig.show()
    input()


def plot_truncated_full_noise_kernel() -> None:
    kernel = get_full_noise_kernel(21, 10 / Boltzmann)
    kernel = truncate_noise_kernel(kernel, n=kernel["basis"][0].n)
    truncated = truncate_noise_kernel(kernel, n=8)
    fig, _, _ = plot_kernel(truncated)
    fig.show()

    truncated["data"] = truncated["data"] - kernel["data"]
    fig, _, _ = plot_diagonal_kernel(truncated)
    fig.show()
    input()


def plot_full_noise_kernel_truncation_error() -> None:
    kernel = get_full_noise_kernel(21, 10 / Boltzmann)
    truncated = truncate_noise_kernel(kernel, n=kernel["basis"][0].n)
    fig, ax = plot_kernel_truncation_error(
        truncated,
        truncations=list(range(kernel["basis"][0].n - 40, kernel["basis"][0].n, 1)),
        scale="linear",
    )
    ax.set_ylim(0, 2e-5)
    fig.show()


def plot_state_boltzmann_occupation() -> None:
    hamiltonian = get_hamiltonian(21)
    fig, _, _ = plot_eigenstate_occupations_operator(hamiltonian, 10 / Boltzmann)
    fig.show()
    input()


def plot_full_noise_kernel_largest_operator() -> None:
    kernel = get_full_noise_kernel(21, 10 / Boltzmann)
    operators = get_single_factorized_noise_operators(kernel)
    operator = select_operator(
        operators,
        idx=np.argmax(np.abs(operators["eigenvalue"])),
    )

    fig, ax, _ = plot_eigenvalues_operator(operator, measure="real")
    _, _, _ = plot_eigenvalues_operator(operator, measure="imag", ax=ax)
    _, _, _ = plot_eigenvalues_operator(operator, measure="abs", ax=ax)
    fig.show()
