import numpy as np
from scipy.constants import Boltzmann
from surface_potential_analysis.kernel.kernel import (
    get_single_factorized_noise_operators,
    get_single_factorized_noise_operators_diagonal,
    truncate_diagonal_noise_kernel,
)
from surface_potential_analysis.kernel.plot import (
    plot_diagonal_kernel,
    plot_diagonal_kernel_truncation_error,
)
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.operator.operator_list import (
    select_operator,
    select_operator_diagonal,
)
from surface_potential_analysis.operator.plot import (
    plot_diagonal_operator_along_diagonal,
    plot_eigenstate_occupations,
    plot_eigenvalues,
    plot_operator_2d,
    plot_operator_along_diagonal,
)

from caldeira_legget_examples.util import get_temperature_corrected_noise_operators

from .system import (
    get_hamiltonian,
    get_potential_noise_kernel,
)


def plot_noise_kernel() -> None:
    kernel = get_potential_noise_kernel(31, 3 / Boltzmann)
    fig, _, _ = plot_diagonal_kernel(kernel)
    fig.show()

    fig, _, _ = plot_diagonal_kernel(kernel, measure="real")
    fig.show()
    input()


def plot_truncated_noise_kernel() -> None:
    temperature = 3 / Boltzmann
    kernel = get_potential_noise_kernel(31, temperature)
    truncated = truncate_diagonal_noise_kernel(kernel, n=13)
    fig, _, _ = plot_diagonal_kernel(truncated)
    fig.show()

    truncated["data"] = truncated["data"] - kernel["data"]
    fig, _, _ = plot_diagonal_kernel(truncated)
    fig.show()
    input()


def plot_noise_kernel_truncation_error() -> None:
    temperature = 3 / Boltzmann
    kernel = get_potential_noise_kernel(31, temperature)

    fig, _ = plot_diagonal_kernel_truncation_error(kernel, scale="linear")
    fig.show()
    input()


def plot_potential_noise_kernel_largest_operator() -> None:
    kernel = get_potential_noise_kernel(61, 10 / Boltzmann)
    operators = get_single_factorized_noise_operators_diagonal(kernel)

    for idx in np.argsort(np.abs(operators["eigenvalue"]))[::-1][:6]:
        operator = select_operator_diagonal(
            operators,
            idx=idx,
        )

        fig, ax, _ = plot_diagonal_operator_along_diagonal(operator, measure="abs")
        _, _, _ = plot_diagonal_operator_along_diagonal(operator, measure="real", ax=ax)
        _, _, _ = plot_diagonal_operator_along_diagonal(operator, measure="imag", ax=ax)
        ax.legend()

        ax.set_title(f"{idx}, E={operators['eigenvalue'][idx]}")
        fig.show()
    input()


def plot_state_boltzmann_occupation() -> None:
    temperature = 3 / Boltzmann
    n_states = 31

    hamiltonian = get_hamiltonian(n_states)
    fig, _, _ = plot_eigenstate_occupations(hamiltonian, temperature)
    fig.show()
    input()


def plot_full_noise_kernel_largest_operator() -> None:
    pos_basis = get_hamiltonian(21)["basis"]
    kernel = get_potential_noise_kernel(61, 10 / Boltzmann)
    operators = get_single_factorized_noise_operators_diagonal(kernel)
    operators = get_single_factorized_noise_operators(kernel)
    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    for idx in args[:5]:
        operator = select_operator(operators, idx=idx)

        fig, ax, _ = plot_eigenvalues(operator, measure="real")
        _, _, _ = plot_eigenvalues(operator, measure="imag", ax=ax)
        _, _, _ = plot_eigenvalues(operator, measure="abs", ax=ax)
        fig.show()

        converted = convert_operator_to_basis(operator, pos_basis)

        fig, ax, _ = plot_operator_along_diagonal(converted, measure="abs")
        _, _, _ = plot_operator_along_diagonal(converted, measure="real", ax=ax)
        _, _, _ = plot_operator_along_diagonal(converted, measure="imag", ax=ax)
        ax.legend()

        fig.show()

        fig, _, _ = plot_operator_2d(converted)
        fig.show()

    input()


def plot_temperature_corrected_operator() -> None:
    temperature = 10 / Boltzmann
    size = 61
    kernel = get_potential_noise_kernel(61, temperature)
    operators = get_single_factorized_noise_operators_diagonal(kernel)

    hamiltonian = get_hamiltonian(size)
    corrected = get_temperature_corrected_noise_operators(
        hamiltonian,
        operators,
        temperature,
    )

    args = np.argsort(np.abs(corrected["eigenvalue"]))[::-1]

    for idx in args[:5]:
        operator = select_operator(corrected, idx=idx)

        fig, ax, _ = plot_eigenvalues(operator, measure="real")
        _, _, _ = plot_eigenvalues(operator, measure="imag", ax=ax)
        _, _, _ = plot_eigenvalues(operator, measure="abs", ax=ax)
        fig.show()

        converted = convert_operator_to_basis(operator, hamiltonian["basis"])

        fig, ax, _ = plot_operator_along_diagonal(converted, measure="abs")
        _, _, _ = plot_operator_along_diagonal(converted, measure="real", ax=ax)
        _, _, _ = plot_operator_along_diagonal(converted, measure="imag", ax=ax)
        ax.legend()

        fig.show()

        fig, _, _ = plot_operator_2d(converted)
        fig.show()

    input()
