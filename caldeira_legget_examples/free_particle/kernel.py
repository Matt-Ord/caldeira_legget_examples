import numpy as np
from scipy.constants import Boltzmann
from surface_potential_analysis.basis.stacked_basis import StackedBasis
from surface_potential_analysis.kernel.kernel import (
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
    plot_operator_2d,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)

from caldeira_legget_examples.util import get_noise_operators_sampled

from .system import (
    get_hamiltonian,
    get_noise_operators,
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


def plot_largest_collapse_operator() -> None:
    temperature = 3 / Boltzmann
    n_states = 31

    operators = get_noise_operators(n_states, temperature)
    operators = get_noise_operators_sampled(
        get_noise_operators(n_states * 2, temperature),
    )

    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    x_basis = stacked_basis_as_fundamental_position_basis(
        operators["basis"][1][0],
    )
    k_basis = stacked_basis_as_fundamental_momentum_basis(
        operators["basis"][1][0],
    )

    for idx in args[:3]:
        operator = select_operator(operators, idx=idx)

        # fig, ax, _ = plot_eigenvalues(operator, measure="real")
        # _, _, _ = plot_eigenvalues(operator, measure="imag", ax=ax)
        # _, _, _ = plot_eigenvalues(operator, measure="abs", ax=ax)
        # fig.show()

        operator_x = convert_operator_to_basis(
            operator,
            StackedBasis(x_basis, x_basis),
        )

        # fig, ax, _ = plot_operator_along_diagonal(operator_x, measure="abs")
        # _, _, _ = plot_operator_along_diagonal(operator_x, measure="real", ax=ax)
        # _, _, _ = plot_operator_along_diagonal(operator_x, measure="imag", ax=ax)
        # ax.legend()
        # fig.show()

        fig, ax, _ = plot_operator_2d(operator_x)
        ax.set_title("Operator in X")
        fig.show()

        operator_k = convert_operator_to_basis(
            operator,
            StackedBasis(k_basis, k_basis),
        )

        # fig, ax, _ = plot_operator_along_diagonal(operator_k, measure="abs")
        # _, _, _ = plot_operator_along_diagonal(operator_k, measure="real", ax=ax)
        # _, _, _ = plot_operator_along_diagonal(operator_k, measure="imag", ax=ax)
        # ax.legend()
        # fig.show()

        fig, ax, _ = plot_operator_2d(operator_k)
        ax.set_title("Operator in K")
        fig.show()
    input()
