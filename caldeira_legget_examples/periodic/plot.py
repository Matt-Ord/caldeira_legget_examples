from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from scipy.constants import Boltzmann, hbar  # type: ignore unknown
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.explicit_basis import (
    ExplicitStackedBasisWithLength,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisWithVolumeLike,
    TupleBasis,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.kernel.build import (
    truncate_diagonal_noise_operator_list,
)
from surface_potential_analysis.kernel.conversion import (
    convert_noise_operator_list_to_basis,
)
from surface_potential_analysis.kernel.gaussian import (
    get_effective_gaussian_isotropic_noise_kernel,
)
from surface_potential_analysis.kernel.kernel import (
    as_diagonal_noise_operators_from_full,
    as_isotropic_kernel_from_diagonal,
    get_diagonal_kernel_from_operators,
)
from surface_potential_analysis.kernel.plot import (
    plot_diagonal_kernel_2d,
    plot_diagonal_kernel_truncation_error,
    plot_diagonal_noise_operators_eigenvalues,
    plot_isotropic_noise_kernel_1d_x,
    plot_noise_operators_single_sample_x,
)
from surface_potential_analysis.kernel.solve import (
    get_periodic_noise_operators_real_isotropic_fft,
)
from surface_potential_analysis.operator.build import (
    get_displacements_matrix_x_stacked,
)
from surface_potential_analysis.operator.conversion import (
    convert_operator_to_basis,
)
from surface_potential_analysis.operator.operations import (
    exp_operator,
    get_anti_commutator,
    scale_operator,
)
from surface_potential_analysis.operator.operator import (
    SingleBasisDiagonalOperator,
    add_operator,
    apply_operator_to_state,
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
    _get_average_k,
    _get_k_operator,
    _get_x_operator,
    animate_state_over_list_1d_k,
    animate_state_over_list_1d_x,
    get_average_displacements,
    plot_average_displacement_1d_x,
    plot_average_eigenstate_occupation,
    plot_averaged_occupation_1d_k,
    plot_k_distribution_1d,
    plot_periodic_averaged_occupation_1d_x,
    plot_periodic_x_distribution_1d,
    plot_spread_1d,
    plot_spread_against_k,
    plot_spread_against_x,
    plot_spread_distribution_1d,
    plot_state_1d_k,
    plot_state_1d_x,
)
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_average_value_list_against_time,
    plot_value_list_against_time,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    StateVectorList,
    as_state_vector_list,
    get_state_along_axis,
    state_vector_list_into_iter,
)
from surface_potential_analysis.util.plot import get_figure

from .dynamics import (
    PeriodicSimulationConfig,
    get_coherent_evolution,
    get_coherent_evolution_decomposition,
    get_langevin_evolution,
    get_stochastic_evolution,
)
from .system import (
    PeriodicSystem,
    PeriodicSystemConfig,
    get_basis,
    get_extended_interpolated_potential,
    get_hamiltonian,
    get_temperature_corrected_noise_operators,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        TransformedPositionBasis,
    )
    from surface_potential_analysis.basis.time_basis_like import BasisWithTimeLike
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.util.plot import Scale
    from surface_potential_analysis.util.util import Measure

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

    ax1 = cast(Axes, ax.twinx())
    fig2, ax2 = plt.subplots()  # type: ignore unknown
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

    ax1 = cast(Axes, ax.twinx())
    for state in state_vector_list_into_iter(states):
        plot_state_1d_x(state, ax=ax1)
    fig.show()

    fig, ax, _ = plot_potential_1d_x(potential)
    _, _, _anim0 = animate_state_over_list_1d_x(states, ax=cast(Axes, ax.twinx()))
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

    ax1 = cast(Axes, ax.twinx())
    for state in state_vector_list_into_iter(states):
        plot_state_1d_x(state, ax=ax1)
    fig.show()

    fig, ax, _ = plot_potential_1d_x(potential)
    _, _, _anim0 = animate_state_over_list_1d_x(states, ax=cast(Axes, ax.twinx()))
    fig.show()

    fig, _, _anim1 = animate_state_over_list_1d_k(states)
    fig.show()

    input()


def plot_stochastic_evolution(  # noqa: PLR0914
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    simulation_config: PeriodicSimulationConfig,
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
        simulation_config,
    )

    states = get_state_along_axis(all_states, axes=(1,), idx=(0,))

    ax1 = cast(Axes, ax.twinx())
    fig2, ax2 = get_figure(None)
    for _i, state in enumerate(state_vector_list_into_iter(states)):
        if _i < 500:  # noqa: PLR2004
            plot_state_1d_x(state, ax=ax1)
            plot_state_1d_k(state, ax=ax2)
    fig.show()
    fig2.show()

    input()

    fig, ax, _ = plot_potential_1d_x(potential)
    _, _, _anim0 = animate_state_over_list_1d_x(states, ax=cast(Axes, ax.twinx()))
    fig.show()

    ax.set_title("Plot of wavefunction in Position against Time")  # type: ignore unknown
    line0 = Line2D([0], [0], label="Abs Wavefunction", color="tab:blue")
    ax.legend(handles=[line0])  # type: ignore unknown

    fig, ax, _anim1 = animate_state_over_list_1d_k(states)
    fig, _, _anim2 = animate_state_over_list_1d_k(states, ax=ax, measure="real")
    fig, _, _anim3 = animate_state_over_list_1d_k(states, ax=ax, measure="imag")
    fig.show()

    seq_2 = list[list[Line2D]]()
    for frame in _anim2.frame_seq:
        line: Line2D = frame[0]  # type: ignore unknown
        line.set_color("tab:orange")  # type: ignore unknown
        line.set_linestyle("--")  # type: ignore unknown

        seq_2.append([line])

    _anim2.frame_seq = iter(seq_2)  # type: ignore unknown

    seq_3 = list[list[Line2D]]()
    for frame in _anim3.frame_seq:
        line: Line2D = frame[0]  # type: ignore unknown
        line.set_color("tab:green")  # type: ignore unknown
        line.set_linestyle("--")  # type: ignore unknown

        seq_3.append([line])

    _anim3.frame_seq = iter(seq_3)  # type: ignore unknown

    ax.set_title("Plot of Wavefunction in Momentum against Time")  # type: ignore unknown

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
    ax.legend(handles=[line1, line2, line3])  # type: ignore unknown

    input()


def get_free_displacement_rate(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> float:
    # Note factor of hbar**2 here...
    # I think this is due to conventions we use when defining gamma
    return 2 * Boltzmann * config.temperature / (system.gamma * hbar**2)


def plot_free_displacement_rate(  # noqa: PLR0913
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    times: BasisWithTimeLike[Any, Any],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D]:
    rate = get_free_displacement_rate(system, config)

    return plot_value_list_against_time(
        {"basis": times, "data": (times.times * rate).astype(np.complex128)},
        ax=ax,
        scale=scale,
        measure=measure,
    )


def plot_point_evolution(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    simulation_config: PeriodicSimulationConfig,
) -> None:
    states = get_stochastic_evolution(
        system,
        config,
        simulation_config,
    )

    fig, ax = plot_periodic_averaged_occupation_1d_x(states)
    ax.set_title("Plot of restored x against time")  # type: ignore unknown

    y_start = 3 * states["basis"][1][0].delta_x[0]
    ax.vlines(  # type: ignore unknown
        states["basis"][0][1].times[100],
        ymin=y_start,
        ymax=y_start + (0.33 * states["basis"][1][0].delta_x),
        colors="black",
    )
    ax.text(  # type: ignore unknown
        states["basis"][0][1].times[800],
        y_start,
        "Unit Cell Width",
    )
    fig.show()

    fig, ax = plot_averaged_occupation_1d_k(states)
    ax.set_title("Plot of k against time")  # type: ignore unknown
    fig.show()

    fig, ax = plot_spread_1d(states)
    ax.set_title("Plot of spread against time")  # type: ignore unknown
    fig.show()

    fig, ax, line0 = plot_average_displacement_1d_x(states)
    line0.set_label("Simulation")

    _, _, line1 = plot_free_displacement_rate(
        system,
        config,
        states["basis"][0][1],
        ax=ax,
    )
    line1.set_label("Classical Limit")
    ax.legend()  # type: ignore unknown

    fig.show()
    input()


def plot_gaussian_distribution(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    simulation_config: PeriodicSimulationConfig,
) -> None:
    states = get_stochastic_evolution(
        system,
        config,
        simulation_config,
    )

    fig, ax = plot_spread_distribution_1d(states)
    ax.set_title("Plot of spread distribution")  # type: ignore unknown
    fig.show()

    input()

    fig, ax = plot_spread_against_k(states)
    ax.set_title("Plot of spread vs k")  # type: ignore unknown
    fig.show()

    fig, ax = plot_spread_against_x(states)
    ax.set_title("Plot of spread vs x")  # type: ignore unknown
    fig.show()

    fig, ax = plot_k_distribution_1d(states)
    ax.set_title("Plot of k distribution")  # type: ignore unknown
    fig.show()

    fig, ax = plot_periodic_x_distribution_1d(states)
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    _, _, _ = plot_potential_1d_x(potential, ax=cast(Axes, ax.twinx()))
    ax.set_title("Plot of x distribution")  # type: ignore unknown
    fig.tight_layout()
    fig.show()

    input()


def plot_stochastic_occupation(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    simulation_config: PeriodicSimulationConfig,
) -> None:
    states = get_stochastic_evolution(
        system,
        config,
        simulation_config,
    )
    hamiltonian = get_hamiltonian(system, config)

    fig2, ax2, line = plot_average_eigenstate_occupation(hamiltonian, states)

    for ax in [ax2]:
        _, _, line = plot_eigenstate_occupations(hamiltonian, config.temperature, ax=ax)
        line.set_linestyle("--")
        line.set_label("Expected")

        ax.legend([line], ["Boltzmann occupation"])  # type: ignore unknown

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
    hamiltonian = get_hamiltonian(system, config)
    a = get_effective_gaussian_isotropic_noise_kernel(
        hamiltonian["basis"][0],
        system.eta,
        config.temperature,
    )
    b = get_periodic_noise_operators_real_isotropic_fft(a)
    kernel = get_diagonal_kernel_from_operators(b)
    fig, ax, _ = plot_diagonal_kernel_2d(kernel)
    ax.set_title("noise kernel from isotropic without T correction")  # type: ignore unknown
    fig.show()

    cos_kernel = as_isotropic_kernel_from_diagonal(
        get_diagonal_kernel_from_operators(b),
    )
    fig, ax, _ = plot_isotropic_noise_kernel_1d_x(
        cos_kernel,
        measure="abs",
    )
    fig, _, _ = plot_isotropic_noise_kernel_1d_x(
        cos_kernel,
        ax=ax,
        measure="real",
    )
    fig, _, _ = plot_isotropic_noise_kernel_1d_x(
        cos_kernel,
        ax=ax,
        measure="imag",
    )

    fig, _, line = plot_isotropic_noise_kernel_1d_x(a, ax=ax)
    line.set_label("actual")
    line.set_linestyle("--")
    ax.set_title("noise kernel as fitted")  # type: ignore unknown
    ax.legend()  # type: ignore unknown
    fig.show()

    operators = get_temperature_corrected_noise_operators(system, config)
    basis_x = stacked_basis_as_fundamental_position_basis(operators["basis"][1][0])
    converted = as_diagonal_noise_operators_from_full(
        convert_noise_operator_list_to_basis(
            operators,
            TupleBasis(basis_x, basis_x),
        ),
    )
    kernel = get_diagonal_kernel_from_operators(converted)
    fig, ax, _ = plot_diagonal_kernel_2d(kernel)
    ax.set_title("noise kernel as diagonal")  # type: ignore unknown
    fig.show()

    truncated = truncate_diagonal_noise_operator_list(converted, range(5))
    kernel_error = get_diagonal_kernel_from_operators(truncated)
    kernel_error["data"] -= kernel["data"]
    fig, ax, _ = plot_diagonal_kernel_2d(kernel_error)
    ax.set_title("truncated noise kernel as diagonal")  # type: ignore unknown
    fig.show()

    fig, ax, _ = plot_diagonal_kernel_truncation_error(kernel)
    ax.set_title("truncated noise kernel error")  # type: ignore unknown
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
    operators = get_temperature_corrected_noise_operators(system, config)

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
        ax.set_title("Operator in X")  # type: ignore unknown
        fig.show()

        operator_k = convert_operator_to_basis(
            operator,
            TupleBasis(k_basis, k_basis),
        )

        fig, ax, _ = plot_operator_2d(operator_k)
        ax.set_title("Operator in K")  # type: ignore unknown
        fig.show()

    operator = get_hamiltonian(system, config)
    operator_k = convert_operator_to_basis(
        operator,
        TupleBasis(k_basis, k_basis),
    )

    fig, ax, _ = plot_operator_2d(operator_k)
    ax.set_title("Hamiltonian in K")  # type: ignore unknown
    fig.show()

    input()


def plot_largest_collapse_operator_eigenvalues(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    operators = get_temperature_corrected_noise_operators(
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
    operators = get_temperature_corrected_noise_operators(
        system,
        config,
    )
    fig, _ax, _ = plot_diagonal_noise_operators_eigenvalues(operators)

    fig.show()

    input()


def plot_collapse_operator_1d(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    operators = get_temperature_corrected_noise_operators(
        system,
        config,
    )

    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    x_basis = stacked_basis_as_fundamental_position_basis(
        operators["basis"][1][0],
    )

    for idx in args[7:8]:
        operator = select_operator(operators, idx=idx)
        operator_x = convert_operator_to_basis(
            operator,
            TupleBasis(x_basis, x_basis),
        )

        fig, ax, line = plot_operator_along_diagonal(operator_x, measure="abs")
        line.set_label("standard operator")
        fig.show()

        ax.legend()  # type: ignore unknown
        fig.show()

    input()


def plot_collapse_operator_2d(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    operators = get_temperature_corrected_noise_operators(
        system,
        config,
    )
    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    k_basis = stacked_basis_as_fundamental_momentum_basis(
        operators["basis"][1][0],
    )

    for idx in args[:4]:
        operator = select_operator(operators, idx=idx)
        operator_k = convert_operator_to_basis(
            operator,
            TupleBasis(k_basis, k_basis),
        )

        fig, ax = plot_operator_sparsity(operator_k)
        ax.set_title("Operator sparsity in K")  # type: ignore unknown
        fig.show()

        fig, ax = plot_operator_diagonal_sparsity(operator_k)
        ax.set_title("Operator diagonal sparsity in K")  # type: ignore unknown
        fig.show()

        fig, ax, _ = plot_operator_2d(operator_k, measure="abs")
        ax.set_title(f"standard operator, idx={idx}")  # type: ignore unknown
        fig.show()

        fig, ax, _ = plot_operator_2d(operator, measure="abs")
        ax.set_title(f"standard operator original basis, idx={idx}")  # type: ignore unknown
        fig.show()

        ax.set_title(f"corrected operator, idx={idx}")  # type: ignore unknown
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

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9.6, 4.8))  # type: ignore unknown
    fig, ax0, _ = plot_potential_1d_x(potential, ax=ax[0])
    ax0.set_xlabel("")  # type: ignore unknown

    standard_operators = get_temperature_corrected_noise_operators(
        system,
        config,
    )
    fig, _, line = plot_noise_operators_single_sample_x(
        standard_operators,
        ax=ax[1],
        truncation=range(3),
    )
    line.set_label("standard")

    linear_config = config.with_operator_type("linear")
    linear_operators = get_temperature_corrected_noise_operators(
        system,
        linear_config,
    )
    fig, _, line = plot_noise_operators_single_sample_x(
        linear_operators,
        ax=ax[1],
    )
    line.set_label("linear")

    ax[1].legend()
    fig.show()

    input()


_SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike[Any, Any, Any])


def _get_squeeze_operator(basis: _SBV0) -> SingleBasisOperator[_SBV0]:
    x_operator = _get_x_operator(basis, 0)
    k_operator = _get_k_operator(basis, 0)
    return exp_operator(
        scale_operator(10 / 637, get_anti_commutator(x_operator, k_operator)),
    )


def get_squeeze_state(basis: _SBV0) -> StateVector[_SBV0]:
    operator = _get_squeeze_operator(basis)

    basis_k = stacked_basis_as_fundamental_position_basis(basis)
    data = np.zeros(basis_k.n, dtype=np.complex128)
    data[0] = 1
    return apply_operator_to_state(operator, {"basis": basis_k, "data": data})


def _get_displacement_operator(
    basis: _SBV0,
    alpha: complex,
) -> SingleBasisOperator[_SBV0]:
    x_operator = _get_x_operator(basis, 0)
    k_operator = _get_k_operator(basis, 0)
    return exp_operator(
        add_operator(
            scale_operator(-1j * np.conj(alpha), x_operator),
            scale_operator(1j * alpha, k_operator),
        ),
    )


def get_displacement_state(basis: _SBV0, alpha: complex) -> StateVector[_SBV0]:
    operator = _get_displacement_operator(basis, alpha)

    basis_k = stacked_basis_as_fundamental_position_basis(basis)
    data = np.zeros(basis_k.n, dtype=np.complex128)
    data[0] = 1
    return apply_operator_to_state(operator, {"basis": basis_k, "data": data})


def _get_coherent_state_generator(
    basis: _SBV0,
    x_0: tuple[float, ...],
    k_0: tuple[float, ...],
    sigma_0: tuple[float, ...],
) -> SingleBasisOperator[_SBV0]:
    basis_x = stacked_basis_as_fundamental_position_basis(basis)

    # x - x' - x_0
    displacements = get_displacements_matrix_x_stacked(basis, x_0)
    distance = np.linalg.norm(
        displacements["data"].reshape(displacements["basis"].shape)
        / np.array(sigma_0)[:, np.newaxis],
        axis=0,
    )

    # i k.(x - x')
    phi = (2 * np.pi) * np.einsum(  # type: ignore unknown lib type
        "ij,i->j",
        displacements["data"].reshape(displacements["basis"].shape),
        k_0,
    )

    data = np.exp(-1j * phi - np.square(distance) / 2)
    norm = np.sqrt(np.sum(np.square(np.abs(data))))

    return convert_operator_to_basis(
        {
            "basis": TupleBasis(basis_x, basis_x),
            "data": data / norm,
        },
        TupleBasis(basis, basis),
    )


def get_coherent_state(
    basis: _SBV0,
    x_0: tuple[float,],
    k_0: tuple[float,],
    sigma_0: tuple[float,],
) -> StateVector[_SBV0]:
    operator = _get_coherent_state_generator(basis, x_0, k_0, sigma_0)

    basis_k = stacked_basis_as_fundamental_position_basis(basis)
    data = np.zeros(basis_k.n, dtype=np.complex128)
    data[0] = 1
    return apply_operator_to_state(operator, {"basis": basis_k, "data": data})


def _get_coherent_states(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    x_0: tuple[np.ndarray[Any, np.dtype[np.float64]],],
    k_0: tuple[np.ndarray[Any, np.dtype[np.float64]],],
    sigma_0: tuple[np.ndarray[Any, np.dtype[np.float64]],],
) -> StateVectorList[
    FundamentalBasis[int],
    TupleBasisWithLengthLike[TransformedPositionBasis[int, int, Literal[1]]],
]:
    basis = get_basis(system, config)
    return as_state_vector_list(
        (
            get_coherent_state(basis, x, k, sigma)
            for (x, k, sigma) in zip(zip(*x_0), zip(*k_0), zip(*sigma_0))
        ),
    )


_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])


def _get_boltzmann_state_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[_B0],
    temperature: float,
    phase: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> StateVector[_B0]:
    boltzmann_distribution = np.exp(
        -hamiltonian["data"] / (2 * Boltzmann * temperature),
    )
    normalization = np.sqrt(sum(np.square(boltzmann_distribution)))
    boltzmann_state = boltzmann_distribution * np.exp(1j * phase) / normalization
    return {"basis": hamiltonian["basis"][0], "data": boltzmann_state}


def get_random_boltzmann_state(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> StateVector[ExplicitStackedBasisWithLength[Any, Any]]:
    """Generate a random Boltzmann state.

    Follows the formula described in eqn 5 in
    https://doi.org/10.48550/arXiv.2002.12035.


    Args:
    ----
        system (PeriodicSystem): system
        config (PeriodicSystemConfig): config
        temperature (float): temperature of the system

    Returns:
    -------
        StateVector[Any]: state with boltzmann distribution

    """
    hamiltonian = get_hamiltonian(system, config)
    eigenvectors = calculate_eigenvectors_hermitian(hamiltonian)
    basis = ExplicitStackedBasisWithLength(eigenvectors)
    return _get_boltzmann_state_from_hamiltonian(
        {"basis": TupleBasis(basis, basis), "data": eigenvectors["eigenvalue"]},
        config.temperature,
        np.zeros_like(eigenvectors["eigenvalue"], dtype=np.float64),
    )


def _get_coherent_thermal_state(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    x_0: tuple[float,],
    k_0: tuple[float,],
    sigma_0: tuple[float,],
) -> StateVector[
    TupleBasisWithLengthLike[TransformedPositionBasis[int, int, Literal[1]]]
]:
    basis = get_basis(system, config)
    operator = _get_coherent_state_generator(basis, x_0, k_0, sigma_0)

    state = get_random_boltzmann_state(system, config)
    return apply_operator_to_state(operator, state)


def _get_coherent_thermal_states(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    x_0: tuple[np.ndarray[Any, np.dtype[np.float64]],],
    k_0: tuple[np.ndarray[Any, np.dtype[np.float64]],],
    sigma_0: tuple[np.ndarray[Any, np.dtype[np.float64]],],
) -> StateVectorList[
    FundamentalBasis[int],
    TupleBasisWithLengthLike[TransformedPositionBasis[int, int, Literal[1]]],
]:
    return as_state_vector_list(
        (
            _get_coherent_thermal_state(system, config, x_0, k_0, s)
            for (x_0, k_0, s) in zip(zip(*x_0), zip(*k_0), zip(*sigma_0))
        ),
    )


def plot_coherent_state(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    basis = get_basis(system, config)

    coherent_state = get_coherent_state(basis, (0,), (360,), (9,))
    thermal_state = _get_coherent_thermal_state(system, config, (0,), (360,), (9,))

    fig, ax, line = plot_state_1d_x(coherent_state)
    line.set_label("coherent")
    _, _, line = plot_state_1d_x(thermal_state, ax=ax)
    line.set_label("thermal")
    ax.legend()  # type: ignore unknown
    fig.show()

    fig, ax, line = plot_state_1d_k(coherent_state)
    line.set_label("coherent")
    _, _, line = plot_state_1d_k(thermal_state, ax=ax)
    line.set_label("thermal")
    ax.legend()  # type: ignore unknown

    fig.show()

    fig, ax = get_figure(None)
    k0 = np.arange(0, 360, 120).astype(np.float64)
    for temperature in [config.temperature]:
        config = config.with_temperature(temperature)
        for sigma_0 in [1200 / 120, 1200 / 90, 1200 / 60, 1200 / 30]:
            thermal_states = _get_coherent_thermal_states(
                system,
                config,
                (np.zeros_like(k0),),
                (k0,),
                (sigma_0 * np.zeros_like(k0),),
            )

            thermal_k = _get_average_k(thermal_states, 0)
            coherent_states = _get_coherent_states(
                system,
                config,
                (np.zeros_like(k0),),
                (k0,),
                (sigma_0 * np.zeros_like(k0),),
            )

            coherent_k = _get_average_k(coherent_states, 0)

            line = ax.plot(coherent_k["data"], thermal_k["data"])  # type: ignore unknown

    ax.set_xlabel("Coherent $K_0$")  # type: ignore unknown
    ax.set_ylabel("Thermal $K_0$")  # type: ignore unknown
    fig.show()
    input()


def plot_rate_against_potential(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    simulation_config: PeriodicSimulationConfig,
    energies: np.ndarray[Any, np.dtype[np.float64]],
) -> None:
    fig, ax = get_figure(None)
    fig1, ax1 = get_figure(None)

    for energy in energies:
        modified_system = system.with_barrier_energy(energy)

        states = get_stochastic_evolution(modified_system, config, simulation_config)

        _, _, _ = plot_average_displacement_1d_x(states, ax=ax)

        classical_positions = get_langevin_evolution(
            system,
            config,
            simulation_config,
        )
        plot_average_value_list_against_time(
            get_average_displacements(classical_positions),
            ax=ax1,
        )

    states = get_stochastic_evolution(system, config, simulation_config)
    _, _, line1 = plot_free_displacement_rate(
        system,
        config,
        states["basis"][0][1],
        ax=ax,
    )
    line1.set_color("black")
    fig.show()

    _, _, line1 = plot_free_displacement_rate(
        system,
        config,
        states["basis"][0][1],
        ax=ax1,
    )
    line1.set_color("black")
    ax1.set_title("langevin")  # type: ignore lib
    fig1.show()
    input()
