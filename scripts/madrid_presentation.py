from typing import Any, TypeVar, cast

import numpy as np
from matplotlib.animation import ArtistAnimation
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import StackedBasisWithVolumeLike
from surface_potential_analysis.kernel.sample import (
    get_diagonal_split_noise_components,
    sample_noise_from_diagonal_operators,
    sample_noise_from_diagonal_operators_split,
)
from surface_potential_analysis.operator.build import get_displacements_matrix_nx
from surface_potential_analysis.operator.plot import (
    animate_diagonal_operator_list_along_diagonal,
)
from surface_potential_analysis.state_vector.plot import (
    animate_state_over_list_1d_x,
    get_coherent_coordinates,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    StateVectorList,
    as_state_vector_list,
    get_state_along_axis,
)
from surface_potential_analysis.util.plot import get_figure

from caldeira_legget_examples.periodic.dynamics import (
    PeriodicSimulationConfig,
    get_stochastic_evolution,
)
from caldeira_legget_examples.periodic.plot import (
    get_coherent_state,
)
from caldeira_legget_examples.periodic.system import (
    FREE_SYSTEM,
    PeriodicSystemConfig,
    get_dimensionless_temperature,
    get_noise_operators,
)
from examples.noise import build_time_correllated_noise


def _stochastic_v_animation() -> None:
    system = FREE_SYSTEM
    config = PeriodicSystemConfig(
        shape=(10,),
        resolution=(21,),
        temperature=2 * get_dimensionless_temperature(system),
        operator_truncation=range(1, 5),
        operator_type="periodic",
    )

    full_operators = get_noise_operators(system, config)

    noise = sample_noise_from_diagonal_operators(
        full_operators,
        n_samples=400,
    )

    correllation = np.exp(
        -(get_displacements_matrix_nx(noise["basis"][0])[0] ** 2) / 16,
    )
    correllated_noise = build_time_correllated_noise(noise, correllation)

    fig, ax, _anim0 = animate_diagonal_operator_list_along_diagonal(
        correllated_noise,
        measure="real",
        periodic=True,
    )
    fig.show()
    ax.set_yticks([])  # type:ignore lib
    fig.set_size_inches((12, 4))
    fig.tight_layout()
    _anim0.save("out0.mp4")


def _normal_modes_animation() -> None:
    system = FREE_SYSTEM
    config = PeriodicSystemConfig(
        shape=(10,),
        resolution=(21,),
        temperature=2 * get_dimensionless_temperature(system),
        operator_truncation=range(1, 5),
        operator_type="periodic",
    )

    full_operators = get_noise_operators(system, config)
    n_frames = 400
    noise_split = sample_noise_from_diagonal_operators_split(
        full_operators,
        n_samples=n_frames,
    )

    correllation = np.exp(
        -(get_displacements_matrix_nx(noise_split["basis"][0][1])[0] ** 2) / 16,
    )

    fig, ax = get_figure(None)
    animations = list[ArtistAnimation]()
    for i, noise_i in enumerate(get_diagonal_split_noise_components(noise_split)):
        noise = build_time_correllated_noise(noise_i, correllation)
        noise["data"] -= i * 80
        _fig, _ax, _anim = animate_diagonal_operator_list_along_diagonal(
            noise,
            measure="real",
            ax=ax,
        )
        animations.append(_anim)

    fig.show()
    ax.set_yticks([])  # type:ignore lib
    fig.set_size_inches((12, 4))

    frames = [list[Artist]() for _ in range(n_frames)]
    for i, anim in enumerate(animations):
        for j, frame in enumerate(cast(list[list[Artist]], anim.frame_seq)):
            frames[j].extend(frame)
            for child in frame:
                if isinstance(child, Line2D):
                    child.set_color(f"C{i}")
                    child.set_linewidth(4)

    fig.tight_layout()
    saved_anim = ArtistAnimation(fig, frames)
    saved_anim.save("out1.mp4")


def _taylor_expansion_animation() -> None:
    system = FREE_SYSTEM
    config = PeriodicSystemConfig(
        shape=(10,),
        resolution=(21,),
        temperature=2 * get_dimensionless_temperature(system),
        operator_truncation=range(1, 11),
        operator_type="linear",
    )

    full_operators = get_noise_operators(system, config)

    noise = sample_noise_from_diagonal_operators(
        full_operators,
        n_samples=400,
    )

    correllation = np.exp(
        -(get_displacements_matrix_nx(noise["basis"][0])[0] ** 2) / 16,
    )
    correllated_noise = build_time_correllated_noise(noise, correllation)

    fig, ax, _anim0 = animate_diagonal_operator_list_along_diagonal(
        correllated_noise,
        measure="real",
        periodic=True,
    )
    fig.show()
    ax.set_yticks([])  # type: ignore lib
    fig.set_size_inches((12, 4))
    fig.tight_layout()
    _anim0.save("out2.mp4")


def _taylor_expanded_simulation_example() -> None:
    system = FREE_SYSTEM
    config = PeriodicSystemConfig(
        shape=(10,),
        resolution=(21,),
        temperature=2 * get_dimensionless_temperature(system),
        operator_truncation=range(1, 2),
        operator_type="linear",
    )

    simulation_config = PeriodicSimulationConfig(
        n=1000,
        step=50,
        dt_ratio=60,
        n_trajectories=1,
    )

    states = get_stochastic_evolution(system, config, simulation_config)

    fig, ax, _anim = animate_state_over_list_1d_x(
        get_state_along_axis(states, axes=(1,), idx=(0,)),
    )
    fig.show()
    ax.set_yticks([])  # type: ignore lib
    # fig.set_size_inches((12, 4))
    fig.tight_layout()
    _anim.save("out3.mp4", fps=30)
    input()


def _periodic_simulation_example() -> None:
    system = FREE_SYSTEM
    config = PeriodicSystemConfig(
        shape=(5,),
        resolution=(21,),
        temperature=2 * get_dimensionless_temperature(system),
        operator_truncation=range(1, 3),
        operator_type="periodic",
    )

    simulation_config = PeriodicSimulationConfig(
        n=1000,
        step=100,
        dt_ratio=10,
        n_trajectories=1,
    )

    states = get_stochastic_evolution(system, config, simulation_config)

    fig, _ax, _anim = animate_state_over_list_1d_x(
        get_state_along_axis(states, axes=(1,), idx=(0,)),
    )
    fig.show()

    # fig.set_size_inches((12, 4))
    fig.tight_layout()
    _anim.save("out4.mp4", fps=60)
    input()


_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])


def _get_all_coherent_states(
    states: StateVectorList[
        _B0,
        StackedBasisWithVolumeLike[Any, Any, Any],
    ],
) -> StateVectorList[
    _B0,
    StackedBasisWithVolumeLike[Any, Any, Any],
]:
    assert states["basis"][1].ndim == 1
    a = get_coherent_coordinates(states, axis=0)
    coherent_states = as_state_vector_list(
        [
            get_coherent_state(
                states["basis"][1],
                (np.real(x_0).item(),),
                (np.real(k_0).item(),),
                (np.real(sigma_0).item(),),
            )
            for (x_0, k_0, sigma_0) in zip(
                a["data"].reshape(3, -1)[0],
                a["data"].reshape(3, -1)[1],
                a["data"].reshape(3, -1)[2],
            )
        ],
    )
    return {
        "basis": states["basis"],
        "data": coherent_states["data"],
    }


def _gaussian_fit_example() -> None:
    system = FREE_SYSTEM
    config = PeriodicSystemConfig(
        shape=(5,),
        resolution=(21,),
        temperature=2 * get_dimensionless_temperature(system),
        operator_truncation=range(1, 3),
        operator_type="periodic",
    )
    n_frames = 500
    simulation_config = PeriodicSimulationConfig(
        n=n_frames,
        step=100,
        dt_ratio=10,
        n_trajectories=1,
    )

    animations = list[ArtistAnimation]()

    states = get_stochastic_evolution(system, config, simulation_config)

    fig, ax = get_figure(None)
    fig, _ax, _anim = animate_state_over_list_1d_x(
        get_state_along_axis(states, axes=(1,), idx=(0,)),
        ax=ax,
    )
    animations.append(_anim)
    # TODO: fix issue with this...
    fitted_states = _get_all_coherent_states(states)

    fig, _ax, _anim = animate_state_over_list_1d_x(
        get_state_along_axis(fitted_states, axes=(1,), idx=(0,)),
        ax=ax,
    )
    animations.append(_anim)

    fig.show()
    input()

    frames = [list[Artist]() for _ in range(n_frames)]
    for i, anim in enumerate(animations):
        for j, frame in enumerate(cast(list[list[Artist]], anim.frame_seq)):
            frames[j].extend(frame)
            for child in frame:
                if isinstance(child, Line2D):
                    child.set_color(f"C{i}")
                    if i == 1:
                        child.set_linestyle("--")

    fig.set_size_inches((12, 4))
    ax.set_yticks([])  # type: ignore lib
    fig.tight_layout()
    saved_anim = ArtistAnimation(fig, frames)
    saved_anim.save("out6.mp4", fps=30)


if __name__ == "__main__":
    # _stochastic_v_animation()
    # _normal_modes_animation()
    # _taylor_expansion_animation()
    # _taylor_expanded_simulation_example()
    # _periodic_simulation_example()
    _gaussian_fit_example()
    input()
