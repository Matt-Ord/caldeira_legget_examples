from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from matplotlib.animation import ArtistAnimation
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.util import get_displacements_nx
from surface_potential_analysis.kernel.sample import (
    diagonal_operator_list_from_diagonal_split,
    get_diagonal_split_noise_components,
    sample_noise_from_diagonal_operators_split,
)
from surface_potential_analysis.operator.plot import (
    animate_diagonal_operator_list_along_diagonal,
)

from caldeira_legget_examples.periodic.system import (
    FREE_SYSTEM,
    PeriodicSystem,
    PeriodicSystemConfig,
    get_dimensionless_temperature,
    get_noise_operators,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalPositionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        TupleBasis,
        TupleBasisWithLengthLike,
    )
    from surface_potential_analysis.operator.operator_list import DiagonalOperatorList

_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])


def _build_time_correllated_noise(
    independent_noise: DiagonalOperatorList[FundamentalBasis[int], _B0, _B1],
    time_correllation: np.ndarray[Any, Any],
) -> DiagonalOperatorList[FundamentalBasis[int], _B0, _B1]:
    noise_transformed = np.fft.fft(
        independent_noise["data"].reshape(independent_noise["basis"][0].n, -1),
        axis=0,
    )
    time_correllation_transformed = np.fft.fft(time_correllation)
    return {
        "basis": independent_noise["basis"],
        "data": np.fft.ifft(
            noise_transformed * time_correllation_transformed[:, np.newaxis],
            axis=0,
        ).ravel(),
    }


def _get_sampled_noise(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    n_samples: int,
) -> DiagonalOperatorList[
    TupleBasis[FundamentalBasis[int], FundamentalBasis[int]],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
]:
    full_operators = get_noise_operators(system, config)

    return sample_noise_from_diagonal_operators_split(
        full_operators,
        n_samples=n_samples,
    )


def _animate_periodic_noise_over_time() -> None:
    system = FREE_SYSTEM
    config = PeriodicSystemConfig(
        shape=(10,),
        resolution=(21,),
        temperature=2 * get_dimensionless_temperature(system),
        operator_truncation=range(1, 5),
        operator_type="linear",
    )

    noise_split = _get_sampled_noise(system, config, n_samples=40)
    noise = diagonal_operator_list_from_diagonal_split(noise_split)

    fig, ax, _anim1 = animate_diagonal_operator_list_along_diagonal(
        noise,
        measure="real",
    )
    ax.set_title("Completely Random Noise")  # type: ignore lib
    fig.show()

    correllation = np.exp(-(get_displacements_nx(noise["basis"][0])[0] ** 2) / 16)

    correllated_noise = _build_time_correllated_noise(noise, correllation)

    fig, _ax, _anim0 = animate_diagonal_operator_list_along_diagonal(
        correllated_noise,
        measure="real",
    )
    fig.show()

    animations = list[ArtistAnimation]()
    for n in get_diagonal_split_noise_components(noise_split):
        fig, _ax, _anim = animate_diagonal_operator_list_along_diagonal(
            _build_time_correllated_noise(n, correllation),
            measure="real",
        )
        animations.append(_anim)
        fig.show()

    input()


if __name__ == "__main__":
    _animate_periodic_noise_over_time()
