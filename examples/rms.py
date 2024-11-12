import numpy as np
from scipy.constants import Boltzmann  # type: ignore unknown

from caldeira_legget_examples.periodic.dynamics import (
    PeriodicSimulationConfig,
)
from caldeira_legget_examples.periodic.plot import (
    plot_rate_against_potential,
)
from caldeira_legget_examples.periodic.system import (
    FREE_SYSTEM,
    PeriodicSystemConfig,
    get_dimensionless_temperature,
)

if __name__ == "__main__":
    system = FREE_SYSTEM
    system_config = PeriodicSystemConfig(
        shape=(4,),
        resolution=(21,),
        temperature=2 * get_dimensionless_temperature(system),
        operator_truncation=range(1, 3),
    )

    dt_ratio = 10  # 10
    simulation_config = PeriodicSimulationConfig(
        n=100000,  # 20000, 100000
        step=dt_ratio * 10,
        dt_ratio=dt_ratio,
        n_trajectories=3,
    )
    # plot_thermal_occupation(system, system_config)
    # fig, ax, line = plot_stochastic_occupation(
    #     system,
    #     system_config,
    #     simulation_config,
    # )
    # fig.show()
    # input()

    t0 = get_dimensionless_temperature(system)
    energies = np.array([0]) * t0 * Boltzmann
    plot_rate_against_potential(
        system,
        system_config,
        simulation_config,
        energies,
        labels=[
            r"$E_{barrier} = 0$",
            # r"$E_{barrier}= K_b T$",
            # r"$E_{barrier} = 2 K_b T$",
        ],
    )
