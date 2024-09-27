import numpy as np
from scipy.constants import Boltzmann  # type: ignore unknown

from caldeira_legget_examples.periodic.dynamics import PeriodicSimulationConfig
from caldeira_legget_examples.periodic.plot import (
    plot_rate_against_potential,
    plot_thermal_occupation,
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
    dt_ratio = 10
    simulation_config = PeriodicSimulationConfig(
        n=100000,  # 20000
        step=dt_ratio * 10,
        dt_ratio=dt_ratio,
        n_trajectories=3,
    )
    plot_thermal_occupation(system, system_config)

    t0 = get_dimensionless_temperature(system)
    energies = np.array([0, 1, 2]) * t0 * Boltzmann
    plot_rate_against_potential(
        system,
        system_config,
        simulation_config,
        energies,
    )
