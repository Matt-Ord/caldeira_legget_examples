from caldeira_legget_examples.periodic.dynamics import PeriodicSimulationConfig
from caldeira_legget_examples.periodic.plot import (
    plot_gaussian_distribution,
    plot_point_evolution,
)
from caldeira_legget_examples.periodic.system import (
    FREE_SYSTEM,
    PeriodicSystemConfig,
    get_dimensionless_temperature,
)

if __name__ == "__main__":
    system = FREE_SYSTEM
    system_config = PeriodicSystemConfig(
        shape=(3,),
        resolution=(21,),
        temperature=2 * get_dimensionless_temperature(system),
        operator_truncation=range(1, 3),
    )
    dt_ratio = 5
    simulation_config = PeriodicSimulationConfig(
        n=10008,
        step=dt_ratio * 100,
        dt_ratio=dt_ratio,
        n_trajectories=3,
    )
    plot_point_evolution(
        system,
        system_config,
        simulation_config,
    )
    plot_gaussian_distribution(system, system_config, simulation_config)
