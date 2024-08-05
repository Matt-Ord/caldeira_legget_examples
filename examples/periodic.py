from caldeira_legget_examples.periodic.dynamics import PeriodicSimulationConfig
from caldeira_legget_examples.periodic.plot import (
    plot_coherent_state,
    plot_gaussian_distribution,
    plot_kernel,
    plot_point_evolution,
    plot_stochastic_evolution,
    plot_stochastic_occupation,
)
from caldeira_legget_examples.periodic.system import (
    FREE_SYSTEM,
    PeriodicSystemConfig,
    get_dimensionless_temperature,
)

if __name__ == "__main__":
    system = FREE_SYSTEM
    system_config = PeriodicSystemConfig(
        shape=(5,),
        resolution=(21,),
        temperature=2 * get_dimensionless_temperature(system),
    )

    print(system.gamma)

    # takes about 30min
    # !dt_ratio, step, n = 500, 300, 204800
    # takes about 3.1h, 5.2h when 6 trajectories good for long timescale behavior
    # !dt_ratio, step, n = 500, 2500, 102400

    # n_trajectories, dt_ratio, step, n = 6, 500, 10000, 102400
    simulation_config = PeriodicSimulationConfig(20002, 5000, 500, 6, 1)
    # n_trajectories, dt_ratio, step, n = 6, 500, 2500, 5000
    # !dt_ratio, step, n = 500, 2500, 51200
    # !dt_ratio, step, n = 500, 2500, 640  # Small timestep, hopping dt = 3200
    # !dt_ratio, step, n = 80, 200, 640 # Large timestep, hopping dt = 1600
    # !dt_ratio, step, n = 500, 1250, 640 # Small timestep, no hopping dt = 1600

    plot_kernel(system, system_config)

    plot_coherent_state(system, system_config)

    plot_gaussian_distribution(
        system,
        system_config,
        simulation_config,
    )
    plot_point_evolution(
        system,
        system_config,
        simulation_config,
    )
    plot_stochastic_evolution(
        system,
        system_config,
        simulation_config,
    )
    plot_stochastic_occupation(
        system,
        system_config,
        simulation_config,
    )
