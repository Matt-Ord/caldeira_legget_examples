from caldeira_legget_examples.periodic.plot import (
    plot_gaussian_distribution,
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
    config = PeriodicSystemConfig(
        shape=(8,),
        resolution=(31,),
        n_states=(15,),
        temperature=get_dimensionless_temperature(system),
    )
    print(system.gamma)

    # takes about 30min
    # !dt_ratio, step, n = 500, 300, 204800
    # takes about 3.1h, 5.2h when 6 trajectories good for long timescale behavior
    # !dt_ratio, step, n = 500, 2500, 102400

    # n_trajectories, dt_ratio, step, n = 6, 500, 10000, 102400
    n_trajectories, dt_ratio, step, n = 6, 500, 5000, 20000
    # n_trajectories, dt_ratio, step, n = 6, 500, 2500, 5000
    # !dt_ratio, step, n = 500, 2500, 51200
    # !dt_ratio, step, n = 500, 2500, 640  # Small timestep, hopping dt = 3200
    # !dt_ratio, step, n = 80, 200, 640 # Large timestep, hopping dt = 1600
    # !dt_ratio, step, n = 500, 1250, 640 # Small timestep, no hopping dt = 1600

    # plot_kernel(system, config)
    # plot_effective_potential(system, config)

    plot_gaussian_distribution(
        system,
        config,
        dt_ratio=dt_ratio,
        step=step,
        n=n,
        n_trajectories=n_trajectories,
    )
    plot_point_evolution(
        system,
        config,
        dt_ratio=dt_ratio,
        step=step,
        n=n,
        n_trajectories=n_trajectories,
    )
    plot_stochastic_evolution(
        system,
        config,
        dt_ratio=dt_ratio,
        step=step,
        n=n,
        n_trajectories=n_trajectories,
    )
    plot_stochastic_occupation(
        system,
        config,
        dt_ratio=dt_ratio,
        step=step,
        n=n,
        n_trajectories=n_trajectories,
    )
