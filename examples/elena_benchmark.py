from surface_potential_analysis.state_vector.plot import plot_average_isf_1d_x

from caldeira_legget_examples.periodic.dynamics import (
    PeriodicSimulationConfig,
    get_stochastic_evolution,
)
from caldeira_legget_examples.periodic.system import (
    SODIUM_COPPER_SYSTEM,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    system = SODIUM_COPPER_SYSTEM
    system.with_gamma(system.gamma / 4)
    system_config = PeriodicSystemConfig(
        shape=(3,),
        resolution=(150,),
        temperature=155,
        operator_truncation=(1, 2),
    )
    dt_ratio = 10
    simulation_config = PeriodicSimulationConfig(
        n=100000,  # 20000, 100000
        step=dt_ratio * 10,
        dt_ratio=dt_ratio,
        n_trajectories=3,
    )

    # plot_stochastic_evolution(
    #     system,
    #     system_config,
    #     simulation_config,
    # )

    states = get_stochastic_evolution(
        system,
        system_config,
        simulation_config,
    )
    scattered_k = 5 * 10**10

    fig, ax, line = plot_average_isf_1d_x(states, scattered_k)
    fig.show()
    input()
