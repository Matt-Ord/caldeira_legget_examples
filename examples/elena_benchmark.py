from caldeira_legget_examples.periodic.dynamics import PeriodicSimulationConfig
from caldeira_legget_examples.periodic.plot import (
    plot_stochastic_evolution,
)
from caldeira_legget_examples.periodic.system import (
    SODIUM_COPPER_SYSTEM,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    system = SODIUM_COPPER_SYSTEM
    system_config = PeriodicSystemConfig(
        shape=(27,),
        resolution=(150,),
        temperature=155,
        operator_truncation=(1, 2),
    )
    simulation_config = PeriodicSimulationConfig(
        n=5,
        step=1000,
        dt_ratio=500,
        n_trajectories=1,
    )

    plot_stochastic_evolution(
        system,
        system_config,
        simulation_config,
    )
