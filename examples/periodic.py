from caldeira_legget_examples.sodium_copper.plot import (
    plot_stochastic_evolution,
    plot_stochastic_occupation,
)
from caldeira_legget_examples.sodium_copper.system import (
    LITHIUM_COPPER_SYSTEM,
)

if __name__ == "__main__":
    system = LITHIUM_COPPER_SYSTEM
    size = (3,)
    resolution = (21,)

    dt_ratio, step, n = 500, 2500, 51200
    # dt_ratio, step, n = 500, 2500, 640  # Small timestep, hopping dt = 3200
    # dt_ratio, step, n = 80, 200, 640 # Large timestep, hopping dt = 1600
    # dt_ratio, step, n = 500, 1250, 640 # Small timestep, no hopping dt = 1600

    plot_stochastic_evolution(
        system,
        size,
        resolution,
        dt_ratio=dt_ratio,
        step=step,
        n=n,
    )
    plot_stochastic_occupation(
        system,
        size,
        resolution,
        dt_ratio=dt_ratio,
        step=step,
        n=n,
    )
