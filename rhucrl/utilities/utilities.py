"""Training utilities helpers."""
import numpy as np
from hucrl.environment.hallucination_wrapper import HallucinationWrapper
from rllib.util.rollout import rollout_episode

from rhucrl.environment.adversarial_environment import AdversarialEnv


def change_mass(environment, body_idx, new_mass):
    """Change mass of body index."""
    environment.env.model.body_mass[body_idx] = new_mass


def change_friction(environment, body_idx, new_friction):
    """Change friction of body index."""
    environment.env.model.geom_friction[body_idx] = new_friction


def evaluate_domain_shift(env_args, agent, seed, num_runs):
    """Evaluate agent on domain shift."""
    environment = AdversarialEnv(env_args["name"], seed=seed)
    if environment.dim_action[0] < agent.policy.dim_action[0]:
        environment.add_wrapper(HallucinationWrapper)
    mass_names = env_args.get("mass_names", [])
    friction_names = env_args.get("friction_names", [])
    size = len(mass_names) + len(friction_names)

    mass_names = {
        name: (
            environment.env.model.body_names.index(name),
            environment.env.model.body_mass[
                environment.env.model.body_names.index(name)
            ],
        )
        for name in mass_names
    }
    friction_names = {
        name: (
            environment.env.model.body_names.index(name),
            environment.env.model.geom_friction[
                environment.env.model.body_names.index(name)
            ],
        )
        for name in friction_names
    }
    if env_args["name"] == "MBSwimmer-v0":
        k = np.linspace(-0.5, 1, 7)
    else:
        k = np.linspace(-1, 1, num=11)
    for values in np.array(np.meshgrid(*np.tile(k, size).reshape(size, -1))).T.reshape(
        -1, size
    ):
        for i in range(num_runs):
            # Change mass.
            for i, (name, (idx, base_mass)) in enumerate(mass_names.items()):
                new_mass = base_mass * (1 + values[i] + 0.01)
                change_mass(environment, body_idx=idx, new_mass=new_mass)
                agent.logger.update(**{f"mass-change-{i}": new_mass})

            # Change friction.
            for i, (name, (idx, base_friction)) in enumerate(friction_names.items()):
                new_friction = base_friction * (1 + values[len(mass_names) + i] + 0.01)

                change_friction(environment, body_idx=idx, new_friction=new_friction)
                agent.logger.update(**{f"friction-change-{i}": new_friction})

            rollout_episode(
                agent=agent,
                environment=environment,
                max_steps=env_args["max_steps"],
                render=False,
            )
