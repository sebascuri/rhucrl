"""Utility functions for RHUCRL project."""
import importlib

from rllib.model import TransformedModel

from rhucrl.environment.wrappers import (
    AdversarialPendulumWrapper,
    MujocoAdversarialWrapper,
    NoisyActionRobustWrapper,
    ProbabilisticActionRobustWrapper,
)
from rhucrl.model import HallucinatedModel


def wrap_adversarial_environment(
    environment, wrapper_name, alpha, force_body_names=None
):
    """Wrap environment with an adversarial wrapper."""
    if wrapper_name == "noisy_action":
        environment.add_wrapper(NoisyActionRobustWrapper, alpha=alpha)
    elif wrapper_name == "probabilistic_action":
        environment.add_wrapper(ProbabilisticActionRobustWrapper, alpha=alpha)
    elif wrapper_name == "external_force":
        environment.add_wrapper(
            MujocoAdversarialWrapper, alpha=alpha, force_body_names=force_body_names
        )
    elif wrapper_name == "adversarial_pendulum":
        environment.add_wrapper(
            AdversarialPendulumWrapper, alpha=alpha, force_body_names=force_body_names
        )
    else:
        raise NotImplementedError(f"{wrapper_name} not implemented.")
    return environment


def get_default_model(
    environment,
    known_model=None,
    hallucinate=False,
    protagonist=True,
    weak_antagonist=False,
    strong_antagonist=False,
):
    """Get default protagonist/antagonist models."""
    if known_model is not None:
        if not isinstance(known_model, TransformedModel):
            dynamical_model = TransformedModel(known_model, [])
        else:
            dynamical_model = known_model
    else:
        if hallucinate:
            dynamical_model = HallucinatedModel.default(environment)
        else:
            dynamical_model = TransformedModel.default(environment)

    if protagonist:
        return dynamical_model
    elif weak_antagonist:
        return TransformedModel(
            dynamical_model.base_model, dynamical_model.forward_transformations
        )
    elif strong_antagonist:
        return dynamical_model
    else:
        raise NotImplementedError


def get_agent(agent, environment, **kwargs):
    """Get default agent."""
    agent_module = importlib.import_module("rhucrl.agent")
    agent = getattr(agent_module, f"{agent}Agent").default(environment, **kwargs)
    return agent
